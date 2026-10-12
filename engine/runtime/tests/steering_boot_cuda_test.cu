// Small device-materialization checks, no weights, attention bindings, fabric or serving process.
#include "../steering_boot.cuh"
#include <cstdio>
#include <cstdlib>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); std::exit(2); } } while(0)
static int checks=0,failed=0;
static void check(bool value,const char* why) { ++checks; if(!value) { ++failed; std::printf("FAIL %s\n",why); } }
template<class F> void rejects(F fn) { bool threw=false; try { fn(); } catch(const std::exception&) { threw=true; } check(threw,"invalid activation fails before materialization"); }
static g6rt::steering::Profile profile(g6rt::steering::Scope scope) {
    g6rt::steering::Profile p; p.hidden_size=HID; p.layer_count=78; p.scope=scope;
    p.layers={0,3}; p.alphas={.5f,1.f}; p.directions.assign(2*HID,0.f); p.directions[0]=1.f; p.directions[HID+1]=1.f;
    return p;
}
static void initialize(g6rt::Model& m,cudaStream_t st) {
    m.nl=3; m.layers.resize(3); m.layers[0].l=0; m.layers[1].l=1; m.layers[2].l=3;
    CK(cudaMalloc(&m.steering_scale,G6_MX*sizeof(float))); CK(cudaMemsetAsync(m.steering_scale,0xff,G6_MX*sizeof(float),st));
}
static void cleanup(g6rt::Model& m) {
    if(!m.steering_dir.empty()) CK(cudaFree(const_cast<float*>(m.steering_dir[0])));
    if(m.steering_part) CK(cudaFree(m.steering_part));
    if(m.steering_count) CK(cudaFree(m.steering_count));
    CK(cudaFree(m.steering_scale));
}
int main() {
    using namespace g6rt;
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    for(auto scope:{steering::Scope::Full,steering::Scope::Shared}) {
        Model m; initialize(m,st); const auto p=profile(scope);
        materialize_steering(m,p,st); CK(cudaStreamSynchronize(st));
        check(m.steering_scope==scope && m.steering_dir.size()==3,"immutable profile scope and layer map");
        check(m.steering_dir[0] && !m.steering_dir[1] && m.steering_dir[2]==m.steering_dir[0]+HID,"subset maps to actual runtime layer IDs");
        check(m.steering_alpha==std::vector<float>({.5f,0.f,1.f}),"unscheduled layers remain inactive");
        std::vector<float> values(2*HID),scales(MX);
        CK(cudaMemcpy(values.data(),m.steering_dir[0],values.size()*sizeof(float),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(scales.data(),m.steering_scale,G6_MX*sizeof(float),cudaMemcpyDeviceToHost));
        check(values==p.directions,"verified direction bytes copied exactly");
        check(std::all_of(scales.begin(),scales.end(),[](float x){return x==0.f && !std::signbit(x);}),"initial warm/capture rows use positive zero");
        if(scope==steering::Scope::Shared) {
            std::vector<unsigned> counts(MX); CK(cudaMemcpy(counts.data(),m.steering_count,MX*sizeof(unsigned),cudaMemcpyDeviceToHost));
            check(m.steering_part && std::all_of(counts.begin(),counts.end(),[](unsigned x){return x==0;}),"shared counters allocated and zeroed at boot");
        } else check(!m.steering_part && !m.steering_count,"full scope needs no shared scratch");
        rejects([&]{materialize_steering(m,p,st);});
        cleanup(m);
    }
    for(int invalid=0;invalid<3;++invalid) {
        Model m; initialize(m,st); auto p=profile(steering::Scope::Full);
        if(invalid==0) p.layers[1]=4;
        if(invalid==1) p.directions.pop_back();
        if(invalid==2) m.layers[0].built[1]=true;
        rejects([&]{materialize_steering(m,p,st);});
        check(m.steering_dir.empty() && !m.steering_count,"refusal leaves device profile state empty");
        cleanup(m);
    }
    CK(cudaStreamDestroy(st));
    std::printf("steering_boot_cuda_test: %d checks, %d failures\n",checks,failed);
    return failed ? 1:0;
}
