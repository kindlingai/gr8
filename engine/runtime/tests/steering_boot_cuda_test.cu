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
    CK(cudaMalloc(&m.steering_slot,G6_MX*sizeof(int))); CK(cudaMemsetAsync(m.steering_slot,0xff,G6_MX*sizeof(int),st));
}
static void cleanup(g6rt::Model& m) {
    if(!m.steering_dir.empty()) CK(cudaFree(const_cast<float*>(m.steering_dir[0])));
    if(m.steering_part) CK(cudaFree(m.steering_part));
    if(m.steering_count) CK(cudaFree(m.steering_count));
    CK(cudaFree(m.steering_scale));
    CK(cudaFree(m.steering_slot));
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
    {   // two profiles: A steers layers 0 and 3, B layers 1 and 3; each steered layer holds a [2][HID] table
        Model m; initialize(m,st); const auto a=profile(steering::Scope::Full); auto b=a;
        b.name="b"; b.layers={1,3}; b.alphas={2.f,3.f}; b.directions.assign(2*HID,0.f); b.directions[2]=1.f; b.directions[HID+3]=1.f;
        materialize_steering(m,std::vector<steering::Profile>{a,b},st);
        check(m.steering_profiles==2 && m.steering_dir[0] && m.steering_dir[1] && m.steering_dir[2],"a layer either profile steers has a table");
        check(m.steering_alphas==std::vector<float>({.5f,0.f, 0.f,2.f, 1.f,3.f}),"[layer][profile] alphas, 0 where a profile leaves a layer out");
        check(m.steering_alpha==std::vector<float>({.5f,0.f,1.f}),"profile 0's alphas");
        std::vector<float> l0(2*HID),l1(2*HID),l3(2*HID),slot_alpha(3*2);
        CK(cudaMemcpy(l0.data(),m.steering_dir[0],l0.size()*sizeof(float),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(l1.data(),m.steering_dir[1],l1.size()*sizeof(float),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(l3.data(),m.steering_dir[2],l3.size()*sizeof(float),cudaMemcpyDeviceToHost));
        check(l0[0]==1.f && std::count(l0.begin(),l0.end(),0.f)==long(l0.size())-1,"layer 0: A's direction, B's row zero");
        check(l1[HID+2]==1.f && std::count(l1.begin(),l1.end(),0.f)==long(l1.size())-1,"layer 1: A's row zero, B's direction");
        check(l3[1]==1.f && l3[HID+3]==1.f,"layer 3: both directions in slot order");
        check(m.steering_slot_alpha.size()==3,"one device alpha table per layer");
        for(int i=0;i<3;++i) CK(cudaMemcpy(slot_alpha.data()+2*i,m.steering_slot_alpha[i],2*sizeof(float),cudaMemcpyDeviceToHost));
        check(slot_alpha==m.steering_alphas,"device alpha tables match the host's");
        std::vector<int> slots(MX); CK(cudaMemcpy(slots.data(),m.steering_slot,MX*sizeof(int),cudaMemcpyDeviceToHost));
        check(std::all_of(slots.begin(),slots.end(),[](int x){return x==0;}),"row slots start at 0");
        cleanup(m);
        Model n; initialize(n,st); auto shared=b; shared.scope=steering::Scope::Shared;
        rejects([&]{materialize_steering(n,std::vector<steering::Profile>{a,shared},st);});
        cleanup(n);
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
