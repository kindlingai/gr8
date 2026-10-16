// Real pilot-release C ABI gate. Usage: partial_plan_test RELEASE OUTPUT_DIR
// RELEASE has dense/, experts/, index-slices/, manifests/exlr8-manifest-k4.json.
// Build with g++ -std=c++17 partial_plan_test.cpp ../target/release/libg6load.a -ldl -lpthread -lm.
#include "../g6load.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
static void require(bool ok, const char* msg) { if(!ok) { fprintf(stderr,"FAIL: %s\n",msg); exit(1); } }
int main(int argc,char** argv) {
    require(argc==3,"usage RELEASE OUTPUT_DIR");
    const std::string root=argv[1], out=std::string(argv[2])+"/selected.manifest.json";
    const std::string experts=root+"/experts", slices=root+"/index-slices", widths=root+"/manifests/exlr8-manifest-k4.json";
    char err[2048]={}; const uint32_t layers[]={0,1,2,3};
    require(!g6l_release_plan_selected(root.c_str(),experts.c_str(),slices.c_str(),widths.c_str(),4,0,0,1,
                                      nullptr,0,out.c_str(),err,sizeof err),"empty subset must fail");
    require(std::string(err).find("nonempty")!=std::string::npos,"empty-subset error");
    for(int rank=0;rank<4;++rank) {
        void* h=g6l_release_plan_selected(root.c_str(),experts.c_str(),slices.c_str(),widths.c_str(),4,rank,0,1,
                                         layers,4,out.c_str(),err,sizeof err);
        require(h,err);
        std::ifstream f(out);const std::string text((std::istreambuf_iterator<char>(f)),{});
        for(const char* key:{"L000.","L001.","L002.","L003.","G.embed","G.lm_head.gv","exlr8-e4m3"})
            require(text.find(key)!=std::string::npos,key);
        require(text.find("L004.")==std::string::npos,"unselected layer in image");
        unsigned char bytes[64];
        require(g6l_plan_read(h,0,sizeof bytes,bytes,err,sizeof err)==0,err);
        g6l_plan_free(h);
        printf("PASS: selected C ABI rank%d, layers0-3 + globals, payload read\n",rank);
    }
    void* h=g6l_release_plan(root.c_str(),experts.c_str(),slices.c_str(),widths.c_str(),4,0,0,1,out.c_str(),err,sizeof err);
    require(!h,"the incomplete pilot must not pass the full-model planner");
    printf("PASS: normal boot rejects incomplete pilot: %s\n",err);
    std::remove(out.c_str());
}
