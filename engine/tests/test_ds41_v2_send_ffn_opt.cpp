// Selection, slot reuse, interleaving and exact log contracts.
#include "../runtime/ds41_v2_send_ffn_opt.h"
#include <cassert>
#include <climits>
#include <string>
using ds41::v2::SendFfnOpt;
static std::string contents(FILE* f) {
    fflush(f); rewind(f); std::string result; char buffer[256];
    while(fgets(buffer,sizeof(buffer),f))result+=buffer;
    fseek(f,0,SEEK_END); return result;
}
int main() {
    SendFfnOpt selector; std::vector<int> decoded;
    assert(!SendFfnOpt::decode({},decoded));
    for(auto bad:std::vector<int64_t>{-1,3+1,INT64_MAX,INT64_MIN})
        assert(!SendFfnOpt::decode({bad},decoded));
    FILE* log=tmpfile(); assert(log);
    assert(selector.init({0}));
    for(int request=0;request<30;++request)selector.select(request,request%3,2,log);
    assert(contents(log).empty());
    std::vector<int64_t> modes; for(int mode=0;mode<=3;++mode)modes.push_back(mode);
    assert(selector.init(modes)); assert(selector.seen.empty());
    std::string want;
    for(unsigned request=0;request<24;++request) {
        const unsigned slot=request%3; const int mode=request%(3+1);
        selector.select(request,slot,2,log);
        assert(selector.mode==mode && selector.attn_opt()==bool(mode&1) && selector.ffn_opt()==bool(mode&2));
        char line[256];
        if(mode) { snprintf(line,sizeof(line),"# ds41_v2_send_ffn_opt rank=2 request=%llu slot=%u on=1 mode=%d\n",(unsigned long long)request,slot,mode); want+=line; }
        selector.select(request,slot,2,log); // repeated chunk must not log again
        if(request>0)selector.select(request-1,(request-1)%3,2,log); // interleaved active slot
    }
    assert(contents(log)==want); assert(selector.seen.size()==3);
    assert(!selector.init({-1})); assert(selector.seen.empty());
    assert(selector.init({3})); assert(selector.seen.empty());
    selector.select(UINT64_MAX,9,1,log); selector.select(UINT64_MAX,9,1,log);
    assert(selector.seen.size()==1);
    fclose(log); puts("test_ds41_v2_send_ffn_opt: PASS (selection/logging/interleaving/slot reuse)");
}
