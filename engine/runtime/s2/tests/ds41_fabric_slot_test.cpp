// Regression: the production DS41 serve fabric run table versus the exact
// eight E2E prompts. Host-only; no CUDA or GPU required.
#include "ds41_prefill_fabric.h"
#include "s2/ds41_prefill.h"
#include "s2/tests/e2e_c8_prompt_fixture.h"
#include "ds41_pack_layout.h"
#include <cstdio>
#include <numeric>
#include <vector>

static int fails = 0;
#define CHECK(x) do { if (!(x)) { ++fails; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); } } while (0)
int main() {
    // The production serve config: W2048, PF_TAIL128,
    // Engram1, RS1, MX48, HID5120, --top-p-cands 256, four ranks.
    constexpr int W=2048, TAIL=128, NL=40, NX=2, MX=48, HID=5120, NR=4, NC=83;
    constexpr size_t HEAD = size_t(MX) * (32 + 256 * 8 + 20 * 8); // glue::head_bytes_tk(MX,256,true)
    std::vector<int> ids(NL); std::iota(ids.begin(), ids.end(), 0);
    const int il = ds41::br::late_start(ids);
    CHECK(il == 21);
    const auto off = ds41::pf::fabric_runs(W,TAIL,il,NL,NX,MX,HID,NR,HEAD,true,false);
    const auto on  = ds41::pf::fabric_runs(W,TAIL,il,NL,NX,MX,HID,NR,HEAD,true,true);
    g6::FabricDev fo{}, f{}; fo.nr=f.nr=NR; fo.compact=f.compact=1;
    CHECK(g6::fab_slot_layout(off.cfg,off.cfg.ncoll,&fo));
    CHECK(g6::fab_slot_layout(on.cfg,on.cfg.ncoll,&f));
    // Explicit byte-equality contract for the C8PACK=0 table.
    CHECK(off.cfg.run1==42 && off.cfg.run2==81 && off.cfg.run3==83 && off.cfg.ncoll==125);
    CHECK(fo.stride==20972544 && fo.stride1==1310976 && fo.stride2==25166080 && fo.stride3==5243136);
    CHECK(off.cfg.slot_bytes==20972536 && off.cfg.slot_bytes1==1310720 && off.cfg.slot_bytes2==25165824 && off.cfg.slot_bytes3==5242880);
    const auto prompts=e2e_c8_prompts();
    int rows=0; for (const auto& p:prompts) rows+=int(p.size())-1;
    CHECK(prompts.size()==8 && rows==476);
    const size_t before=2*g6::fab_slot_base(fo,fo.ncoll)*(1+g6::fab_recv_slots(fo));
    const size_t after =2*g6::fab_slot_base(f,f.ncoll)*(1+g6::fab_recv_slots(f));
    printf("BOOT_TABLE OFF [0,42)=%zu [42,81)=%zu [81,83)=%zu [83,125)=%zu\n",fo.stride,fo.stride1,fo.stride2,fo.stride3);
    printf("BOOT_TABLE ON  [0,42)=%zu [42,81)=%zu [81,83)=%zu [83,125)=%zu\n",f.stride,f.stride1,f.stride2,f.stride3);
    printf("FABRIC_BYTES_OFF=%zu ON=%zu DELTA=%zu COMPACT_RECV=%d\n",before,after,after-before,g6::fab_recv_slots(f));
    for (int R: {rows,ds41::pack::MAX_ROWS}) {
        const auto coll=g6s2::ds41_chunk_collectives(ids,NX,R,size_t(HID)*2,size_t(6144)*2,il,R,false);
        CHECK(coll.sizes.size()==NC);
        int bad_off=0,bad_on=0;
        for (size_t q=0;q<coll.sizes.size();++q) {
            const size_t need=size_t(coll.sizes[q])+8, old=g6::fab_slot_stride(fo,int(q)), now=g6::fab_slot_stride(f,int(q));
            const bool fits=need<=now;
            printf("ROWS=%d q=%zu size_plus_8=%zu off_stride=%zu on_stride=%zu %s\n",R,q,need,old,now,fits?"OK":"TOO_SMALL");
            bad_off += need>old; bad_on += !fits;
        }
        CHECK(bad_off==38); // q42..79 are under-sized in the C8PACK=0 table (128-row late run).
        CHECK(bad_on==0);
        CHECK(ds41::rs::job_error(coll.sizes,NC,on.rs_k,R,NR,size_t(HID)*2)==nullptr);
        const auto j=ds41::rs::job(coll.sizes,coll.order,NC,on.rs_k,R,NR,size_t(HID)*2);
        const size_t direct=size_t(ds41::rs::quarter(R,NR))*HID*2*NR+8*NR;
        printf("ROWS=%d RS_BYTES=%zu DIRECT_BYTES=%zu RUN0_STRIDE=%zu AG_NEED=%zu AG_STRIDE=%zu ENGRAM_NEED=%zu ENGRAM_STRIDE=%zu\n",
               R,j.ss*NR,direct,g6::fab_slot_stride(f,0),size_t(j.sizes[NC])+8,g6::fab_slot_stride(f,NC),
               size_t(coll.sizes[81])+8,g6::fab_slot_stride(f,81));
        for (int q:j.rs) {
            CHECK(j.ss*NR<=g6::fab_slot_stride(f,q));
            CHECK(direct<=g6::fab_slot_stride(f,q));
        }
        for (int q=0;q<on.rs_k;++q) CHECK(size_t(j.sizes[NC+q])+8<=g6::fab_slot_stride(f,NC+q));
        printf("ROWS=%d PRIMARY_BAD_OFF=%d PRIMARY_BAD_ON=%d RS_AG_DIRECT=CHECKED\n",R,bad_off,bad_on);
    }
    printf("ds41_fabric_slot_test: %s\n",fails?"FAIL":"PASS");
    return fails?1:0;
}
