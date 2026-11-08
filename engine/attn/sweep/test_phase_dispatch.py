"""CPU-only dispatch contract test; no CUDA runtime or GPU access.

The parent adapter owns phase metadata. Its pipe_enabled seam is stubbed here.
Only CUDA launch decorations are erased, so ordinary host functions record which
specialization the real variant launch dispatcher selects. This tests branching,
not CUDA launch execution (the parent integration build covers CUDA compilation).
"""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class PhaseDispatchTests(unittest.TestCase):
    def test_explicit_phase_and_off_fallback(self):
        preamble = r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstddef>
using cudaError_t=int;
using cudaStream_t=int;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=1;
const void* attribute_kernel=nullptr;
int attribute_smem=-1, launched=-1;
int cudaFuncSetAttribute(const void* k,int attr,int smem) {
    assert(attr==cudaFuncAttributeMaxDynamicSharedMemorySize);
    attribute_kernel=k; attribute_smem=smem; return 0;
}
struct dim3 { dim3(int,int) {} };
namespace ds41a {
enum Phase { unknown, prefill_v2, main, late };
struct AttnP { int TK, floor, n, heads; Phase phase; };
constexpr int AT_THREADS=128, AT_SMEM_CPA=100, AT_SMEM_PIPE=200, AT_SMEM_PIPE_SWZ=300;
template<bool floor,bool tk> void k_attn_cpa_t(AttnP) { launched=floor?2:tk?1:0; }
template<bool swz> void k_attn_cpa_pipe_q(AttnP) { launched=swz?4:3; }
}
namespace sweep_adapter {
inline bool pipe_enabled(const ds41a::AttnP& p) { return p.phase==ds41a::prefill_v2; }
}
'''
        bodies = []
        for variant in ("off", "pipe", "swz"):
            source = (HERE / "variants" / variant / "variant.cuh").read_text()
            source, erased = re.subn(r"<<<[^\n]*?>>>", "", source)
            self.assertEqual(erased, 3 if variant == "off" else 4)
            bodies.append(source.replace("namespace sweep_variant", f"namespace variant_{variant}"))
        main = r'''
int main() {
    unsigned cases=0;
    for (auto phase : {ds41a::unknown,ds41a::prefill_v2,ds41a::main,ds41a::late})
      for (int n : {1,8192}) for (int tk : {0,512}) for (int floor : {0,16}) {
        ds41a::AttnP p{tk,floor,n,16,phase};
        const bool eligible=tk && !floor && phase==ds41a::prefill_v2;
        const void* off_kernel=variant_off::kernel(p);
        const char* off_key=variant_off::ptxas_key(p);
        variant_off::launch(p,0); const int off_launch=launched;
        for (bool swz : {false,true}) {
            assert((swz?variant_swz::eligible(p):variant_pipe::eligible(p))==eligible);
            const void* kernel=swz?variant_swz::kernel(p):variant_pipe::kernel(p);
            const char* key=swz?variant_swz::ptxas_key(p):variant_pipe::ptxas_key(p);
            const size_t smem=swz?variant_swz::dynamic_smem(p):variant_pipe::dynamic_smem(p);
            const void* optimized=swz?(const void*)ds41a::k_attn_cpa_pipe_q<true>:(const void*)ds41a::k_attn_cpa_pipe_q<false>;
            assert(kernel==(eligible?optimized:off_kernel));
            assert(!strcmp(key,eligible?(swz?"k_attn_cpa_pipe_qILb1EE":"k_attn_cpa_pipe_qILb0EE"):off_key));
            assert(smem==size_t(eligible?(swz?ds41a::AT_SMEM_PIPE_SWZ:ds41a::AT_SMEM_PIPE):ds41a::AT_SMEM_CPA));
            assert((swz?variant_swz::init(p):variant_pipe::init(p))==0);
            assert(attribute_kernel==kernel && attribute_smem==int(smem));
            launched=-1;
            if(swz) variant_swz::launch(p,0); else variant_pipe::launch(p,0);
            assert(launched==(eligible?(swz?4:3):off_launch));
            if((phase==ds41a::main || phase==ds41a::late) && tk && !floor)
                assert(launched==1); // compressed OFF, never window-only OFF
            ++cases;
        }
      }
    printf("dispatch cases=%u phases=unknown,prefill_v2,MAIN,late status=PASS\n",cases);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dispatch.cpp"
            source.write_text("#include <initializer_list>\n" + preamble + "\n".join(bodies) + main)
            build = subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(root / "dispatch")],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(root / "dispatch")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("dispatch cases=64", result.stdout)
            print(result.stdout, end="", flush=True)

if __name__ == "__main__":
    unittest.main()
