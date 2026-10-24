"""CPU-only regression tests for the experimental P1-only DSA512 route."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "engine/attn/flash_attn.cu"
P1_SHA = "a58b316a05e3ae324a9863aaf1ec7c785eccd1f2f5e7256b3a939ed4fe8d2881"
MICRO_KERNEL_SHA = "9fc8bd1281d54521beb6d74bc53d76e3855785e97e0b870c01e0ac598b618478"
MARK = "// Opt-in 512-thread P1 variant: identical per-output math, four PV fragments/warp.\n"
START = "template<class Input>\n__global__ void __launch_bounds__(512) sparse_mla_pf_512("
END = "// Tensor-Core value_project:"

def kernel(source, start=START):
    return source[source.index(start):source.index(END, source.index(start))].rstrip()

def strip_h16c(s):
    """Recover the pre-H16c V2 dispatcher before checking the pinned P1 source."""
    s = s.replace('    // Separate exact H16c arm. Unset keeps the existing H32 V2 instruction path.\n'
                  '    const char* h16c_env=getenv("G6_PF_SPARSE_MLA_H16C");\n'
                  '    const bool fused_mla_h16c=fused_mla_v2&&a.heads<=16&&h16c_env&&atoi(h16c_env)==1;\n', '')
    s = s.replace('''            ce=fused_mla_h16c
                ?cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn,16>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM_H16)
                :cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM);''',
                  '''            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM);''')
    s = s.replace('''            if(fused_mla_v2){
                if(fused_mla_h16c)fa_pf::sparse_mla_pf_v2<G6Attn,16><<<n,512,fa_pf::MLA2_SMEM_H16,stream>>>(a,start,w->qh,w->ctxh);
                else fa_pf::sparse_mla_pf_v2<<<n,512,fa_pf::MLA2_SMEM,stream>>>(a,start,w->qh,w->ctxh);
            }''',
                  '''            if(fused_mla_v2)fa_pf::sparse_mla_pf_v2<<<n,512,fa_pf::MLA2_SMEM,stream>>>(a,start,w->qh,w->ctxh);''')
    return s

def strip_v2(s):
    """The source without the #10 v2 kernel block and its nested dispatch (= the pre-v2 P1 source)."""
    s = strip_h16c(s)
    end = "// ==== END #10 v2 ====\n\n"
    s = s[:s.index("// ==== BEGIN #10 v2")] + s[s.index(end) + len(end):]
    s = s.replace('    // V2 defaults on only with sparse MLA + _512 and heads<=32. Explicit _V2=0 restores _512;\n'
                  '    // explicit _V2=1 retains the original opt-in outside _512 (still within sparse MLA).\n', '')
    s = s.replace('    const char* mla_v2_env=getenv("G6_PF_SPARSE_MLA_V2");\n'
                  '    const bool fused_mla_v2=fused_mla&&(mla_v2_env?atoi(mla_v2_env)==1:fused_mla512);\n', '')
    s = s.replace("        if(fused_mla_v2)\n"
                  "            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM);\n"
                  "        else if(fused_mla512)", "        if(fused_mla512)")
    s = s.replace("            if(fused_mla_v2)fa_pf::sparse_mla_pf_v2<<<n,512,fa_pf::MLA2_SMEM,stream>>>(a,start,w->qh,w->ctxh);\n"
                  "            else if(fused_mla512)", "            if(fused_mla512)")
    return s

def tiles(warps):
    return [(w//(warps//2)*16, w%(warps//2)*(512//(warps//2))+j*16)
            for w in range(warps) for j in range(64//warps)]

class DSA512Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = SOURCE.read_text()

    def test_candidate_is_byte_identical_to_tested_micro_kernel(self):
        self.assertEqual(hashlib.sha256(kernel(self.s).encode()).hexdigest(), MICRO_KERNEL_SHA)

    def test_baseline_kernel_restored_by_only_mapping_changes(self):
        k = kernel(self.s)
        changes = [
            ("__launch_bounds__(512) sparse_mla_pf_512(", "__launch_bounds__(256) sparse_mla_pf("),
            ("H=32,N=32,T=512,LD=LAT+8", "H=32,N=32,T=256,LD=LAT+8"),
            ("out[4],outlo[4]", "out[8],outlo[8]"), ("j<4", "j<8"),
            ("h+=16", "h+=8"), ("(w/8)", "(w/4)"), ("(w%8)*64", "(w%4)*128")]
        for old, new in changes:
            self.assertIn(old, k)
            k = k.replace(old, new)
        start = self.s.index("template<class Input>\n__global__ void __launch_bounds__(256) sparse_mla_pf(")
        self.assertEqual(k, self.s[start:self.s.index(MARK)].rstrip())

    def test_only_candidate_and_nested_dispatch_changed(self):
        s = strip_v2(self.s)
        a, z = s.index(MARK), s.index(END, s.index(MARK))
        s = s[:a]+s[z:]
        s = s.replace('    const char* mla512_env=getenv("G6_PF_SPARSE_MLA_512");\n'
                      '    const bool fused_mla512=fused_mla&&mla512_env&&atoi(mla512_env)==1;\n', '')
        s = s.replace('''        if(fused_mla512)
            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf_512<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM);
        else
            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM);''',
                      '''        ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf<G6Attn>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA_SMEM);''')
        s = s.replace('''            if(fused_mla512)fa_pf::sparse_mla_pf_512<<<n,512,fa_pf::MLA_SMEM,stream>>>(a,start,w->qh,w->ctxh);
            else fa_pf::sparse_mla_pf<<<n,256,fa_pf::MLA_SMEM,stream>>>(a,start,w->qh,w->ctxh);''',
                      '''            fa_pf::sparse_mla_pf<<<n,256,fa_pf::MLA_SMEM,stream>>>(a,start,w->qh,w->ctxh);''')
        self.assertEqual(hashlib.sha256(s.encode()).hexdigest(), P1_SHA)

    def test_actual_dispatch_host_compiles_and_all_flag_combinations(self):
        a = self.s.index('    const char* mla_env=getenv("G6_PF_SPARSE_MLA");')
        z = self.s.index('    if(fused_mla){', a)
        dispatch = self.s[a:z]
        code = '#include <cstdlib>\n#include <cstdio>\nint main(int argc,char**argv){\n'
        code += 'struct {int heads;} a{atoi(argv[1])};\n' + dispatch
        code += 'printf("%d %d %d\\n",int(fused_mla),int(fused_mla512),int(fused_mla_v2));}\n'
        values = [None, "", "0", "1", "2", "-1"]
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d)/"dispatch.cpp", Path(d)/"dispatch"
            cpp.write_text(code)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(exe)],
                           check=True, capture_output=True)
            for heads in (1, 16, 22, 32, 33, 64):
                for p1 in values:
                    for opt in values:
                        for v2 in values:
                            env = dict(os.environ)
                            for name, value in [("G6_PF_SPARSE_MLA", p1), ("G6_PF_SPARSE_MLA_512", opt),
                                                ("G6_PF_SPARSE_MLA_V2", v2)]:
                                env.pop(name, None)
                                if value is not None:
                                    env[name] = value
                            got = subprocess.check_output([str(exe), str(heads)], env=env, text=True).strip()
                            enabled = p1 == "1" and heads <= 32
                            on512 = enabled and opt == "1"
                            onv2 = enabled and (v2 == "1" or (v2 is None and on512))
                            self.assertEqual(got, f"{int(enabled)} {int(on512)} {int(onv2)}",
                                             (heads, p1, opt, v2))

    def test_launch_and_attribute_are_nested_in_p1_path(self):
        self.assertIn('''    if(fused_mla){
        if(fused_mla512)
            ce=cudaFuncSetAttribute(fa_pf::sparse_mla_pf_512<G6Attn>''', strip_v2(self.s))
        self.assertIn('''        if(fused_mla){
            if(fused_mla512)fa_pf::sparse_mla_pf_512<<<n,512,fa_pf::MLA_SMEM,stream>>>''', strip_v2(self.s))
        self.assertEqual(strip_v2(self.s).count("sparse_mla_pf_512<<<"), 1)
        self.assertEqual(strip_v2(self.s).count("cudaFuncSetAttribute(fa_pf::sparse_mla_pf_512"), 1)

    def test_v2_is_nested_in_p1_path_and_defaults_only_with_512(self):
        s = strip_h16c(self.s)
        self.assertEqual(s.count("sparse_mla_pf_v2<<<"), 1)
        self.assertEqual(s.count("cudaFuncSetAttribute(fa_pf::sparse_mla_pf_v2"), 1)
        self.assertIn("const bool fused_mla_v2=fused_mla&&(mla_v2_env?atoi(mla_v2_env)==1:fused_mla512);", s)
        self.assertIn('getenv("G6_PF_SPARSE_MLA_V2")', s)
        self.assertIn("        if(fused_mla){\n            if(fused_mla_v2)fa_pf::sparse_mla_pf_v2<<<n,512,fa_pf::MLA2_SMEM", s)

    def test_h16c_is_separate_opt_in_on_v2_h16_only(self):
        s = self.s
        self.assertIn('fused_mla_h16c=fused_mla_v2&&a.heads<=16&&h16c_env&&atoi(h16c_env)==1;', s)
        self.assertIn('sparse_mla_pf_v2<G6Attn,16><<<n,512,fa_pf::MLA2_SMEM_H16', s)
        self.assertIn('sparse_mla_pf_v2<G6Attn,16>,cudaFuncAttributeMaxDynamicSharedMemorySize,fa_pf::MLA2_SMEM_H16', s)
        self.assertEqual(s.count('sparse_mla_pf_v2<G6Attn,16><<<'), 1)
        self.assertEqual(s.count('sparse_mla_pf_v2<<<'), 1)
        self.assertEqual(s.count('if(H==32 || w<H/2)'), 2)  # H32 keeps the unconditional PV and epilogue paths

    def test_h16c_host_dispatch(self):
        a = self.s.index('    const char* mla_env=getenv("G6_PF_SPARSE_MLA");')
        z = self.s.index('    if(fused_mla){', a)
        code = '#include <cstdlib>\n#include <cstdio>\nint main(int argc,char**argv){\n'
        code += 'struct {int heads;} a{atoi(argv[1])};\n' + self.s[a:z]
        code += 'printf("%d\\n",int(fused_mla_h16c));}\n'
        with tempfile.TemporaryDirectory() as d:
            cpp, exe = Path(d)/"h16c_dispatch.cpp", Path(d)/"h16c_dispatch"
            cpp.write_text(code)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(exe)],
                           check=True, capture_output=True)
            for heads in (16, 32, 33):
                for sparse in (None, "1"):
                    for mla512 in (None, "1"):
                        for v2 in (None, "0", "1"):
                            for h16c in (None, "0", "1"):
                                env = dict(os.environ)
                                for name, value in (("G6_PF_SPARSE_MLA", sparse), ("G6_PF_SPARSE_MLA_512", mla512),
                                                    ("G6_PF_SPARSE_MLA_V2", v2), ("G6_PF_SPARSE_MLA_H16C", h16c)):
                                    env.pop(name, None)
                                    if value is not None: env[name] = value
                                got = int(subprocess.check_output([str(exe), str(heads)], env=env, text=True))
                                expected = sparse == "1" and heads <= 16 and h16c == "1" and \
                                    (v2 == "1" or (v2 is None and mla512 == "1"))
                                self.assertEqual(got, int(expected), (heads, sparse, mla512, v2, h16c))

    def test_pv_mapping_has_same_complete_output_coverage(self):
        wanted = {(h, d) for h in (0, 16) for d in range(0, 512, 16)}
        for warps in (8, 16):
            self.assertEqual(len(tiles(warps)), 64)
            self.assertEqual(set(tiles(warps)), wanted)
            for heads in (1, 16, 22, 32):
                cells = {(h,d) for hb,db in tiles(warps)
                         for h in range(hb,min(heads,hb+16)) for d in range(db,db+16)}
                self.assertEqual(len(cells), heads*512)

    def test_load_and_softmax_coverage(self):
        for threads in (256,512):
            for size in (32*512//8, 32*512):
                self.assertEqual(sorted(i for t in range(threads) for i in range(t,size,threads)),
                                 list(range(size)))
        for warps in (8,16):
            self.assertEqual(sorted(h for w in range(warps) for h in range(w,32,warps)), list(range(32)))

    def test_qk_warp_mapping_and_pv_reduction_order_unchanged(self):
        k = kernel(self.s)
        self.assertIn("if(w<4)", k)
        self.assertIn("base+=N", k)
        self.assertIn("k<N;k+=16", k)
        seq = tuple((base, inner) for base in range(0,2304,32) for inner in (0,16))
        self.assertEqual({key:seq for key in tiles(8)}, {key:seq for key in tiles(16)})

if __name__ == "__main__":
    unittest.main()
