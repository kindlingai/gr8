"""CPU source guard for C2: mapped direct W4096 only, never cudaMalloc fallback."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ops = (ROOT / 'engine/attn/ds41_attn_ops.cuh').read_text()
entry = (ROOT / 'engine/attn/ds41_attn.cu').read_text()
attn = (ROOT / 'engine/attn/ds41_attn.cuh').read_text()
step = (ROOT / 'engine/runtime/ds41_step.cuh').read_text()

def check(o, e, a, s):
    # The only C entry invocation passes the direct-chunk flag, not a constant.
    call = re.findall(r'ds41_attn_out\(&a->W, &a->S, rows, 0, a->heads, m\.sc\.part\[0\], stream, m\.pos,\s*([^;]+)\);', o)
    assert call == ['m.ds41_direct && m.ds41_dchunk && rows == 4096'], call
    assert len(re.findall(r'\bds41_attn_out\(', o)) == 2  # declaration + call
    assert 'dstart, direct_c2 != 0)' in e
    assert 'bool direct_c2 = false' in a
    # C2 sits under the runtime guard. The original wo_b projection remains
    # outside it for cudaMalloc or any unsupported shape.
    body = a[a.index('inline void attn_out('):a.index('// The whole layer in plan order')]
    guard = re.search(r'if \(direct_c2 && g_use_tc && n == 4096 && W\.wo_b\.N == 5120 && W\.wo_b\.K == 2048 && G \* OLORA == 2048 && !use_gv\(W\.wo_b, n\)\)\s*\{([\s\S]*?)\n    \}\s*proj_mm\(W\.wo_b, S\.oaq', body)
    assert guard and guard.group(1).count('launch_tc_t_c2<128, 128, 2>') == 1
    assert body.count('launch_tc_t_c2<128, 128, 2>') == 1
    assert body.count('proj_mm(W.wo_b, S.oaq') == 1
    assert 'if (tc_ok(p)) { launch_tc_t_c2<128, 128, 2>(p, 1, st); return; }' in guard.group(1)
    # Only a direct chunk routes sc.part[0] to mapped send memory. With
    # PF_DIRECT off, the same output stays on cudaMalloc and C2 is false.
    assert 'return m.ds41_direct && m.ds41_dchunk;' in s
    assert 'if(!ds41_direct_chunk(m)) return g_ds41_dir_inner[K](model,i,rows,stream);' in s
    assert 'm.sc.part[K]=ds41_dir_dst(m,2*i+K);' in s

check(ops, entry, attn, step)
for mutant in (
    (ops.replace('m.ds41_direct && m.ds41_dchunk && rows == 4096', 'true'), entry, attn, step),
    (ops.replace('m.ds41_direct && m.ds41_dchunk && rows == 4096', 'm.ds41_dchunk && rows == 4096'), entry, attn, step),
    (ops, entry, attn.replace('if (direct_c2 && g_use_tc', 'if (g_use_tc'), step),
    (ops, entry, attn.replace('    proj_mm(W.wo_b, S.oaq', '    launch_tc_t_c2<128, 128, 2>(p, 1, st);\n    proj_mm(W.wo_b, S.oaq'), step),
):
    try:
        check(*mutant)
    except (AssertionError, ValueError):
        continue
    raise AssertionError('C2 dispatch mutant was accepted')

print('DS41_C2_DISPATCH_HOST_PASS direct_only=1 full4096=1 dev_original=1 mutants=4')
