"""CPU source checks of G6_PF_ROUTER_SPLIT (flash-G4), no GPU:

    python3 engine/tests/test_flash_route_split_source.py -v

1. OFF surface: the decode router (k_router_linear / glue.cuh / flash_mlp.cuh / rank.cuh / flash_step.cuh) and the router kernels
   themselves keep BASE's bytes, except NK's additive prefill-only MhcFabricPeerBlock; Expert::pre_routed is the LAST member of
   Expert; launch_route runs k_pf_route unless pre_routed,
   count and fill always (the skip sits INSIDE bindp::launch_route, so every caller inherits it); the OFF block calls
   launch_router on all M exactly as before.
2. ON order: router on own rows -> own-row route -> publish -> hx -> shared -> receive -> pre_routed -> experts.
3. K < TOPK: k_pf_route writes all TOPK slots of a row (dropped = -1 / 0.0f) and the exchange moves all of them.
4. Boot: strict env, m4sim fails closed, any rank count 2..8 on a mesh or the ring, one unpack instance for every rank count,
   ports / proxy CPU offsets disjoint from every other fabric of a Flash g6m4.

OffSurface compares with the commit that added the split (BASE); a checkout without it (a shallow CI clone) skips those cases.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = 'b99f9d9'


def src(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def base(name):
    return subprocess.check_output(['git', '-C', ROOT, 'show', f'{BASE}:{name}']).decode()


def have_base():
    return subprocess.run(['git', '-C', ROOT, 'cat-file', '-e', f'{BASE}^{{commit}}'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


class OffSurface(unittest.TestCase):
    @unittest.skipUnless(have_base(), f'commit {BASE} is not in this checkout')
    def test_decode_router_is_base_bytes(self):
        for name in ['engine/kernels/glue.cuh', 'engine/kernels/flash_mlp.cuh', 'engine/runtime/rank.cuh',
                     'engine/model_geom.h', 'engine/runtime/envparse.h']:
            self.assertEqual(base(name), src(name), name)
        name = 'engine/runtime/flash_step.cuh'
        step = src(name)
        added = between(step, '// Async prefill only.', '// Decode-only pinned-partial stage.')
        self.assertIn('struct MhcFabricPeerBlock : MhcFabricBlock', added)
        self.assertEqual(base(name), step.replace(added, ''))

    @unittest.skipUnless(have_base(), f'commit {BASE} is not in this checkout')
    def test_router_kernels_are_base_bytes(self):
        a, b = src('engine/runtime/flash_mlp_pf.cuh'), base('engine/runtime/flash_mlp_pf.cuh')
        for start, end in [('template <int MR = 4>\n__global__', '// Register-tiled router GEMM'),
                           ('// Router logits y [M][N]', '__global__ void __launch_bounds__(256) k_swiglu_bf16')]:
            self.assertEqual(between(b, start, end), between(a, start, end))
        pa, pb = src('engine/kernels/bind_prefill.cuh'), base('engine/kernels/bind_prefill.cuh')
        for start, end in [('template <int K = TOPK> __global__ void __launch_bounds__(256) k_pf_route(Expert A)', '// grid NEXP (expert), block 256\n__global__'),
                           ('__global__ void __launch_bounds__(256) k_pf_count', '// grid NEXP (expert), block 256: the expert')]:
            self.assertEqual(between(pb, start, end), between(pa, start, end))

    def test_pre_routed_is_last_member(self):
        s = src('engine/kernels/bind_prefill.cuh')
        body = between(s, 'struct Expert\n{', '\n};\n')
        decl = [l for l in body.split('\n') if re.match(r'    (int|const|float|__half|uint|long)', l)]
        self.assertTrue(decl[-1].startswith('    int pre_routed = 0;'), decl[-1])
        self.assertEqual(body.count('pre_routed'), 1)

    @unittest.skipUnless(have_base(), f'commit {BASE} is not in this checkout')
    def test_pre_routed_base_members(self):
        s = src('engine/kernels/bind_prefill.cuh')
        body = between(s, 'struct Expert\n{', '\n};\n')
        b = between(base('engine/kernels/bind_prefill.cuh'), 'struct Expert\n{', '\n};\n')
        # Compare the pre-existing declarations exactly; only their explanatory comment changed.
        no_comments = lambda s: re.sub(r'//[^\n]*', '', s)
        self.assertEqual(no_comments(body[:body.index('    int pre_routed')]), no_comments(b + '\n'))

    def test_skip_inside_launch_route(self):
        s = src('engine/kernels/bind_prefill.cuh')
        lr = between(s, 'inline void launch_route(const Expert& A', '// EB: blocks per expert')
        self.assertIn('if (!A.pre_routed) launch_route_rows(A, st);', lr)
        self.assertIn('k_pf_count<<<NEXP, 256, 0, st>>>(A);', lr)
        self.assertIn('if (A.topk < TOPK) k_pf_fill<true>', lr)
        self.assertEqual(s.count('pre_routed'), 3)          # the member, the launch_route skip and its comment
        rows = between(s, 'inline void launch_route_rows', 'inline void launch_route_slice')
        for k in range(1, 8):
            self.assertIn(f'case {k}: k_pf_route<{k}>', rows)
        self.assertIn('default: k_pf_route<TOPK>', rows)
        # launch_experts reaches the skip only through launch_route
        self.assertIn('    launch_route(A, st);\n    launch_fc1<EB>(A, st);', s)

    def test_off_block_is_base_call(self):
        a = src('engine/runtime/flash_mlp_pf.cuh')
        self.assertIn('else flash_mlp_pf::launch_router(P.xn2, w.router, S.logits, HID, geom::NEXP, M, st);', a)
        self.assertIn('const bool split = P.route_split.fabric && flash_route_split::active(M, P.route_split.f.nr);', a)

    def test_route_slice_views(self):
        s = src('engine/kernels/bind_prefill.cuh')
        sl = between(s, 'inline void launch_route_slice', 'inline void launch_route(const Expert& A')
        for line in ['B.M = rows;', 'B.logits = A.logits + static_cast<size_t>(r0) * NEXP;',
                     'B.tok_e = A.tok_e + static_cast<size_t>(r0) * TOPK;', 'B.tok_w = A.tok_w + static_cast<size_t>(r0) * TOPK;']:
            self.assertIn(line, sl)


class OnOrder(unittest.TestCase):
    def test_order(self):
        a = between(src('engine/runtime/flash_mlp_pf.cuh'), 'if (L.moe)', 'bindp::launch(L.xq_x2, st);')
        order = ['launch_router(P.xn2 + ', 'bindp::launch_route_slice(L.ex, r0, rows, st);', 'flash_route_publish(',
                 'bindp::launch_hx(L.ex, st);', 'bindp::launch_shared<g6nr::EB>(L, st);', 'flash_route_receive(',
                 'L.ex.pre_routed = 1;', 'bindp::launch_experts<g6nr::EB>(L.ex, st);']
        pos = [a.index(t) for t in order]
        self.assertEqual(pos, sorted(pos))
        self.assertEqual(a.count('launch_hx'), 1)

    def test_sizes_agree(self):
        self.assertIn('static_assert(flash_route_split::SLOTS == geom::TOPK', src('engine/runtime/flash_mlp_pf.cuh'))


class DroppedSlots(unittest.TestCase):
    def test_route_writes_all_slots(self):
        s = src('engine/kernels/bind_prefill.cuh')
        k = between(s, 'template <int K = TOPK> __global__ void __launch_bounds__(256) k_pf_route(Expert A)', '// grid NEXP (expert), block 256\n__global__')
        self.assertIn('if (l < TOPK)', k)                               # all 8 lanes store: slots >= K get -1 / 0
        self.assertIn('A.tok_e[m * TOPK + l] = -1;', k)
        self.assertIn('A.tok_w[m * TOPK + l] = 0.f;', k)

    def test_exchange_moves_all_slots(self):
        h = src('engine/runtime/flash_route_split.h')
        self.assertIn('constexpr int SLOTS = 8, ROW_BYTES = SLOTS * 8;', h)
        c = src('engine/runtime/flash_route_split.cuh')
        self.assertIn('out[i] = (i & 3) < 2 ? e[(i >> 2) * 2 + (i & 3)] : w[(i >> 2) * 2 + (i & 3) - 2];', c)
        self.assertIn('for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * 4;', c)   # 4 uint4 = 64 B per row


class Boot(unittest.TestCase):
    def test_env_strict(self):
        c = src('engine/runtime/flash_route_split.cuh')
        self.assertIn('env_choice("G6_PF_ROUTER_SPLIT", 1, 0, &e)', c)
        self.assertIn('if (v < 0) die(e);', c)
        g = src('engine/runtime/g6m4.cu')
        self.assertIn('if (getenv("G6_PF_ROUTER_SPLIT")) die("G6_PF_ROUTER_SPLIT is a Flash-model option', g)
        rs = between(g, 'const bool route_split = route_split_mode == 1;', 'auto init_rs')
        self.assertIn('g6nr::NR >= 2 && g6nr::NR <= flash_route_split::MAX_NR', rs)
        self.assertIn('cfg.ips.size() == (size_t)g6nr::NR', rs)
        self.assertNotIn('ring', rs)          # the ring fabric relays the slot as it does the prefill fabric's
        self.assertNotIn('NR == 4', rs)
        self.assertIn('if (getenv("G6_PF_ROUTER_SPLIT"))', src('engine/runtime/m4sim.cu'))

    def test_one_unpack_for_every_rank_count(self):
        c = src('engine/runtime/flash_route_split.cuh')
        self.assertIn('k_rs_unpack<<<dim3(RS_UNPACK_CTA, p.f.nr), 256, 0, st>>>', c)
        self.assertIn('if (p >= f.nr || p == f.rank', c)
        self.assertNotIn('template <int NR>', c)
        self.assertIn('p.f.nr < 2 || p.f.nr > MAX_NR', c)
        self.assertIn('constexpr int MAX_NR = 8;', src('engine/runtime/flash_route_split.h'))

    def test_ports_and_cpus_disjoint(self):
        g = src('engine/runtime/g6m4.cu')
        ports = [int(x) for x in re.findall(r'cfg\.port \+ (\d+)', g)] + [int(x) for x in re.findall(r'init_(?:ids|rs)\(\w+, (\d+), \d+\)', g)]
        self.assertEqual(sorted(ports).count(80), 1)
        self.assertEqual(sorted(ports).count(88), 1)
        self.assertEqual(len(ports), len(set(ports)), sorted(ports))
        # the replicated-GEMV gather fabric (gc, offset 9) is GLM-5.3 only: a Flash build dies on G6_QA_SPLIT and friends,
        # and the pool-ID fabric takes offset 9 there
        self.assertIn('if (G6_MODEL_FLASH && (qs_mode || iq_split || mtp_split)) die(', g)
        flash = '\n'.join(l for l in g.split('\n') if 'gc.proxy_cpu' not in l)
        cpus = [int(x) for x in re.findall(r'proxy_cpu - (\d+)', flash)] + [int(x) for x in re.findall(r'init_(?:ids|rs)\(\w+, \d+, (\d+)\)', flash)]
        self.assertEqual(len(cpus), len(set(cpus)), sorted(cpus))
        self.assertTrue({11, 12} <= set(cpus))

    def test_teardown_and_error_word(self):
        g = src('engine/runtime/g6m4.cu')
        self.assertIn('release_flash_route_split(FP.route_split);', g)
        self.assertIn('route-table fabric error', g)
        self.assertIn('init_flash_route_split(FP.route_split, fabrs, MP);', g)


if __name__ == '__main__':
    unittest.main()
