"""CPU source check of G6_PF_STAGE_ASYNC: default ON at NR=2 only (=0 kills it, =1 asks for it at any NR, an unsupported config dies only when asked), the OFF consume path and MhcFabricBlock are untouched, the async path
issues at the put and waits ready / records used at the consume, one high-priority stream per lane."""
from pathlib import Path
import re
import unittest

RT = Path(__file__).resolve().parents[1]
PF = (RT / 'flash_prefill.cuh').read_text()
ST = (RT / 'flash_step.cuh').read_text()


def body(src, head):
    return src.split(head, 1)[1].split('\n}\n', 1)[0]


class StageAsyncTest(unittest.TestCase):
    def test_env_default_on_at_tp2_with_kill_switch(self):
        env = body(PF, 'inline int pf_stage_async_mode()')
        self.assertIn('getenv("G6_PF_STAGE_ASYNC")', env)
        self.assertIn('if (e && *e) return atoi(e) == 1 ? 2 : 0;', env)   # set: =1 asks for it, anything else (=0) is off
        self.assertIn('return g6nr::NR == 2 ? 1 : 0;', env)                 # unset: on at TP2, off at TP4

    def test_build_gate_and_stream(self):
        b = body(PF, 'inline void build_flash_prefill')
        gate = b.split('pf_stage_async_mode())', 1)[1].split('cudaStreamCreateWithPriority', 1)[0]
        for need in ('!pf_mhc_stage()', 'pf_fuse_level() < 1', 'pf_xtrace_env()', 'G6_PF_RS'):
            self.assertIn(need, gate)
        self.assertIn('if (no && mode == 2)', gate)                          # asked for: die
        self.assertIn('G6_PF_STAGE_ASYNC=1 cannot run with', gate)
        self.assertIn('if (no) fprintf(stderr, "# rank %d: G6_PF_STAGE_ASYNC off (default-on skipped', gate)   # default-on: stay off, say so
        self.assertIn('G6_PF_STAGE_ASYNC on (lane %d, peers %d)', b)
        self.assertIn('check_headroom(f.rank, MP, false)', b)
        self.assertIn('cudaStreamCreateWithPriority(&S.stream, cudaStreamNonBlocking, hi)', b)   # one per lane: built per FlashPrefill
        stage = b.split('PfPeerStage& S = P.peer_stage;', 1)[1]
        self.assertEqual(stage.count('cudaEventCreateWithFlags'), 3)   # put, ready, used (G6_PF_SP's events are its own)

    def test_issue_at_put_consume_waits_ready_records_used(self):
        put = body(PF, 'inline void pf_put_flash')
        self.assertIn('if (P.peer_stage.on) pf_stage_issue(P, c, M, st);', put)
        issue = body(PF, 'inline void pf_stage_issue')
        order = [issue.index(x) for x in ('cudaEventRecord(S.put, st)', 'cudaStreamWaitEvent(S.stream, S.put', 'cudaStreamWaitEvent(S.stream, S.used',
                                         'k_wait_peers<<<1, 32, 0, S.stream>>>', 'cudaMemcpyAsync', 'cudaEventRecord(S.ready, S.stream)')]
        self.assertEqual(order, sorted(order))
        cons = body(PF, 'inline void pf_consume_flash')
        on = cons.split('if (P.peer_stage.on)', 1)[1].split('return;', 1)[0]
        self.assertLess(on.index('cudaStreamWaitEvent(st, S.ready'), on.index('pf_consume_launch'))
        self.assertLess(on.index('pf_consume_launch'), on.index('cudaEventRecord(S.used, st)'))
        self.assertIn('MhcFabricPeerBlock', on)
        self.assertNotIn('k_wait_peers', on)

    def test_off_path_unchanged(self):
        # MhcFabricBlock (all OFF kernels take it by value) is byte-identical to the base commit
        import subprocess
        base = subprocess.run(['git', 'show', 'e2069fca:engine/runtime/flash_step.cuh'], cwd=RT, capture_output=True, text=True)
        if base.returncode != 0:
            self.skipTest('base commit not available')
        def blk(s):
            return s.split('struct MhcFabricBlock {', 1)[1].split('\n};\n', 1)[0]
        self.assertEqual(blk(base.stdout), blk(ST))
        # the OFF consume still launches MhcFabricBlock kernels with the same arguments
        cons = body(PF, 'inline void pf_consume_flash')
        self.assertIn('MhcFabricBlock blk{P.f, P.stepp, c, wp, pf_fuse >= 1 ? P.part[c] : nullptr};', cons)
        self.assertIn('if (stage) blk.peer_part = P.peer_dev;', cons)

    def test_peer_block_sums_in_rank_order(self):
        blk = ST.split('struct MhcFabricPeerBlock', 1)[1].split('\n};\n', 1)[0]
        self.assertIn('for (int p = 0; p < g6nr::NR; ++p) {', blk)
        self.assertIn('sum += __uint_as_float((unsigned)v << 16);', blk)


if __name__ == '__main__':
    unittest.main()
