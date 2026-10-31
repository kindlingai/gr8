#!/usr/bin/env python3
"""Source contract: the serve W-row prefill callback (g6m4.cu dc.prefill) calls ds41_engram_begin.

The nonfinal plan has no step_in, and the decode step enqueues ds41_engram_begin (token window + ring update + hash + NVMe prefetch) right after
step_in (ds41_step.cuh launch_step_ds41). So the callback must call it itself, after its own m.tok / m.pos copies and before the first ds41::run(...) (bounded replay: range A of split.ops).
Without it, with GR8_DS_ENGRAM=1, every prefill chunk uses stale ids_d_ on layers 1 and 14 and the prompt never enters the token ring.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
BEGIN = re.compile(r"ds41_engram_begin\s*\(\s*m\s*,\s*\(int\)\s*c\.rows\s*,\s*st\s*(?:,\s*\(int\)\s*c\.slot\s*)?\)")   # SESS8: optional slot


def prefill_block(rank_src: str) -> str:
    head = rank_src.index("dc.prefill = [&](const g6s2::Ds41PrefillChunk& c) {")
    tail = rank_src.index("dc.after_prefill = ", head)
    return rank_src[head:tail]


def problems(block: str) -> list:
    """Empty list = the contract holds for this lambda body."""
    out = []
    m = BEGIN.search(block)
    if not m:
        return ["dc.prefill does not call ds41_engram_begin(m, (int)c.rows, st)"]
    if len(BEGIN.findall(block)) != 1:
        out.append("dc.prefill calls ds41_engram_begin more than once (the ring put and the hash must run once per chunk)")
    guard = block[max(0, m.start() - 160):m.start()]
    if not re.search(r"if\s*\(\s*m\.engram\s*\)\s*\{?\s*$", guard.rstrip(), re.DOTALL):
        out.append("the call is not guarded by `if (m.engram)` directly in front of it (Engram off must not touch the engine)")
    tok = block.find("cudaMemcpyAsync(m.tok,")
    pos = block.find("cudaMemcpyAsync(m.pos,")
    run = block.find("ds41::run(")   # the first plan walk (range A of the bounded replay split, or the whole nonfinal plan)
    if min(tok, pos, run) < 0:
        out.append("the callback no longer has the m.tok / m.pos copies or a ds41::run(...): update this test with the callback")
        return out
    if not (tok < m.start() and pos < m.start()):
        out.append("ds41_engram_begin runs before the m.tok / m.pos copies: the window would read the previous chunk's tokens")
    if not m.start() < run:
        out.append("ds41_engram_begin runs after the first ds41::run(...): the Engram layers would consume the previous chunk's ids")
    if "fprintf(stderr, \"# rank %d request %llu ds41 engram prefill begin" not in block:
        out.append("the per-chunk proof line '# rank R request Q ds41 engram prefill begin chunk F rows N' is missing")
    return out


class Ds41PrefillEngramBegin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rank = (ROOT / "runtime/g6m4.cu").read_text()
        cls.block = prefill_block(cls.rank)

    def test_w_callback_calls_engram_begin_in_order(self):
        self.assertEqual(problems(self.block), [])

    def test_decode_step_still_calls_it_after_step_in(self):
        step = (ROOT / "runtime/ds41_step.cuh").read_text()
        self.assertRegex(
            step,
            re.compile(r"ds41_unit\(m,ds41::U_STEP_IN,0,M,st\);\s*if\(m\.engram\)\s*ds41_engram_begin\(m,M,st\);", re.DOTALL),
        )

    def test_callers_are_exactly_the_decode_step_and_the_w_callback(self):
        n = 0
        for p in sorted((ROOT / "runtime").rglob("*")):
            if p.suffix in (".cu", ".cuh", ".h") and "tests" not in p.parts:
                n += len(re.findall(r"ds41_engram_begin\s*\(", p.read_text()))
        self.assertEqual(n, 3, "definition + launch_step_ds41 + dc.prefill; a new caller must be added to this test")

    # falsifiers: the checker must fail on each way the callback can be wrong
    def test_falsifier_call_removed(self):
        self.assertNotEqual(problems(BEGIN.sub("(void)0", self.block)), [])

    def test_falsifier_unguarded(self):
        self.assertNotEqual(problems(self.block.replace("if (m.engram) {", "{", 1)), [])

    def test_falsifier_call_before_token_copy(self):
        m = BEGIN.search(self.block)
        call = self.block[m.start():m.end()] + ";"
        b = self.block.replace(call, "", 1)
        i = b.index("G6RT_CK(cudaMemcpyAsync(m.tok,")
        b = b[:i] + "if (m.engram) " + call + "\n" + b[i:]
        self.assertNotEqual(problems(b), [])

    def test_falsifier_call_after_run(self):
        m = BEGIN.search(self.block)
        call = self.block[m.start():m.end()] + ";"
        b = self.block.replace(call, "", 1)
        i = b.index("if (stop.stopped)")
        b = b[:i] + "if (m.engram) " + call + "\n" + b[i:]
        self.assertNotEqual(problems(b), [])


if __name__ == "__main__":
    unittest.main()
