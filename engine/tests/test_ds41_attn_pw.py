"""Host-only contract for the attention prefill row capacity PW (GR8_DS_ATTN_PW): source rules of ds41_attn_ops.cuh and the window-ring arithmetic.

The ring part replays k_kv_prep (ds41_attn.cuh: slot = position % R, only the last R rows of a chunk are written) and k_attn's decode window read
(slot of position start + t - 127 + i, masked below 0) for a chunk of n rows after the ring already holds the earlier positions, and checks that
every row still finds its 128 window keys when R = WIN + rowcap - 1 and n <= rowcap. A chunk of rowcap + 1 rows must fail (negative control).
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
WIN = 128


def ring_ok(rowcap: int, n: int, start: int) -> bool:
    R = WIN + rowcap - 1
    ring = [None] * R
    for p in range(max(0, start - R), start):          # earlier chunks
        ring[p % R] = p
    for row in range(n):                                # k_kv_prep: only the last R rows are written
        if row >= n - R:
            ring[(start + row) % R] = start + row
    for t in range(n):                                  # k_attn decode read, list slot i holds position start + t - 127 + i
        for i in range(WIN):
            pos = start + t - (WIN - 1) + i
            if pos >= 0 and ring[pos % R] != pos:
                return False
    return True


def check_source(ops: str) -> None:
    assert "rows > MX" not in ops, "a per-op row limit is still MX"
    assert ops.count("rows > a->rowcap") >= 4          # proj, comp_row, core, out
    assert "rows > (a0 ? a0->rowcap : MX)" in ops      # prepare
    assert "a->C.ring_rows = ds41a::WIN + a->rowcap - 1;" in ops
    assert "a->rowcap = g_ds41_attn_pw > MX ? g_ds41_attn_pw : MX;" in ops
    assert "if (a->rowcap > MX) a->S.carve(ds41_attn_shared_scratch(a->rowcap, a->heads), a->rowcap, a->heads);" in ops
    assert "else a->S.carve(dmalloc<char>(ds41a::Scratch::bytes(MX, a->heads)), MX, a->heads);" in ops   # default: per-layer scratch at MX
    assert "knob::ds_attn_pw()" in ops
    assert "inline int g_ds41_attn_pw = 0;" in ops      # default off


class Contract(unittest.TestCase):
    def test_source_contract(self):
        check_source((ROOT / "engine/attn/ds41_attn_ops.cuh").read_text())

    def test_old_limit_mutation_is_caught(self):
        ops = (ROOT / "engine/attn/ds41_attn_ops.cuh").read_text()
        with self.assertRaises(AssertionError):
            check_source(ops.replace("rows > a->rowcap", "rows > MX", 1))

    def test_ring_holds_every_window_key(self):
        for rowcap in (8, 32, 128, 256, 512, 1024):
            R = WIN + rowcap - 1
            for start in (0, 1, 2, 127, 128, 129, 255, 300, R - 1, R, R + 1, 2 * R + 5, 4097):
                for n in {1, 2, 7, 8, 9, 31, 32, rowcap // 2 or 1, rowcap - 1 or 1, rowcap}:
                    if 1 <= n <= rowcap:
                        self.assertTrue(ring_ok(rowcap, n, start), (rowcap, n, start))

    def test_chunk_longer_than_rowcap_loses_a_key(self):
        for rowcap in (8, 32, 128, 256):
            self.assertFalse(ring_ok(rowcap, rowcap + 1, 4097), rowcap)

    def test_shared_prefill_ring_stage_commit(self):
        mx, pw, ns = 48, 4096, 32
        r_dec, r_pf = WIN + mx - 1, WIN + pw - 1
        self.assertEqual((r_dec, r_pf), (175, 4223))
        ring_dec = [[None] * r_dec for _ in range(ns)]
        ring_pf = [None] * r_pf

        def run_chunk(slot: int, start: int, n: int) -> None:
            if n > mx:
                # k_ds41_ring_stage: copy WIN-1 preceding keys from ring_dec[slot] -> ring_pf
                for b in range(WIN - 1):
                    p = start - (WIN - 1) + b
                    if p >= 0:
                        ring_pf[p % r_pf] = ring_dec[slot][p % r_dec]
                # k_kv_prep into ring_pf
                for row in range(max(0, n - r_pf), n):
                    ring_pf[(start + row) % r_pf] = (slot, start + row)
                # k_ds41_ring_commit: copy last min(n, WIN) keys from kvrows -> ring_dec[slot]
                for b in range(WIN):
                    row = n - WIN + b
                    if row >= 0:
                        ring_dec[slot][(start + row) % r_dec] = (slot, start + row)
                # k_attn read from ring_pf
                for t in range(n):
                    for i in range(WIN):
                        pos = start + t - (WIN - 1) + i
                        if pos >= 0:
                            self.assertEqual(ring_pf[pos % r_pf], (slot, pos))
            else:
                # decode / tail chunk directly on ring_dec[slot]
                for row in range(max(0, n - r_dec), n):
                    ring_dec[slot][(start + row) % r_dec] = (slot, start + row)
                for t in range(n):
                    for i in range(WIN):
                        pos = start + t - (WIN - 1) + i
                        if pos >= 0:
                            self.assertEqual(ring_dec[slot][pos % r_dec], (slot, pos))

        # Interleave wide prefill chunks across slots 0, 15, 31 and follow with tail + decode steps
        run_chunk(0, 0, 4096)
        run_chunk(31, 0, 2000)
        run_chunk(15, 0, 65)
        run_chunk(0, 4096, 1500)
        run_chunk(31, 2000, 4096)
        for slot, cur in ((0, 5596), (15, 65), (31, 6096)):
            run_chunk(slot, cur, 1)       # PromptTail (n=1)
            run_chunk(slot, cur + 1, 6)   # MTP=5 verify (n=6)
            run_chunk(slot, cur + 7, 48)  # max decode width (n=48)



if __name__ == "__main__":
    unittest.main()
