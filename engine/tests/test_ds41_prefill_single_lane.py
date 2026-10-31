#!/usr/bin/env python3
"""Source contract for the DS41 W callback's scheduler lane override.

GR8_DS_PF_LANES unset / 1 keeps the one-lane path: ds41_prefill_lanes(sc, W, 1) is ds41_prefill_single_lane, and serve_ds41
installs the one-lane prefill_seq unless the two-lane callee (prefill2) is set."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Ds41PrefillSingleLane(unittest.TestCase):
    def test_w_callback_overrides_scheduler_default_without_touching_decode(self):
        scheduler = (ROOT / "runtime/s2/scheduler.h").read_text()
        native = (ROOT / "runtime/s2/ds41_native.cuh").read_text()
        rank = (ROOT / "runtime/g6m4.cu").read_text()
        self.assertIn("bool two_lanes = true;", scheduler)
        self.assertIn("(!sc.two_lanes || c.prefill2) && sc.prefill_rows <= uint32_t(c.main.hc_rows)", native)
        block = rank.split("#elif G6_MODEL_DS41\n    if (s2mode)", 1)[1].split("#else\n    if (s2mode)", 1)[0]
        self.assertRegex(
            block,
            re.compile(
                r"const int pf_lanes = ds41::lanes::parse\(getenv\(\"GR8_DS_PF_LANES\"\)\);\s*"
                r"if\s*\(!pf_lanes\)\s*die\(ds41::lanes::kParseError\);\s*"
                r".*?"
                r"bc\.prefill_min\s*=\s*\(uint32_t\)pf_min;\s*"
                r"if\s*\(pf_min\)\s*"
                r"g6s2::ds41_prefill_lanes\s*\(sc,\s*\(uint32_t\)m\.hc_rows,\s*pf_lanes\);",
                re.DOTALL,
            ),
        )

    def test_lanes_one_is_the_single_lane_helper(self):
        lanes = (ROOT / "runtime/s2/ds41_prefill_lanes.h").read_text()
        prefill = (ROOT / "runtime/s2/ds41_prefill.h").read_text()
        self.assertIn("if (lanes == 1) { ds41_prefill_single_lane(sc, rows); return; }", lanes)
        self.assertRegex(
            prefill,
            re.compile(
                r"inline void ds41_prefill_single_lane\(Config& sc, uint32_t rows\) \{\s*"
                r"require\(rows > 0, \"DS41 prefill width must be positive\"\);\s*"
                r"sc\.prefill_rows = rows;\s*"
                r"sc\.two_lanes = false;\s*\}",
                re.DOTALL,
            ),
        )
        # the S2 headers every model includes never pull the lane contract in
        self.assertNotIn("ds41_lanes.h", prefill)
        self.assertNotIn("ds41_prefill_lanes.h", (ROOT / "runtime/s2/sessions.h").read_text())

    def test_one_lane_prefill_seq_is_kept_and_two_lane_is_opt_in(self):
        native = (ROOT / "runtime/s2/ds41_native.cuh").read_text()
        self.assertIn("if (c.prefill && !c.prefill2)   // one chunk: ONE id of the main fabric sequence", native)
        self.assertIn("if (c.prefill2)   // a quantum of W + 1 .. 2W rows", native)
        one = native.split("if (c.prefill && !c.prefill2)", 1)[1].split("ds41::lanes::Counter lane_b;", 1)[0]
        self.assertIn("ds41_make_chunk(p.prompt, first, rows, sc.prefill_rows, sequence, s.req.id, s.key.slot, s.run);", one)
        self.assertIn("c.prefill(chunk);", one)
        self.assertNotIn("prefill2", one.split("\n", 1)[1])


if __name__ == "__main__":
    unittest.main()
