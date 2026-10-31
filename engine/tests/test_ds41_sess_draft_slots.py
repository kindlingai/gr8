#!/usr/bin/env python3
"""Source contract: SESS > 1 with DSpark. g6m4 gives the draft Model the S2 slots (DS.ds_slots = m.ds_slots) and
ds41dsp_ops::build requires draft.ds_slots == main.ds_slots, so build_ds41 must accept ds_slots 1..8 for the draft Model too.
A guard `draft && m.ds_slots != 1` in build_ds41 makes every rank die at boot with --sess 8 and MTP.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SessDraftSlots(unittest.TestCase):
    def test_build_accepts_draft_slots(self):
        src = (ROOT / "runtime" / "ds41_step.cuh").read_text()
        self.assertRegex(src, r"ds41slots::slots_error\(\s*m\.ds_slots\s*\)", "build_ds41 does not call ds41slots::slots_error(m.ds_slots)")
        self.assertNotRegex(src, r"draft\s*&&\s*m\.ds_slots", "build_ds41 still refuses ds_slots > 1 for the draft Model")
        hdr = (ROOT / "runtime" / "ds41_slots.h").read_text()
        self.assertNotIn("draft", hdr.split("inline const char* slots_error")[1].split("}")[0])

    def test_launcher_and_ops_agree(self):
        g6 = (ROOT / "runtime" / "g6m4.cu").read_text()
        self.assertRegex(g6, r"DS\.ds_slots\s*=\s*m\.ds_slots\s*;")
        ops = (ROOT / "runtime" / "ds41_dspark_ops.cuh").read_text()
        self.assertRegex(ops, r"dsp_nsess_error\(nsess, main\.ds_slots, draft\.ds_slots, k, MX, ::ds41dsp::MARKOV_MAXS, BLOCK\)")


if __name__ == "__main__":
    unittest.main()
