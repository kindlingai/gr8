#!/usr/bin/env python3
"""Source contract: the S2 scheduler knobs reach the DS41 serve Config. g6m4.cu builds the DS41 scheduler Config in its own
'#elif G6_MODEL_DS41' block (auto& sc = bc.scheduler); the '#else' block is GLM's. A knob parsed only in
the GLM block does nothing on g6m4_ds41, so each knob must be parsed inside the DS41 block.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
KNOBS = {"GR8_S2_TAIL_FIRST": r"sc\.tail_first\s*=", "GR8_S2_NOTICE_MAX": r"sc\.notice_max\s*="}


def ds41_block(src):
    m = re.search(r"^#elif G6_MODEL_DS41\n(?:(?!^#(?:elif|else|endif)).*\n)*?\s*auto& sc = bc\.scheduler;\n", src, re.M)
    assert m, "DS41 serve block (auto& sc = bc.scheduler) not found"
    end = re.compile(r"^#(?:elif|else|endif)", re.M).search(src, m.end())
    return src[m.start():end.start()]


class Ds41S2Knobs(unittest.TestCase):
    def test_knobs_parsed_in_ds41_block(self):
        block = ds41_block((ROOT / "runtime" / "g6m4.cu").read_text())
        for env, field in KNOBS.items():
            self.assertTrue('"%s"' % env in block, "%s is not parsed in the DS41 serve block" % env)
            self.assertRegex(block, field, "%s does not set the DS41 Config field" % env)

    def test_boot_echo(self):
        # a boot checker greps '# rank <R> ds41 s2 scheduler tail_first 1 notice_max <k>' on every rank
        block = ds41_block((ROOT / "runtime" / "g6m4.cu").read_text())
        self.assertRegex(block, r'"# rank %d ds41 s2 scheduler tail_first %d notice_max %u\\n", R, sc\.tail_first \? 1 : 0, sc\.notice_max\)')


if __name__ == "__main__":
    unittest.main()
