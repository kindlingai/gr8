"""Host-only source contract for the DS41 M=9..32 step admission."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def check_contract(step: str, moe: str) -> None:
    assert "if(!ds41_prepare_rows_supported(M)) die" in step
    assert "if(M>8 && (table.fn[ds41::ST_ROUTER]" not in step
    assert "if(M>8" not in step[step.index("inline void prepare_ds41("):step.index("// ---- the driver's own ops")]
    assert "constexpr int MTMAX = dsr::MT_MAX" in moe
    assert "rows <= MTMAX" in moe
    assert "k_plan_mt" in moe


class Contract(unittest.TestCase):
    def test_wired_m32_contract(self):
        check_contract((ROOT / "engine/runtime/ds41_step.cuh").read_text(),
                       (ROOT / "engine/kernels/ds41_moe_ops.cuh").read_text())

    def test_old_guard_mutation_is_caught(self):
        step = (ROOT / "engine/runtime/ds41_step.cuh").read_text()
        moe = (ROOT / "engine/kernels/ds41_moe_ops.cuh").read_text()
        step = step.replace("    head_for(m,M);", "    if(M>8 && (table.fn[ds41::ST_ROUTER])) die(\"old guard\");\n    head_for(m,M);", 1)
        with self.assertRaises(AssertionError):
            check_contract(step, moe)


if __name__ == "__main__":
    unittest.main()
