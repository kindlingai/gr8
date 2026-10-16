"""CPU source-placement checks; GPU output identity is a separate gate."""
from pathlib import Path
import unittest

class ResetPlacement(unittest.TestCase):
    def test_offline_snapshot(self):
        source = (Path(__file__).resolve().parents[1] / 'runtime/g6m4.cu').read_text()
        self.assertIn('bool spec_reset_per_request = false;', source)
        self.assertIn('else if (a == "--spec-reset-per-request") spec_reset_per_request = true;', source)
        self.assertIn('if (spec_reset_per_request && s2mode) die(', source)
        snapshot = source.index('const SpecLen spec_initial = spec;')
        self.assertLess(source.index('spec.flash_tp2();'), snapshot)
        loop = source.index('for (size_t q = 0; !s2mode && q < plans.size(); ++q)')
        reset = source.index('if (spec_reset_per_request) spec = spec_initial;')
        self.assertLess(snapshot, loop)
        self.assertLess(loop, reset)
        self.assertLess(reset, source.index('Plan sp = plans[q];', loop))
        self.assertEqual(source.count('spec = spec_initial;'), 1)
        self.assertIn('#if G6_MODEL_FLASH\n        if (spec_reset_per_request) spec = spec_initial;\n#endif', source)

if __name__ == '__main__':
    unittest.main()
