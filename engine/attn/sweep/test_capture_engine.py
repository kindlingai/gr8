from pathlib import Path
import unittest
HERE = Path(__file__).resolve().parent

class CaptureEngineTests(unittest.TestCase):
    def test_default_compile_guard(self):
        self.assertIn('#define GR8_ATTN_SWEEP_CAPTURE 0', (HERE/'capture_build.h').read_text())
        for filename in ['engine/attn/ds41_attn.cu','engine/attn/ds41_attn_ops.cuh','engine/runtime/g6m4.cu']:
            source=(HERE.parents[2]/filename).read_text()
            self.assertIn('#if defined(GR8_ATTN_SWEEP_CAPTURE) && GR8_ATTN_SWEEP_CAPTURE',source)
