import importlib.util
import pathlib
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts/check_ds41_moe_topk_boot.py"
spec = importlib.util.spec_from_file_location("checker", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

class BootLineTest(unittest.TestCase):
    def test_accept_and_reject(self):
        with tempfile.TemporaryDirectory() as root:
            a = pathlib.Path(root) / "r0.log"
            b = pathlib.Path(root) / "r1.log"
            a.write_text("# rank 0 ds41 moe topk main 6 decode 2\n")
            b.write_text("# rank 1 ds41 moe topk main 6 decode 2\n")
            self.assertEqual(checker.check([a, b], 6, 2), {0, 1})
            with self.assertRaises(ValueError):
                checker.check([a, b], 6, 3)
            b.write_text(a.read_text())
            with self.assertRaises(ValueError):
                checker.check([a, b], 6, 2)
            b.write_text("not a boot line\n")
            with self.assertRaises(ValueError):
                checker.check([a, b], 6, 2)

if __name__ == "__main__":
    unittest.main()
