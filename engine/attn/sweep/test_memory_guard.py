"""GPU regression gate; skipped in CPU-only discovery.

Compile test_memory_guard.cu as shown in its header, then run:
  SWEEP_MEMORY_TEST_BIN=/absolute/path/to/binary python3 -m unittest discover \
      -s engine/attn/sweep -p test_memory_guard.py -v
Every subprocess takes the shared 5090 lock, including negative tests.
"""
import os
from pathlib import Path
import subprocess
import unittest

BINARY = os.environ.get("SWEEP_MEMORY_TEST_BIN")

@unittest.skipUnless(BINARY, "set SWEEP_MEMORY_TEST_BIN to run the locked GPU gate")
class MemoryGuardTests(unittest.TestCase):
    def run_mode(self, mode):
        self.assertTrue(Path(BINARY).is_file(), BINARY)
        result = subprocess.run(
            ["flock", "/tmp/attn-sweep-5090.lock", BINARY, mode],
            text=True, capture_output=True, timeout=180)
        print(result.stdout, end="", flush=True)
        return result

    def test_off_refresh_rotation_and_allocation_coverage(self):
        result = self.run_mode("sanity")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("status=EXACT integrity=CLEAN"), 4)
        self.assertIn("coverage guard_mutations=40", result.stdout)
        self.assertIn("status=PASS", result.stdout)

    def negative(self, mode, diagnostic):
        result = self.run_mode(mode)
        # A crash/CUDA failure is not successful detection: the check must
        # return corruption to the caller, which deliberately exits 1.
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("output_status=EXACT status=MEMORY_CORRUPTION", result.stdout)
        self.assertIn(diagnostic, result.stderr)
        print(result.stderr, end="", flush=True)

    def test_input_mutation_must_fail(self):
        self.negative("mutate-input", "INTEGRITY input name=q")

    def test_output_overrun_must_fail(self):
        self.negative("overrun", "name=candidate_output side=trailing")

    def test_output_underrun_must_fail(self):
        self.negative("underrun", "name=candidate_output side=leading")

    def test_adapter_scratch_overrun_must_fail(self):
        self.negative("scratch-overrun", "name=integrity_hash_scratch side=trailing")

    def test_unsnapshotted_check_must_fail(self):
        self.negative("missing-snapshot", "INTEGRITY missing input snapshot")

if __name__ == "__main__":
    unittest.main()
