"""CPU check of the engine's actual compile-time lane-default block."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class LaneDefaultTest(unittest.TestCase):
    def test_default_and_explicit_override(self):
        src = (Path(__file__).resolve().parents[1] / 'g6m4.cu').read_text()
        block = src.split('    // Flash lane default:', 1)[1].split('    const bool s2mode', 1)[0]
        block = block[block.index('#if'):]
        code = '#include <cassert>\nint choose(int MP, int lanes, bool lanes_explicit) {\n' + block + '\nreturn lanes; }\n'
        code += '''int main() {
#if G6_MODEL_FLASH
assert(choose(8192,1,false)==2); assert(choose(4096,1,false)==2);
#else
assert(choose(8192,1,false)==1); assert(choose(4096,1,false)==1);
#endif
assert(choose(0,1,false)==1); assert(choose(16384,1,false)==1);
assert(choose(8192,1,true)==1); assert(choose(16384,2,true)==2);
assert(choose(8192,0,true)==0); // explicit invalid value still reaches existing validation
}'''
        with tempfile.TemporaryDirectory() as d:
            p, exe = Path(d) / 'test.cpp', Path(d) / 'test'
            p.write_text(code)
            for flash in (0, 1):
                subprocess.run(['g++', '-std=c++17', f'-DG6_MODEL_FLASH={flash}', str(p), '-o', str(exe)], check=True)
                subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    unittest.main()
