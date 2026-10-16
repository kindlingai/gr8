"""Compile the actual serve check lambda against CPU fabric/error mocks."""
from pathlib import Path
import subprocess
import tempfile


def main():
    source = (Path(__file__).resolve().parents[1] / 'runtime/s2/flash_native.cuh').read_text()
    start = source.index('    auto check = [&] {')
    check = source[start:source.index('\n    };', start) + len('\n    };')]
    program = r'''
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
struct Dev { unsigned* error; };
struct Fabric {
    unsigned error = 0;
    mutable int reads = 0;
    Dev dev() const { ++reads; return {const_cast<unsigned*>(&error)}; }
};
struct Transport { Fabric* fabric = nullptr; };
struct Prefill { Transport pool_transport; };
struct Context {
    Prefill prefill, prefill2;
    Fabric main_fabric, draft_fabric, prefill_fabric, prefill_fabric2;
    unsigned main = 0;
    struct Draft { unsigned d = 0; } draft;
} c;
struct Config { bool two_lanes = false; } sc;
unsigned attn_errors(unsigned m) { return m; }
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
''' + check + r'''
    int cases = 0;
    auto expect = [&](const char* error) {
        std::string actual;
        try { check(); } catch (const std::runtime_error& e) { actual = e.what(); }
        assert(actual == error); ++cases;
    };
    expect(""); // owner OFF: null transports, one lane
    sc.two_lanes = true; expect(""); // owner OFF, two lanes
    Fabric owner, owner2;
    c.prefill.pool_transport.fabric = &owner;
    sc.two_lanes = false; expect(""); // owner ON, lane one only
    owner.error = 1; expect("Flash serve pool-ID fabric error");
    owner.error = 0;
    c.prefill2.pool_transport.fabric = &owner2;
    owner2.error = 2; expect(""); assert(owner2.reads == 0); // inactive lane is not read
    sc.two_lanes = true; expect("Flash serve pool-ID fabric error");
    owner2.error = 0; expect("");
    owner.error = 4; expect("Flash serve pool-ID fabric error"); owner.error = 0;
    for (auto* f : {&c.main_fabric, &c.draft_fabric, &c.prefill_fabric, &c.prefill_fabric2}) {
        f->error = 1; expect("Flash serve fabric/attention error"); f->error = 0;
    }
    c.main = 1; expect("Flash serve fabric/attention error"); c.main = 0;
    c.draft.d = 1; expect("Flash serve fabric/attention error");
    std::printf("flash_serve_check: PASS %d CPU mock cases (actual serve lambda)\n", cases);
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        cpp, binary = Path(tmp) / 'check.cpp', Path(tmp) / 'check'
        cpp.write_text(program)
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        str(cpp), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
