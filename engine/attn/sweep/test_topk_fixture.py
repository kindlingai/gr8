"""CPU tests exercise the same sampler used by the CUDA fixture."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
HERE = Path(__file__).resolve().parent

class TopkTests(unittest.TestCase):
    def test_patterns_metrics_and_fixture(self):
        fixture = (HERE / 'fixtures.cuh').read_text()
        fill = fixture[fixture.index('static void fill_topk('):fixture.index('\nstruct Built')]
        src = r'''
#include "topk_fixture.h"
#include <cassert>
#include <iostream>
static std::mt19937_64 rng(12345);
''' + fill + r'''
int main() {
    int32_t row[512];
    for (int ratio : {1,2}) for (int visible : {0,1,127,511,512,513,1024,12288,16384}) {
        fill_topk(row, visible * ratio, ratio, visible);
        for (int j=0;j<512;++j) {
            if (j >= visible) assert(row[j] == -1);
            else { assert(row[j]>=0 && row[j]<visible); if(j) assert(row[j-1]<row[j]); }
        }
    }
    // Small-domain frequency check: each 512-of-513 subset should be reachable.
    int omitted[513] = {};
    for(int trial=0;trial<20000;++trial) {
        sweep_topk::uniform(row,513,rng);
        int sum=0; for(int x:row) sum+=x;
        ++omitted[512*513/2-sum];
    }
    for(int x:omitted) assert(x>10 && x<80);
    std::vector<int32_t> positions(128);
    for(int i=0;i<128;++i) positions[i]=24576+i;
    for(const char* pattern : {"uniform","shared","local"}) {
        auto x=sweep_topk::make(positions,1,32768,pattern,7);
        assert(x==sweep_topk::make(positions,1,32768,pattern,7));
        auto m=sweep_topk::metrics(x);
        assert(m.rows==128 && m.within_row_unique_fraction==1 && m.adjacent_pairs==127);
        if(std::string(pattern)=="shared") assert(m.adjacent_overlap_fraction==1);
        if(std::string(pattern)=="local") assert(m.adjacent_overlap_fraction==511./512);
        if(std::string(pattern)=="uniform") assert(m.adjacent_overlap_fraction < .05);
        std::cout<<pattern<<" unique="<<m.within_row_unique_fraction<<" adjacent_overlap="<<m.adjacent_overlap_fraction<<"\n";
    }
    auto x=sweep_topk::make({600,601},1,1000,"shared",1);
    assert(sweep_topk::metrics(x,{0,1}).adjacent_pairs==0);
    bool failed=false;try{sweep_topk::make({0},0,1,"uniform",1);}catch(...){failed=true;}assert(failed);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp=Path(tmp)/'test.cpp'; cpp.write_text(src)
            exe=Path(tmp)/'test'
            subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(HERE),str(cpp),'-o',str(exe)], check=True)
            result=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            print(result.stdout,end='')

    def test_provenance_and_locality_weights(self):
        spec=importlib.util.spec_from_file_location('cases',HERE/'cases.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for suite in ('all','real','smoke','repro','locality'):
            rows=m.cases(suite)
            self.assertTrue(rows)
            self.assertTrue(all(r['input_provenance']=='synthetic_source_derived_shape' and not r['decision_rows'] for r in rows))
        self.assertEqual(len(m.cases('locality')),6)
        self.assertTrue(all(not any(r['weights'].values()) for r in m.cases('locality')))
