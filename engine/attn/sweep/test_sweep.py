import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

runner, shapes = module("run"), module("cases")

class SweepTests(unittest.TestCase):
    def test_real_shape_weights(self):
        rows = shapes.cases("real")
        self.assertEqual(len(rows), len({r["id"] for r in rows}))
        for label, chunks, end in [("32K",4,32768), ("128K",16,131072)]:
            active = [r for r in rows if r["weights"][label]]
            self.assertEqual(sum(r["weights"][label] for r in active), chunks*40+40)
            self.assertEqual(sum(r["weights"][label] for r in active if r["args"][3]), chunks*19)
            self.assertEqual(sum(r["weights"][label] for r in active if r["args"][0]==8191),21)
            self.assertEqual(sum(r["weights"][label] for r in active if r["args"][0]==1),40)
            self.assertTrue(all(r["args"][1]+r["args"][0]<=end for r in active))
        late=shapes.cases("real",True)
        for label,chunks in [("32K",4),("128K",16)]:
            self.assertEqual(sum(r["weights"][label] for r in late),chunks*21+19+40)

    def test_ptxas_uses_only_selected_kernel(self):
        log="""ptxas info : Function properties for _Zbaseline
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 168 registers, used 1 barriers
ptxas info : Function properties for _Zcandidate
  32 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads
ptxas info : Used 255 registers, used 1 barriers
"""
        self.assertEqual(runner.ptxas_resources(log,"candidate"),dict(stack_bytes=32,spill_store_bytes=8,spill_load_bytes=16,ptxas_regs=255))
        self.assertEqual(runner.ptxas_resources(log,"missing"),{})
        self.assertEqual(runner.ptxas_resources(log,"_Z"),{})

    def test_bad_cases(self):
        for rows in [[],[{"id":"../x","args":[]}],[{"id":"x","args":[],"weights":{"x":-1}}]]:
            with self.assertRaises(ValueError): runner.validate_cases(rows)

    def test_tables_incomplete_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases=[{"id":"a","args":[],"weights":{"32K":2}}, {"id":"b","args":[],"weights":{"32K":3}}]
            rows=[dict(kernel="toy",variant=v,case=c,status="EXACT",mean_us=t,sd_us=1)
                  for v,t in [("off",100),("fast",50)] for c in ("a","b")]
            rows[-1]["status"]="MISMATCH"
            rows.append(dict(kernel="toy",variant="bad",case="a",status="LOW_MEMORY"))
            out=runner.write_tables(Path(tmp),rows,cases,["off","fast","bad"],"off","toy")
            totals={r["variant"]:r for r in out if r["case"].startswith("TOTAL")}
            self.assertEqual(totals["off"]["contribution_32K_ms"],.5)
            self.assertEqual(totals["fast"]["vs_reference_pct"],-50)
            self.assertEqual(totals["fast"]["status"],"MISMATCH")
            self.assertEqual(totals["bad"]["status"],"INCOMPLETE")

    def test_generic_driver_build_isolation_retry_and_case_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for v in ("off","retry","broken"):
                (root/"variants"/v).mkdir(parents=True)
                (root/"variants"/v/"variant.cuh").write_text("// fixture\n")
            (root/"adapter.json").write_text(json.dumps(dict(name="not_attention",header="fixture.h",variants_dir="variants")))
            (root/"cases.json").write_text(json.dumps([dict(id="opaque",args=["this-is-not-a-shape"],weights={"toy":2})]))
            fake=root/"nvcc"
            fake.write_text('''#!/usr/bin/env python3
import sys,pathlib
if '--version' in sys.argv: print('test compiler');sys.exit(0)
out=pathlib.Path(sys.argv[sys.argv.index('-o')+1])
if out.name=='broken': sys.exit(9)
out.write_text("""#!/usr/bin/env python3
import sys,pathlib,json
me=pathlib.Path(sys.argv[0]); marker=me.with_suffix('.attempt')
if me.name=='retry' and not marker.exists():
 marker.write_text('1'); print('LOW_MEMORY fixture',file=sys.stderr); sys.exit(77)
print(json.dumps(dict(status='EXACT',mean_us=10,sd_us=1,mismatch_bytes=0,ptxas_key='fixture',regs=20)))
""")
out.chmod(0o755)
print('ptxas info : Function properties for fixture',file=sys.stderr)
print('  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads',file=sys.stderr)
print('ptxas info : Used 20 registers, used 1 barriers',file=sys.stderr)
''')
            fake.chmod(0o755)
            cmd=[sys.executable,str(HERE/"run.py"),"--kernel",str(root/"adapter.json"),"--cases",str(root/"cases.json"),
                 "--nvcc",str(fake),"--arch","120a","--out",str(root/"out"),"--retry-delay","0"]
            p=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(p.returncode,1,p.stdout+p.stderr)
            with (root/"out/summary.csv").open() as f:
                rows=list(csv.DictReader(f))
            cases={r["variant"]:r for r in rows if r["case"]=="opaque"}
            self.assertEqual(cases["off"]["status"],"EXACT")
            self.assertEqual(cases["retry"]["status"],"EXACT")
            self.assertEqual(cases["broken"]["status"],"BUILD_FAILED")
            self.assertIn("LOW_MEMORY",p.stdout)
            self.assertTrue(all(r["kernel"]=="not_attention" for r in rows))
            self.assertTrue((root/"out/retry.opaque.try2.stdout").exists())

if __name__ == "__main__":
    unittest.main()
