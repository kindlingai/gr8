"""g6plandump on a synthetic Flash release: split dense tensors keep their exact bits, and bad splits are refused.
    cargo build --release --manifest-path engine/loader-rs/Cargo.toml && python3 engine/loader-rs/tests/test_flash_plan.py"""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import sys
import numpy as np

META = dict(model='Glm5NextForConditionalGeneration', hidden_size='4096', moe_intermediate_size='2048',
            n_routed_experts='288', num_hidden_layers='45', first_moe_layer='3', tensor_prefix='model.language_model.layers.')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.convert.st import ST  # noqa: E402
DUMP = ROOT/'engine/loader-rs/target/release/g6plandump'


def write(path, arrays, meta):
    head, data, offset = {}, [], 0
    types = {np.dtype('uint16'):'BF16', np.dtype('float32'):'F32', np.dtype('int64'):'I64'}
    for key, value in arrays.items():
        b = value.tobytes()
        head[key] = dict(dtype=types[value.dtype], shape=list(value.shape), data_offsets=[offset,offset+len(b)])
        data.append(b); offset += len(b)
    head['__metadata__'] = meta
    h = json.dumps(head).encode(); h += b' '*(-len(h)%8)
    path.write_bytes(struct.pack('<Q',len(h))+h+b''.join(data))


class PlanTests(unittest.TestCase):
    @unittest.skipUnless(DUMP.exists(), 'build g6plandump --release first')
    def test_split_source_shapes_exact_bits_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'dense').mkdir()
            write(p/'exlr8-vocab.safetensors', {}, META)   # the planner takes the model from the vocabulary file
            (p/'index-slices/tp4/k2').mkdir(parents=True)
            write(p/'index-slices/tp4/k2/rank0.safetensors',{}, {})
            man = p/'widths.json'; man.write_text(json.dumps(dict(META,format='exlr8-v2-manifest',layers={})))
            full = 'model.language_model.layers.0.hc_attn_fn'
            # Signed zero and NaN payloads must survive without any float conversion.
            bits = np.arange(256*16,dtype=np.uint16).reshape(256,16)
            bits[0,:4] = [0x8000,0x7fc1,0x7f81,0xff80]
            parts = []
            for part in range(2):
                path = p/f'dense/exlr8-split-test.p{part:03d}.safetensors'; parts.append(path)
                m = dict(META,layer='0',rows=f'{128*part}-{128*(part+1)}',parts='2',
                    source_names=json.dumps({'hc_attn_fn':full}),source_shapes=json.dumps({'hc_attn_fn':[256,16]}))
                write(path, {'hc_attn_fn.record':np.array([6,256,16,0,0,0,0,0],np.int64),
                             'hc_attn_fn.bf16':bits[128*part:128*(part+1)]},m)
            raw = np.arange(24,dtype=np.float32).reshape(2,3,4)
            m = dict(META,layer='0',source_names=json.dumps({'self_attn.conv':'model.language_model.layers.0.self_attn.conv.weight'}),
                     source_shapes=json.dumps({'self_attn.conv':[2,3,4]}))
            write(p/'dense/exlr8-dense-000.safetensors',{'self_attn.conv.record':np.array([7,2,12,0,0,0,0,0],np.int64),
                'self_attn.conv.fp32':raw.reshape(2,12)},m)
            out = p/'out.safetensors'
            cmd = [str(DUMP),str(p),str(p),str(p/'index-slices'),str(man),'4','0',str(out),'--layers','0','--partial']
            result = subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            f = ST(out)
            self.assertEqual(f.get('L000.hc_attn_fn').tobytes(),bits.tobytes())
            self.assertEqual(f.shape('L000.self_attn.conv.weight'),(2,3,4))
            self.assertEqual(f.get('L000.self_attn.conv.weight').tobytes(),raw.tobytes())
            self.assertIn('L000.hc_attn_fn.gv',f)
            f.close()
            saved = parts[1].read_bytes()
            for key, val, error in [('rows','0-128','split gaps or overlaps'),
                                    ('hidden_size','6144','model identity mismatch')]:
                n = struct.unpack('<Q',saved[:8])[0]
                head = json.loads(saved[8:8+n])
                head['__metadata__'][key] = val
                h = json.dumps(head).encode(); h += b' '*(-len(h)%8)
                parts[1].write_bytes(struct.pack('<Q',len(h))+h+saved[8+n:])
                result = subprocess.run(cmd,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(error,result.stderr)
            parts[1].write_bytes(saved)
            parts[1].unlink()
            result = subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('incomplete split tensor',result.stderr)

if __name__ == '__main__': unittest.main()
