"""Slim rank packs (G6_PF_OWN=3): leave out the tensors no OWN=3 engine path reads.

dropped(name, qo_layers):
  Lnnn.kv_b.uk.gv, Lnnn.kv_b.uv.gv   every layer: no decode binder reads them (bind.cuh NAMES); G6AttnPack reads the row-major kv_b.uk / kv_b.uv, kept
  Lnnn.q_b.mx.{codes,scales}         layers in qo_layers: prefill reads the QO pack's L%03d.qo.q_b.* (pf_attn_block_qo), decode reads q_b.mx.gv, kept
  Lnnn.o_proj.mx.{codes,scales}      same, decode reads o_proj.gptq, kept
The MTP layer (78) is not in the QO pack: it keeps all of its tensors. The engine takes a slim pack only when built with
-DPF_QO_DEDUP=1 and booted with G6_PF_OWN=3 + --qo-pack; otherwise bindp::make_layer fails loud on the missing q_b / o_proj.
Pack manifests of slim packs carry slim_qo = true.
"""
import json
import re

from tools.convert.pack import PackWriter

QO_SRC = ('q_b.mx.codes', 'q_b.mx.scales', 'o_proj.mx.codes', 'o_proj.mx.scales')
DEAD_GV = ('kv_b.uk.gv', 'kv_b.uv.gv')
_NAME = re.compile(r'^L(\d{3})\.(.+)$')


def qo_layers(qo_manifest):
    """layers whose full-head tensors the QO pack holds: from its manifest (tools/convert/qopack.py)"""
    if isinstance(qo_manifest, str):
        with open(qo_manifest) as f:
            qo_manifest = json.load(f)
    m = qo_manifest
    need = {f'qo.{t}' for t in QO_SRC}
    have = {}
    for e in m['tensors']:
        r = _NAME.match(e['name'])
        if r and r.group(2) in need:
            have.setdefault(int(r.group(1)), set()).add(r.group(2))
    return {l for l, s in have.items() if s == need}


def dropped(name, qo):
    r = _NAME.match(name)
    if not r:
        return False
    l, key = int(r.group(1)), r.group(2)
    return key in DEAD_GV or (l in qo and key in QO_SRC)


class SlimWriter(PackWriter):
    """PackWriter that skips the tensors dropped() names: convert.py --slim-qo QO_MANIFEST gives the pack slimpack.py would"""

    def __init__(self, *a, qo=(), **k):
        super().__init__(*a, **k)
        self.qo = set(qo)
        self.meta['slim_qo'] = True

    def put(self, name, arr, dtype, **info):
        if not dropped(name, self.qo):
            super().put(name, arr, dtype, **info)
