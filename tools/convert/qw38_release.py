#!/usr/bin/env python3
"""Assemble an exlr8 v2 release for Qwen3.8-Flash-Next (qwen4_exp_text) from:
  1. Harvested pmcgx expert pieces (--pieces DIR: layer-LLL/eEEE.safetensors, 48 x 513 files)
  2. The BF16 Hugging Face checkpoint (--bf16 DIR: Qwen/Qwen3.8-Flash-Next)

Outputs into --out DIR:
  experts/exlr8-layer-LLL.k{2,3,4}.safetensors     64 KiB-aligned pmcgx routed + shared trellis halves
  experts/exlr8-layer-LLL.index.safetensors        offsets, xxhash64 checksums, su_block/sv_block/svh/suh, signs
  index-slices/tp{1,2,3,4}/k{2,3,4}/rank{r}.safetensors
  exlr8-manifest-{bit200,bit300,bit400}.json       width manifests (model=qw38, hidden_size=2560, nh=5)
  dense/exlr8-dense.safetensors                    BF16 non-expert trunk weights (GDN, GQA, QSA, HC, PLE proj, router, shared)
  dense/exlr8-vocab.safetensors                    BF16 embed_tokens + lm_head
  dense/exlr8-mtp.safetensors                      BF16 MTP layer (full attn + QSA + HC + MoE)
  ple/exlr8-ple-L001.bin + exlr8-ple-L001.json     4 KiB-aligned contiguous BF16 PLE n-gram table (layers.1.ple)
  config.json, generation_config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja, merges.txt

Every file is written atomically via .tmp + os.replace and skipped when already present (unless --force).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import time

import numpy as np
import xxhash

ALIGN = 65536
H = 2560
I = 640
NE = 512
SHARED = 512
NHALF = 5
HBLOCKS = H // 128   # 20
L_MAIN = 48
PROJS = ("gate", "up", "down")
SIDE = {"gate": ("svh", "su_block"), "up": ("svh", "su_block"), "down": ("suh", "sv_block")}
NP = {
    "U8": np.uint8, "I8": np.int8, "I16": np.int16, "F16": np.float16,
    "BF16": np.uint16, "I32": np.int32, "U32": np.uint32, "F32": np.float32,
    "I64": np.int64, "U64": np.uint64,
}
ST_DTYPE = {
    np.dtype(np.uint8): "U8", np.dtype(np.int8): "I8", np.dtype(np.int16): "I16",
    np.dtype(np.float16): "F16", np.dtype(np.int32): "I32", np.dtype(np.uint32): "U32",
    np.dtype(np.float32): "F32", np.dtype(np.int64): "I64", np.dtype(np.uint64): "U64",
}
TOKENIZER_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "merges.txt",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pread_all(fd, n, off):
    chunks = []
    rem = n
    pos = off
    while rem > 0:
        b = os.pread(fd, rem, pos)
        if not b:
            raise EOFError(f"short pread at {pos} ({rem} bytes left)")
        chunks.append(b)
        rem -= len(b)
        pos += len(b)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


class STReader:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        hlen = struct.unpack("<Q", pread_all(self.fd, 8, 0))[0]
        hdr = json.loads(pread_all(self.fd, hlen, 8))
        self.meta = hdr.pop("__metadata__", {})
        self.info = hdr
        self.data_start = 8 + hlen

    def get(self, name):
        info = self.info[name]
        a, b = info["data_offsets"]
        raw = pread_all(self.fd, b - a, self.data_start + a)
        return np.frombuffer(raw, NP[info["dtype"]]).reshape(info["shape"])

    def raw(self, name):
        info = self.info[name]
        a, b = info["data_offsets"]
        return pread_all(self.fd, b - a, self.data_start + a), info["dtype"], info["shape"]

    def close(self):
        os.close(self.fd)


def half_bytes(K):
    # H * 128 * K / 8 = 40960 * K bytes
    return H * 128 * K // 8


def reorder_halves(proj, t):
    """Reorder [1, k16, n16, 16K] int16 trellis into 5 halves of stored tile order.
    gate/up: t shape [160, 40, 16K] -> [5 halves][160 k16][8 col16][32K bytes]
    down:    t shape [40, 160, 16K] -> [5 halves][20 hblocks][8 k16][8 col16][32K bytes]
    """
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    a = np.ascontiguousarray(t).view(np.uint8)
    tb = a.shape[-1]
    if proj == "down":
        r = a.reshape(NHALF, 8, HBLOCKS, 8, tb).transpose(0, 2, 1, 3, 4)
    else:
        r = a.reshape(H // 16, NHALF, 8, tb).transpose(1, 0, 2, 3)
    return [np.ascontiguousarray(r[h]).tobytes() for h in range(NHALF)]


def half_checksum(trellis_bytes, inter, hidden, h):
    x = xxhash.xxh64(seed=0)
    x.update(trellis_bytes)
    x.update(np.ascontiguousarray(inter[128 * h:128 * (h + 1)], "<f2").tobytes())
    x.update(np.ascontiguousarray(hidden, "<f2").tobytes())
    return x.intdigest()


def write_aligned_safetensors(path, specs, meta):
    """Write 64 KiB-aligned safetensors file; returns (file_handle, data_start, {name: rel_off})."""
    # Compute 64 KiB-aligned offsets with __pad.N gaps
    header = {}
    off = 0
    layout = []
    for idx, (name, dtype, shape, nbytes) in enumerate(specs):
        pad = (-off) % ALIGN
        if pad:
            header[f"__pad.{idx}"] = {"dtype": "U8", "shape": [pad], "data_offsets": [off, off + pad]}
            off += pad
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        layout.append((name, off, nbytes))
        off += nbytes
    header["__metadata__"] = {k: str(v) for k, v in meta.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    if 8 + len(hb) > ALIGN:
        data_start = ((8 + len(hb) + ALIGN - 1) // ALIGN) * ALIGN
    else:
        data_start = ALIGN
    hb += b" " * (data_start - 8 - len(hb))
    tmp = path + ".tmp"
    f = open(tmp, "wb")
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    f.flush()
    return f, tmp, data_start, {n: (o, sz) for n, o, sz in layout}


def write_simple_safetensors(path, arrays, meta):
    """Write standard 8-byte-aligned safetensors file (used for index slices & dense packs)."""
    def itemsize(v):
        return np.dtype(NP[v[1]]).itemsize if isinstance(v, tuple) else v.dtype.itemsize
    order = sorted(arrays, key=lambda n: (-itemsize(arrays[n]), n))
    header = {}
    off = 0
    for n in order:
        item = arrays[n]
        if isinstance(item, tuple):
            raw, dtype, shape = item
            nb = len(raw)
        else:
            a = np.ascontiguousarray(item)
            dtype = ST_DTYPE[a.dtype]
            shape = list(a.shape)
            nb = a.nbytes
        header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    header["__metadata__"] = {k: str(v) for k, v in meta.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for n in order:
            item = arrays[n]
            if isinstance(item, tuple):
                f.write(item[0])
            else:
                f.write(np.ascontiguousarray(item).tobytes())
    os.replace(tmp, path)


def build_layer_experts(L, pieces_dir, out_experts, widths, force=False):
    """Pack one layer (512 routed + 1 shared piece files) into k{2,3,4} + index safetensors."""
    idx_path = os.path.join(out_experts, f"exlr8-layer-{L:03d}.index.safetensors")
    k_paths = {K: os.path.join(out_experts, f"exlr8-layer-{L:03d}.k{K}.safetensors") for K in widths}
    if not force and os.path.exists(idx_path) and all(os.path.exists(p) for p in k_paths.values()):
        return L, "cached"

    ldir = os.path.join(pieces_dir, f"layer-{L:03d}")
    # Read signs & metadata from e000
    r0 = STReader(os.path.join(ldir, "e000.safetensors"))
    codebook_str = r0.meta["codebook"]
    signs_in = np.array(r0.get("signs.in"), dtype=np.int8, copy=True)
    signs_out = np.array(r0.get("signs.out"), dtype=np.int8, copy=True)
    common_meta = {
        "format": "exlr8-v2",
        "scheme": r0.meta.get("scheme", "exlr8-e4m3-v2"),
        "layer": str(L),
        "codebook": codebook_str,
        "hessian": r0.meta.get("hessian", "qwen38fn-calib-v1"),
        "rotation": r0.meta.get("rotation", ""),
        "sign_seed": r0.meta.get("sign_seed", ""),
        "gate_up_order": "grouped",
        "down_order": "item128",
        "pad_halves": "false",
    }
    r0.close()

    # Pre-open output trellis writers for every width K so we stream each piece ONCE!
    writers = {}
    side = {}
    for K in widths:
        hb = half_bytes(K)
        specs = []
        for p in PROJS:
            specs.append((f"trellis.{p}.pmcgx", "U8", [NE * NHALF * hb], NE * NHALF * hb))
        for p in PROJS:
            specs.append((f"shared.trellis.{p}.pmcgx", "U8", [1 * NHALF * hb], 1 * NHALF * hb))
        meta = dict(common_meta, contents="trellis", k=str(K), index_file=f"exlr8-layer-{L:03d}.index.safetensors")
        f, tmp, data_start, layout = write_aligned_safetensors(k_paths[K], specs, meta)
        writers[K] = (f, tmp, data_start, layout, hb)
        side[("", K)] = (
            np.zeros((NE, 3, NHALF, 2), np.int64),
            np.zeros((NE, 3, NHALF), np.uint64),
            {
                "gate.svh": np.zeros((NE, I), np.float16), "gate.su_block": np.zeros((NE, HBLOCKS), np.float16),
                "up.svh": np.zeros((NE, I), np.float16), "up.su_block": np.zeros((NE, HBLOCKS), np.float16),
                "down.suh": np.zeros((NE, I), np.float16), "down.sv_block": np.zeros((NE, HBLOCKS), np.float16),
            },
        )
        side[("shared.", K)] = (
            np.zeros((1, 3, NHALF, 2), np.int64),
            np.zeros((1, 3, NHALF), np.uint64),
            {
                "gate.svh": np.zeros((1, I), np.float16), "gate.su_block": np.zeros((1, HBLOCKS), np.float16),
                "up.svh": np.zeros((1, I), np.float16), "up.su_block": np.zeros((1, HBLOCKS), np.float16),
                "down.suh": np.zeros((1, I), np.float16), "down.sv_block": np.zeros((1, HBLOCKS), np.float16),
            },
        )

    # Read each expert once and write to each projection's offset via pwrite!
    for e in range(NE + 1):
        is_sh = (e == SHARED)
        jp = "shared." if is_sh else ""
        row = 0 if is_sh else e
        rd = STReader(os.path.join(ldir, f"e{e:03d}.safetensors"))
        for K in widths:
            f, tmp, data_start, layout, hb = writers[K]
            fd = f.fileno()
            index, checksum, sd = side[(jp, K)]
            for pi, p in enumerate(PROJS):
                tname = f"{jp}trellis.{p}.pmcgx"
                t_off, _ = layout[tname]
                inter_n, hid_n = SIDE[p]
                t = rd.get(f"{jp}k{K}.{p}.trellis")
                inter = rd.get(f"{jp}k{K}.{p}.{inter_n}")[0]
                hid = rd.get(f"{jp}k{K}.{p}.{hid_n}")[0]
                sd[f"{p}.{inter_n}"][row] = inter
                sd[f"{p}.{hid_n}"][row] = hid
                hs = reorder_halves(p, t)
                base_off = data_start + t_off + row * NHALF * hb
                for h, b in enumerate(hs):
                    index[row, pi, h] = (base_off + h * hb, hb)
                    checksum[row, pi, h] = half_checksum(b, inter, hid, h)
                os.pwrite(fd, b"".join(hs), base_off)
        rd.close()

    # Finalize trellis files (ensure exact size + atomic rename)
    for K in widths:
        f, tmp, data_start, layout, hb = writers[K]
        end_off = max(o + sz for o, sz in layout.values())
        f.flush()
        os.ftruncate(f.fileno(), data_start + end_off)
        f.close()
        os.replace(tmp, k_paths[K])

    # Write layer index file
    arrays = {"signs.in": signs_in, "signs.out": signs_out}
    for jp in ("", "shared."):
        for K in widths:
            index, checksum, sd = side[(jp, K)]
            pk = f"{jp}k{K}."
            arrays[pk + "index"] = index
            arrays[pk + "checksum"] = checksum
            for n in ("gate.su_block", "up.su_block", "down.sv_block", "gate.svh", "up.svh", "down.suh"):
                arrays[pk + n] = sd[n]
    idx_meta = dict(
        common_meta,
        contents="index",
        experts=f"0-{NE - 1}",
        widths=",".join(map(str, widths)),
        shared_widths=",".join(map(str, widths)),
        trellis_files=",".join(f"exlr8-layer-{L:03d}.k{K}.safetensors" for K in widths),
        sharding=f"half h of expert e at TP=N goes to rank ({NHALF}e + h + L) mod N; shared = {SHARED}",
    )
    write_simple_safetensors(idx_path, arrays, idx_meta)
    return L, "built"


def build_index_slices(out_dir, widths, tps=(1, 2, 3, 4), force=False):
    out_experts = os.path.join(out_dir, "experts")
    out_slices = os.path.join(out_dir, "index-slices")
    expected = [os.path.join(out_slices, f"tp{n}", f"k{k}", f"rank{r}.safetensors")
                for n in tps for k in widths for r in range(n)]
    if not force and all(os.path.exists(p) for p in expected):
        log("index-slices: cached")
        return

    acc = {(n, k, r): {} for n in tps for k in widths for r in range(n)}
    for L in range(L_MAIN):
        idx = STReader(os.path.join(out_experts, f"exlr8-layer-{L:03d}.index.safetensors"))
        signs = np.stack([idx.get("signs.in"), idx.get("signs.out")]).astype(np.int8)
        for K in widths:
            index = np.concatenate([idx.get(f"k{K}.index"), idx.get(f"shared.k{K}.index")])       # [513, 3, 5, 2]
            check = np.concatenate([idx.get(f"k{K}.checksum"), idx.get(f"shared.k{K}.checksum")]) # [513, 3, 5]
            chan = np.stack([
                np.concatenate([idx.get(f"k{K}.{c}"), idx.get(f"shared.k{K}.{c}")])
                for c in ("gate.svh", "up.svh", "down.suh")
            ], axis=1)                                                                             # [513, 3, 640]
            block = np.stack([
                np.concatenate([idx.get(f"k{K}.{b}"), idx.get(f"shared.k{K}.{b}")])
                for b in ("gate.su_block", "up.su_block", "down.sv_block")
            ], axis=1)                                                                             # [513, 3, 20]
            E = NE + 1
            owner = (NHALF * np.arange(E)[:, None] + np.arange(NHALF)[None, :] + L)
            P = f"L{L:03d}."
            for n in tps:
                ow = owner % n
                for r in range(n):
                    e, h = np.nonzero(ow == r)
                    acc[(n, K, r)][P + "units"] = np.stack([e, h], axis=1).astype(np.int16)
                    acc[(n, K, r)][P + "offset"] = index[e, :, h, 0].astype(np.int64)
                    acc[(n, K, r)][P + "checksum"] = check[e, :, h].astype(np.uint64)
                    acc[(n, K, r)][P + "chan"] = np.stack(
                        [chan[e[:, None], p, h[:, None] * 128 + np.arange(128)] for p in range(3)], axis=1
                    )
                    acc[(n, K, r)][P + "block"] = block
                    acc[(n, K, r)][P + "signs"] = signs
        idx.close()

    for (n, K, r), tensors in acc.items():
        d = os.path.join(out_slices, f"tp{n}", f"k{K}")
        os.makedirs(d, exist_ok=True)
        meta = {
            "format": "exlr8-v2-index-slice", "version": 1, "tp": n, "rank": r, "k": K,
            "layers": ",".join(map(str, range(L_MAIN))), "half_bytes": half_bytes(K),
            "rule": f"({NHALF} * expert + half + layer) mod tp",
            "files": f"experts/exlr8-layer-LLL.k{K}.safetensors",
        }
        write_simple_safetensors(os.path.join(d, f"rank{r}.safetensors"), tensors, meta)
    log(f"index-slices: wrote {len(acc)} files for tp={list(tps)}, widths={widths}")


def write_manifests(out_dir, widths):
    for K in widths:
        name = f"bit{K}00"
        layers_str = []
        for L in range(L_MAIN):
            entry = {
                "gate": [[K] * NHALF for _ in range(NE)],
                "up": [[K] * NHALF for _ in range(NE)],
                "down": [K] * NE,
                "shared": {"gate": [4] * NHALF, "up": [4] * NHALF, "down": 4},
            }
            layers_str.append(f'  "{L}": ' + json.dumps(entry, separators=(",", ":")))
        head = {
            "format": "exlr8-v2-manifest",
            "model": "qw38",
            "name": name,
            "hidden_size": H,
            "nh": NHALF,
            "num_experts": NE,
            "rule": f"every routed expert at K{K}; shared expert K4 (or BF16 from dense)",
        }
        text = "{\n" + "".join(f"{json.dumps(k)}: {json.dumps(v)},\n" for k, v in head.items())
        text += '"layers": {\n' + ",\n".join(layers_str) + "\n}}\n"
        path = os.path.join(out_dir, f"exlr8-manifest-{name}.json")
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
        log(f"manifest: wrote {os.path.basename(path)}")


def build_non_expert_and_ple(bf16_dir, out_dir, skip_ple=False, force=False):
    out_dense = os.path.join(out_dir, "dense")
    out_ple = os.path.join(out_dir, "ple")
    os.makedirs(out_dense, exist_ok=True)
    os.makedirs(out_ple, exist_ok=True)

    with open(os.path.join(bf16_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]

    # Group non-expert tensors by destination file
    dense_names, vocab_names, mtp_names = [], [], []
    ple_shards = {}
    for name in sorted(wmap):
        if name.startswith("model.visual."):
            continue
        if name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            vocab_names.append(name)
        elif name.startswith("mtp."):
            mtp_names.append(name)
        elif ".ple.ple_embedding.ngram_embedding.shard_" in name:
            sh = int(name.rsplit("shard_", 1)[1].split(".")[0])
            ple_shards[sh] = (name, wmap[name])
        elif ".mlp.experts." in name:
            continue   # routed experts are in exlr8-layer-LLL.k*.safetensors
        else:
            dense_names.append(name)

    def pack_group(dst_path, names, label):
        if not force and os.path.exists(dst_path):
            log(f"{label}: cached ({os.path.basename(dst_path)})")
            return
        t0 = time.time()
        by_file = {}
        for n in names:
            by_file.setdefault(wmap[n], []).append(n)
        tensors = {}
        for fname, tlist in sorted(by_file.items()):
            rd = STReader(os.path.join(bf16_dir, fname))
            for n in tlist:
                tensors[n] = rd.raw(n)
            rd.close()
        write_simple_safetensors(dst_path, tensors, {"format": "exlr8-qw38-bf16", "contents": label})
        sz = os.path.getsize(dst_path)
        log(f"{label}: wrote {len(tensors)} tensors ({sz / 1e9:.2f} GB) in {time.time() - t0:.1f}s")

    pack_group(os.path.join(out_dense, "exlr8-dense.safetensors"), dense_names, "dense")
    pack_group(os.path.join(out_dense, "exlr8-vocab.safetensors"), vocab_names, "vocab")
    pack_group(os.path.join(out_dense, "exlr8-mtp.safetensors"), mtp_names, "mtp")

    # PLE contiguous 4 KiB-aligned rowstore
    ple_bin = os.path.join(out_ple, "exlr8-ple-L001.bin")
    ple_json = os.path.join(out_ple, "exlr8-ple-L001.json")
    if not skip_ple and (force or not (os.path.exists(ple_bin) and os.path.exists(ple_json))):
        t0 = time.time()
        nshards = len(ple_shards)
        assert nshards == 128, f"expected 128 PLE shards, got {nshards}"
        total_rows = 0
        dim = None
        shard_rows = []
        tmp = ple_bin + ".tmp"
        with open(tmp, "wb") as out_f:
            for sh in range(nshards):
                tname, fname = ple_shards[sh]
                rd = STReader(os.path.join(bf16_dir, fname))
                info = rd.info[tname]
                assert info["dtype"] == "BF16", f"{tname}: expected BF16, got {info['dtype']}"
                rows, d = info["shape"]
                if dim is None:
                    dim = d
                assert d == dim == 160
                a, b = info["data_offsets"]
                # Stream in 32 MiB chunks
                pos = rd.data_start + a
                rem = b - a
                while rem > 0:
                    take = min(rem, 32 << 20)
                    out_f.write(pread_all(rd.fd, take, pos))
                    pos += take
                    rem -= take
                rd.close()
                total_rows += rows
                shard_rows.append(rows)
                if (sh + 1) % 32 == 0:
                    log(f"ple: {sh + 1}/{nshards} shards ({total_rows:,} rows, {time.time() - t0:.1f}s)")
            # Pad to 4096-byte multiple for O_DIRECT
            cur = out_f.tell()
            pad = (-cur) % 4096
            if pad:
                out_f.write(b"\x00" * pad)
        os.replace(tmp, ple_bin)
        meta = {
            "format": "g6ple-v1",
            "layer": 1,
            "dtype": "bf16",
            "rows": total_rows,
            "dim": dim,
            "row_bytes": dim * 2,
            "file": "exlr8-ple-L001.bin",
            "file_bytes": os.path.getsize(ple_bin),
            "shards": nshards,
            "shard_rows": shard_rows,
        }
        with open(ple_json + ".tmp", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(ple_json + ".tmp", ple_json)
        log(f"ple: wrote {total_rows:,} rows x {dim} bf16 ({os.path.getsize(ple_bin) / 1e9:.2f} GB) in {time.time() - t0:.1f}s")
    else:
        log("ple: cached or skipped")

    # Copy tokenizer & config files
    for fname in TOKENIZER_FILES:
        src = os.path.join(bf16_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src) and (force or not os.path.exists(dst)):
            shutil.copy2(src, dst)
    log("tokenizer/config files ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pieces", required=True, help="harvested pmcgx pieces dir (layer-000..layer-047)")
    ap.add_argument("--bf16", required=True, help="BF16 checkpoint dir (Qwen/Qwen3.8-Flash-Next)")
    ap.add_argument("--out", required=True, help="output release directory")
    ap.add_argument("--widths", default="2,3,4", help="expert widths to pack (default: 2,3,4)")
    ap.add_argument("--tps", default="1,2,3,4", help="TP sizes for index-slices (default: 1,2,3,4)")
    ap.add_argument("--workers", type=int, default=6, help="parallel layer workers")
    ap.add_argument("--skip-ple", action="store_true", help="skip writing 95 GiB PLE bin")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()

    widths = [int(x) for x in a.widths.split(",")]
    tps = tuple(int(x) for x in a.tps.split(","))
    out_experts = os.path.join(a.out, "experts")
    os.makedirs(out_experts, exist_ok=True)

    t0 = time.time()
    log(f"assembling Qwen3.8-Flash-Next exlr8 v2 release into {a.out} (widths={widths}, tps={tps}, workers={a.workers})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build_layer_experts, L, a.pieces, out_experts, widths, a.force) for L in range(L_MAIN)]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            L, st = fut.result()
            done += 1
            if done % 8 == 0 or done == L_MAIN:
                log(f"experts: {done}/{L_MAIN} layers done (latest L{L:03d}: {st}, {time.time() - t0:.1f}s)")

    build_index_slices(a.out, widths, tps=tps, force=a.force)
    write_manifests(a.out, widths)
    build_non_expert_and_ple(a.bf16, a.out, skip_ple=a.skip_ple, force=a.force)
    log(f"ALL DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
