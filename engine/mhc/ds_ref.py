#!/usr/bin/env python3
"""Torch reference for the DeepSeek-V4.1 shifted hyper-connection trunk, split into the stages the engine runs.

  embed -> [prepare(s) -> block s -> consume(s)] x NS -> final

All tensors are float32 holding BF16-representable values wherever the model tensor is BF16. No TF32.
Usage:
  ds_ref.py gen   <case.bin> M H NS V eps seed   write a case file (inputs + expected outputs) for the C++/CUDA tests
  ds_ref.py check [vendor_test_py]               stage split == DeepGEMM fused mhc_reference (loaded read-only), random data
Case file layout (little endian): int32[4] M H NS V | float32 eps | tok int32[M] | embed f32[V*H] |
  NS x { fn f32[24*4H] base f32[24] scale f32[3] norm f32[H] block_out f32[M*H] } | final_norm f32[H] |
  expected: NS x { y f32[M*H] post f32[M*4] comb f32[M*16] pre_next f32[M*4] } | streams_last f32[M*4*H] | final f32[M*H]
"""
import ast
import sys
import numpy as np
import torch

HC = 4
MIX = 24
HC_EPS = 1e-6
POST_SCALE = 2.0
ITERS = 20


def bf16(x):
    return x.to(torch.bfloat16).float()


def sinkhorn(v, repeat, eps):
    v = v.softmax(dim=-1) + eps
    v = v / (v.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        v = v / (v.sum(dim=-1, keepdim=True) + eps)
        v = v / (v.sum(dim=-2, keepdim=True) + eps)
    return v


def embed(tok, table, hc=HC):
    t = table[tok]                                    # [M,H]
    streams = t.unsqueeze(1).repeat(1, hc, 1)         # [M,4,H]
    pre = torch.zeros(tok.shape[0], hc)
    pre[:, 0] = 1.0
    return streams, pre


def rmsnorm_bf16(x, w, eps):
    return bf16(x * torch.rsqrt(x.square().mean(1, keepdim=True) + eps) * w)


@torch.no_grad()
def prepare(T, pre_prev, fn, scale, base, norm_w, eps):
    """T [M,4,H], pre_prev [M,4] -> (pre_next [M,4], post [M,4], comb [M,4,4], y [M,H])"""
    M, hc, H = T.shape
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = T.flatten(1) @ fn.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    mixes = mixes * torch.rsqrt(T.square().sum((1, 2)) / (hc * H) + eps).unsqueeze(1)
    scales = torch.cat((scale[0].expand(hc), scale[1].expand(hc), scale[2].expand(hc * hc)))
    mixes = mixes * scales + base
    pre_next = mixes[:, :hc].sigmoid() + HC_EPS
    post = mixes[:, hc:2 * hc].sigmoid() * POST_SCALE
    comb = sinkhorn(mixes[:, 2 * hc:].view(-1, hc, hc), ITERS, HC_EPS)
    x = bf16((T * pre_prev.unsqueeze(2)).sum(1))
    y = rmsnorm_bf16(x, norm_w, eps)
    return pre_next, post, comb, y


@torch.no_grad()
def consume(T, out, post, comb):
    new = out.unsqueeze(1) * post.unsqueeze(2)
    new = new + torch.einsum("tij,tih->tjh", comb, T)
    return bf16(new)


@torch.no_grad()
def final(T, pre_prev, norm_w, eps):
    x = bf16((T * pre_prev.unsqueeze(2)).sum(1))
    return rmsnorm_bf16(x, norm_w, eps)


def random_case(M, H, NS, V, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    c = {"tok": torch.randint(0, V, (M,), generator=g, dtype=torch.int32),
         "embed": bf16(r(V, H) * 0.7), "layers": []}
    for _ in range(NS):
        c["layers"].append({
            "fn": r(MIX, HC * H) * 0.02,
            "base": r(MIX) * 0.8,
            "scale": torch.rand(3, generator=g) * 0.2 + 0.05,
            "norm": bf16(torch.rand(H, generator=g) * 1.0 + 0.5),
            "block_out": bf16(r(M, H) * 0.5),
        })
    c["final_norm"] = bf16(torch.rand(H, generator=g) * 1.0 + 0.5)
    return c


@torch.no_grad()
def run_chain(c, eps):
    T, pre = embed(c["tok"].long(), c["embed"])
    outs = []
    for L in c["layers"]:
        pre_next, post, comb, y = prepare(T, pre, L["fn"], L["scale"], L["base"], L["norm"], eps)
        outs.append((y, post, comb, pre_next))
        T = consume(T, L["block_out"], post, comb)
        pre = pre_next
    return outs, T, final(T, pre, c["final_norm"], eps)


def gen(path, M, H, NS, V, eps, seed):
    c = random_case(M, H, NS, V, seed)
    outs, T, h = run_chain(c, eps)
    f32 = lambda t: np.ascontiguousarray(t.detach().float().numpy(), dtype="<f4")
    with open(path, "wb") as f:
        f.write(np.array([M, H, NS, V], dtype="<i4").tobytes())
        f.write(np.array([eps], dtype="<f4").tobytes())
        f.write(c["tok"].numpy().astype("<i4").tobytes())
        f.write(f32(c["embed"]).tobytes())
        for L in c["layers"]:
            for k in ("fn", "base", "scale", "norm", "block_out"):
                f.write(f32(L[k]).tobytes())
        f.write(f32(c["final_norm"]).tobytes())
        for y, post, comb, pre_next in outs:
            for t in (y, post, comb, pre_next):
                f.write(f32(t).tobytes())
        f.write(f32(T).tobytes())
        f.write(f32(h).tobytes())
    print(f"wrote {path} M={M} H={H} NS={NS} V={V} eps={eps}")


def load_vendor(path):
    """Compile _sinkhorn_reference and mhc_reference from DeepGEMM's test file without importing deep_gemm."""
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_sinkhorn_reference", "mhc_reference")]
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"torch": torch}
    exec(compile(mod, path, "exec"), ns)
    return ns["mhc_reference"]


@torch.no_grad()
def check(vendor_py):
    mhc_reference = load_vendor(vendor_py)
    M, H, NS, V = 3, 192, 6, 40
    for eps in (1e-20, 1e-6):
        c = random_case(M, H, NS, V, 7)
        T, pre = embed(c["tok"].long(), c["embed"])
        worst = 0.0
        prev = None
        for s, L in enumerate(c["layers"]):
            pre_next, post, comb, y = prepare(T, pre, L["fn"], L["scale"], L["base"], L["norm"], eps)
            if prev is not None:     # fused DeepGEMM op = consume(s-1) + prepare(s) with shifted_prev_mix = pre carried
                Tprev, outp, postp, combp, prep = prev
                ref = mhc_reference(
                    outp, Tprev, postp.unsqueeze(2), combp, L["fn"], L["scale"], L["base"], L["norm"], HC, eps,
                    HC_EPS, POST_SCALE, HC_EPS, ITERS, eps, 1.0, "bf16", 0, prep.unsqueeze(2))
                for name, mine, theirs in (("T", T, ref["new_residual"].float()), ("y", y, ref["y_bf16"].float()),
                                           ("post", post, ref["new_post_mix"].squeeze(2)),
                                           ("comb", comb, ref["new_comb_res_mix"]),
                                           ("pre_next", pre_next, ref["new_prev_mix"].squeeze(2))):
                    d = (mine - theirs).abs().max().item()
                    worst = max(worst, d)
                    assert d == 0.0, f"eps={eps} s={s} {name} differs by {d}"
            prev = (T, L["block_out"], post, comb, pre_next)
            T = consume(T, L["block_out"], post, comb)
            pre = pre_next
        print(f"check eps={eps:g}: stage split == DeepGEMM mhc_reference for {NS-1} fused steps, max diff {worst}")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], *[int(x) for x in sys.argv[3:7]], float(sys.argv[7]), int(sys.argv[8]))
    elif sys.argv[1] == "check":
        if len(sys.argv) < 3:
            sys.exit("usage: ds_ref.py check PATH/TO/DeepGEMM/tests/test_mega_mhc.py")
        check(sys.argv[2])
