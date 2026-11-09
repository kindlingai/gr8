"""Self-test for ds41_kernels.py (quantizer scales, fp4 RNE ties, blocked GEMM vs naive, sparse_attn vs dense fp64, sinkhorn). Run: python3 ds41_kernels_selftest.py"""
import sys, torch, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds41_kernels as K
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
# 1 act_quant
x = (torch.randn(5, 3, 128) * 3).to(torch.bfloat16)
q, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
sf = s.float()
assert ((sf.log2() % 1) == 0).all(), "scale not pow2"
amax = x.float().reshape(5,3,4,32).abs().amax(-1)
assert (sf * 448 >= amax - 1e-6).all() and (sf / 2 * 448 < amax + 1e-6).all(), "scale not minimal pow2 >= amax/448"
y = x.clone(); K.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True)
deq = (q.float().reshape(5,3,4,32) * sf.unsqueeze(-1)).reshape(5,3,128).to(torch.bfloat16)
assert torch.equal(y, deq)
print("act_quant ok; max rel err", ((y.float()-x.float()).abs() / x.float().abs().clamp_min(1e-3)).max().item())
# 2 fp4 rounding brute force incl ties
grid = torch.tensor([0,.5,1,1.5,2,3,4,6])
codes = torch.arange(8)
def brute(v):
    a = abs(v); d = (grid - a).abs(); m = d.min()
    c = [i for i in range(8) if d[i] == m]
    c = [i for i in c if i % 2 == 0] if len(c) > 1 else c   # tie -> even code
    return math.copysign(grid[c[0]].item(), v)
vs = torch.cat([torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.0,0.,6.,.2500001,.2499999]), torch.rand(2000)*6])
vs = torch.cat([vs, -vs])
got = K._fp4_round(vs)
exp = torch.tensor([brute(v.item()) for v in vs])
assert torch.equal(got, exp), (got[:20], exp[:20])
print("fp4 rne ok")
# 3 gemm vs naive
M,N,Kd = 7, 70, 128
a = torch.randn(M,Kd)*2; w = torch.randn(N,Kd)
aq, asc = K.act_quant(a.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)
wq, wsc = K.act_quant(w.to(torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)   # use as fake weights, scale per row/block -> pass as [N,KB]
# fp8 weight scale is per 32 rows x 32 K; build such a scale
wsc2 = torch.tensor([[2.0**(i%3-1) for _ in range(4)] for i in range(3)]).to(torch.float8_e8m0fnu)
out = K.fp8_gemm(aq, asc, wq, wsc2, torch.float8_e8m0fnu, 32)
ref = torch.zeros(M,N,dtype=torch.float32)
for b in range(4):
    part = (aq.float()[:, b*32:(b+1)*32].double() @ wq.float()[:, b*32:(b+1)*32].double().T).float()
    ref += (part * asc.float()[:, b:b+1]) * wsc2.float().repeat_interleave(32,0)[:N][:, b][None,:]
assert torch.equal(out, ref.to(torch.bfloat16)), "fp8 gemm mismatch"
print("fp8_gemm ok")
# 3b fp4 nibble order: byte 0x21 -> even k = 1 (lut[1]=.5), odd k = 2 (lut[2]=1.0) ; 0x9A: lo 0xA=-1.0 ; hi 9=-0.5
b = torch.tensor([[0x21, 0x9A]], dtype=torch.uint8)
print("fp4 decode", K.decode_fp4(b).tolist())
assert K.decode_fp4(b).tolist() == [[0.5, 1.0, -1.0, -0.5]]
# 4 sparse attn vs dense reference
b_,s_,h_,d_ = 1, 20, 4, 64
q = torch.randn(b_,s_,h_,d_).to(torch.bfloat16); kv = torch.randn(b_,s_,d_).to(torch.bfloat16); sink = torch.randn(h_)
idx = torch.full((b_,s_,100), -1, dtype=torch.int32)
for i in range(s_):
    for j in range(min(i+1, 100)): idx[0,i,j] = i - j if i-j>=0 else -1
o = K.sparse_attn(q, kv, sink.float(), idx, d_**-0.5)
qd, kd = q.double(), kv.double()
ref = torch.zeros(b_,s_,h_,d_, dtype=torch.float64)
for i in range(s_):
    sc = torch.einsum("hd,kd->hk", qd[0,i], kd[0,:i+1]) * d_**-0.5
    mx = torch.maximum(sc.amax(-1), sink.double())
    e = torch.exp(sc - mx[:,None]); den = e.sum(-1) + torch.exp(sink.double() - mx)
    ref[0,i] = (e @ kd[0,:i+1]) / den[:,None]
print("sparse_attn max abs err vs dense fp64:", (o.double()-ref).abs().max().item(), "(bf16 P rounding => ~1e-2 rel expected)")
assert (o.double()-ref).abs().max() < 0.05
# 5 sinkhorn
mixes = torch.randn(1,6,24).float(); pre,post,comb = K.hc_split_sinkhorn(mixes, torch.tensor([.3,.3,.3]), torch.randn(24)*.1)
print("sinkhorn row sums", comb.sum(-1)[0,0].tolist(), "col sums", comb.sum(-2)[0,0].tolist())
print("ALL OK")
