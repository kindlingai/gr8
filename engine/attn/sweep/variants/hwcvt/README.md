# hwcvt: native E2M1 widening

This standalone attention variant changes only compressed-key widening. It
converts two FP4 codes with `cvt.rn.f16x2.e2m1x2`, converts the exact half
values to float and then BF16, and retains the baseline BF16x2 scale multiply.
The QK/PV accumulation, masking, online softmax, layout, and launch geometry
are the baseline's. Window-only cases dispatch the original kernel. There
is no pre-pass, allocation, scratch buffer, or changed input.

`kernel.cuh` is a private copy of **our** `ds41_attn.cuh:941-1131` at
4fb7f0bf, with only the kernel name and dequant store call changed. Its
store helper is our `ds41_attn.cuh:716-734`, selecting `hwcvt::dq4` instead.
No external implementation was copied. The instruction was selected from
NVIDIA's PTX ISA documentation, section Data Movement and Conversion
Instructions: cvt (syntax, FP4 packing order, target requirements):
https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt

## Exhaustive conversion proof (local 5090)

From repository root:

```sh
nvcc=/usr/local/cuda/bin/nvcc
$nvcc -std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a \
  -DG6_MODEL_DS41=1 -I engine -Xptxas -v \
  engine/attn/sweep/variants/hwcvt/proof.cu -o /tmp/hwcvt-proof
/tmp/hwcvt-proof
```

`proof.cu` compares all 65,536 four-code words × all 256 scale bytes directly
against the existing `ds41a::cq_dq4`. This includes signed zero and both NaN
scale encodings. It compares raw BF16 output bits, not a numerical tolerance.
The domain is exhaustive because every `dq4` call ignores code bits above 15,
uses one scale byte, and produces four independent scaled elements.
Each E2M1 value is exactly representable in FP16, FP32, and BF16; the final
scale and its BF16 multiply are unchanged. No floating-point contraction
or softmax reassociation is introduced.

## Attention smoke gate

```sh
python3 engine/attn/sweep/test_sweep.py
python3 engine/attn/sweep/run.py --arch 120a --variants off hwcvt \
  --suite smoke --warmup 1 --reps 2 --iters 1 --out /tmp/hwcvt-smoke
```

Per the sweep owner’s updated scope, full all-suite testing belongs on GB10.
Use fresh output directories. These short shared-5090 timings are hints,
not speed decisions. GB10 performance and runtime compatibility are left
to the sweep owner. The native conversion can also be compiled for sm_121a
locally without accessing a Spark; this is not a GB10 runtime test.
