# FlashAttention ROCm gfx1201 D=128 upstream comparison

## Result

The final D=128 build was compared against a clean build of the exact upstream revisions from which the work started.

This branch is experimental and narrowly optimized. The product-dual path is selected for batch-mode BF16 backward with Dq=Dv=128, 32x32 tiling, no padding, no bias, no dropout, no deterministic mode, and no transpose-load path. Other configurations remain on the upstream CK dispatch paths.

| Causal backward-only | Upstream median | Final median | PyTorch SDPA | Final vs upstream | Final vs SDPA |
|---:|---:|---:|---:|---:|---:|
| S=1024 | 2.743659 ms | 0.491232 ms | 0.479393 ms | 82.10% lower / 5.585x | 2.47% higher |
| S=2048 | 9.464656 ms | 1.043360 ms | 1.444823 ms | 88.98% lower / 9.071x | 27.79% lower / 1.385x |
| S=4096 | 35.673989 ms | 3.618931 ms | 4.761161 ms | 89.86% lower / 9.858x | 23.99% lower / 1.316x |

The paired median reductions were 82.09%, 88.98%, and 89.85%, respectively. The AB and BA order-specific medians were close at every sequence length, so the result was not materially affected by launch order.

PyTorch SDPA is included as a compact secondary reference. It used `torch.nn.functional.scaled_dot_product_attention` with PyTorch's default ROCm dispatch; flash, memory-efficient, and math SDPA backends were all enabled. Each displayed SDPA value is the median of three fresh-process run medians using the same warmup, round, iteration, dtype, shape, and backward-only timing method. The final build was effectively tied with SDPA at S=1024, then faster at S=2048 and S=4096.

## Fixed environment

- GPU: AMD Radeon RX 9070 XT (`gfx1201`)
- Python: 3.12.10
- PyTorch: `2.15.0a0+rocm10.1.0a20260822`
- HIP: `7.16.26332`
- dtype: BF16
- shape: `B=1, H=16, D=128`
- sequence lengths: 1024, 2048, 4096
- warmup: 20 backward passes per fresh process
- measurement: 7 rounds x 30 iterations
- pairing: 6 fresh-process AB/BA pairs per sequence length
- metric: backward-only GPU event time; forward was enqueued before the start event on the same stream

## Source identity

- FlashAttention upstream commit: `0251105a2fb19d2957484b7f023cd8c115286ced`
- composable_kernel upstream commit: `c56c6750d0fc54ed771d532cc92c316423449614`
- upstream extension SHA-256: `96B84DD18F239EC1107510853E867494EE73E303ABDFE9243A2370ACAC966F2F`
- final extension SHA-256: `532C57C8D94DF814E3BC82466215022B920342C238339B570046C0D9E784E0AC`

The current upstream-facing CK branch lives under
[`projects/composablekernel`](https://github.com/qweqweewqe7-create/rocm-libraries/tree/gfx1201-fa-d128/projects/composablekernel)
in the ROCm Libraries monorepo. The standalone `composable_kernel` fork is kept
only as a build-compatible mirror for FlashAttention's existing submodule
layout.

The upstream kernel and dispatch logic were left unchanged. A three-line import-path shim was applied only to let pristine CK's generator run under Windows embeddable Python, which does not add the invoked script directory to `sys.path`.

## Correctness gate

Before benchmarking, the upstream build passed all 10 dense D=128 cases used by the existing verification harness. These covered causal and non-causal attention, unequal Q/K lengths, GQA, and MQA. Output and gradient relative-L2 values were checked against PyTorch SDPA and all cases passed the harness thresholds.

The final extension is the same SHA-256-identified binary that previously passed its 23-case scope validation: 22 ordinary passes plus the expected explicit rejection of unsupported softcap.

## Attached raw data

- [`d128_upstream_vs_final_causal_abba.json`](../benchmarks/rocm/results/d128_upstream_vs_final_causal_abba.json): every fresh-process pair, round timings, hashes, and aggregate summary
- [`upstream_d128_correctness.json`](../benchmarks/rocm/results/upstream_d128_correctness.json): upstream dense correctness metrics
- [`d128_sdpa_reference.json`](../benchmarks/rocm/results/d128_sdpa_reference.json): three fresh-process PyTorch SDPA reference runs per sequence length

## Reproduce the timing harness

Build and unpack both the exact upstream revision and this branch into separate target directories, then run:

```powershell
python benchmarks/rocm/bench_d128_upstream_vs_final.py `
  --upstream-target D:\path\to\upstream-target `
  --final-target D:\path\to\final-target `
  --upstream-sha256 96B84DD18F239EC1107510853E867494EE73E303ABDFE9243A2370ACAC966F2F `
  --final-sha256 532C57C8D94DF814E3BC82466215022B920342C238339B570046C0D9E784E0AC `
  --hip-visible-devices 1 `
  --output d128_upstream_vs_final_causal_abba.json

python benchmarks/rocm/bench_d128_sdpa_reference.py `
  --hip-visible-devices 1 `
  --output d128_sdpa_reference.json
```

The SHA arguments deliberately make the benchmark refuse to compare the wrong extension binaries. Change the device selector and hashes when reproducing on another build.

Lower latency is better. Percentage values describe latency reduction of the final build relative to upstream; speedup is `upstream_time / final_time`.
