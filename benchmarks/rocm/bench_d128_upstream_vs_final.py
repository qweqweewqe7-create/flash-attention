import os
import sys
import json
import statistics
import subprocess
import argparse
from pathlib import Path

SEQS = [1024, 2048, 4096]
PAIRS = 6

B, H, D = 1, 16, 128
WARMUP = 20
ROUNDS = 7
ITERS = 30
SEED = 20260923

def pct(x, ref):
    return (x / ref - 1.0) * 100.0

def worker(name, target, seq, expected_sha):
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    import hashlib
    import torch

    sys.path.insert(0, target)
    import flash_attn
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func

    torch.cuda.set_device(0)
    device = "cuda"
    dtype = torch.bfloat16

    mod = str(Path(flash_attn.__file__).resolve())
    ext = str(Path(flash_attn_2_cuda.__file__).resolve())
    target_abs = str(Path(target).resolve()).lower()

    if not mod.lower().startswith(target_abs):
        raise RuntimeError(f"wrong flash_attn import: {mod}")
    if not ext.lower().startswith(target_abs):
        raise RuntimeError(f"wrong extension import: {ext}")

    with open(ext, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest().upper()
    if sha != expected_sha:
        raise RuntimeError(f"{name}: wrong SHA {sha}")

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + seq)

    shape = (B, seq, H, D)
    q0 = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    k0 = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    v0 = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    dout = torch.randn(shape, device=device, dtype=dtype, generator=gen)

    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)

    def op():
        return flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )

    for _ in range(WARMUP):
        q.grad = None
        k.grad = None
        v.grad = None
        out = op()
        out.backward(dout)

    torch.cuda.synchronize()

    round_ms = []

    for _ in range(ROUNDS):
        events = []
        for _ in range(ITERS):
            q.grad = None
            k.grad = None
            v.grad = None

            # Forward is enqueued before the start event, so the event timing
            # covers backward only on the same stream.
            out = op()

            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            out.backward(dout)
            en.record()
            events.append((st, en))

        torch.cuda.synchronize()
        vals = [st.elapsed_time(en) for st, en in events]
        round_ms.append(sum(vals) / len(vals))

    payload = {
        "name": name,
        "seq": seq,
        "median_ms": statistics.median(round_ms),
        "mean_ms": statistics.mean(round_ms),
        "min_ms": min(round_ms),
        "max_ms": max(round_ms),
        "rounds_ms": round_ms,
        "sha": sha,
    }

    print("@@JSON@@" + json.dumps(payload))
    return 0

def run_one(script, python, name, target, expected_sha, seq, env):
    p = subprocess.run(
        [python, script, "--worker", name, target, str(seq), expected_sha],
        env=env,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"{name} S{seq} failed")

    for line in reversed(p.stdout.splitlines()):
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    raise RuntimeError(f"{name} S{seq}: missing JSON")

def parent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--upstream-target", required=True)
    parser.add_argument("--final-target", required=True)
    parser.add_argument("--upstream-sha256", required=True)
    parser.add_argument("--final-sha256", required=True)
    parser.add_argument("--hip-visible-devices", default="0")
    parser.add_argument("--output", default="d128_upstream_vs_final_causal_abba.json")
    args = parser.parse_args()

    script = str(Path(__file__).resolve())
    targets = {"UPSTREAM": args.upstream_target, "FINAL": args.final_target}
    expected = {
        "UPSTREAM": args.upstream_sha256.upper(),
        "FINAL": args.final_sha256.upper(),
    }

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = args.hip_visible_devices
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("ROCR_VISIBLE_DEVICES", None)

    all_results = {str(s): [] for s in SEQS}

    for seq in SEQS:
        print()
        print("=" * 100)
        print(f"S={seq} causal backward-only — {PAIRS} AB/BA fresh-process pairs")
        print("=" * 100)

        for i in range(PAIRS):
            order = ["UPSTREAM", "FINAL"] if i % 2 == 0 else ["FINAL", "UPSTREAM"]
            pair = {"pair": i + 1, "order": order, "results": {}}

            print(f"\npair {i+1}/{PAIRS} order={' -> '.join(order)}")

            for name in order:
                x = run_one(
                    script,
                    args.python,
                    name,
                    targets[name],
                    expected[name],
                    seq,
                    env,
                )
                pair["results"][name] = x
                print(f"  {name:7s}: {x['median_ms']:.4f} ms")

            upstream = pair["results"]["UPSTREAM"]["median_ms"]
            final = pair["results"]["FINAL"]["median_ms"]
            pair["delta_pct"] = pct(final, upstream)
            print(f"  FINAL-UPSTREAM: {pair['delta_pct']:+.2f}%")
            all_results[str(seq)].append(pair)

    print()
    print("=" * 100)
    print("ORDER-BALANCED SUMMARY")
    print("=" * 100)

    summary = {}

    for seq in SEQS:
        pairs = all_results[str(seq)]
        upstream_vals = [p["results"]["UPSTREAM"]["median_ms"] for p in pairs]
        final_vals = [p["results"]["FINAL"]["median_ms"] for p in pairs]
        deltas = [p["delta_pct"] for p in pairs]

        ab = [p["delta_pct"] for p in pairs if p["order"][0] == "UPSTREAM"]
        ba = [p["delta_pct"] for p in pairs if p["order"][0] == "FINAL"]

        upstream_med = statistics.median(upstream_vals)
        final_med = statistics.median(final_vals)
        aggregate_delta = pct(final_med, upstream_med)

        print(f"\nS={seq}")
        print(f"  UPSTREAM median-of-medians: {upstream_med:.4f} ms")
        print(f"  FINAL    median-of-medians: {final_med:.4f} ms")
        print(f"  aggregate FINAL-UPSTREAM  : {aggregate_delta:+.2f}%")
        print(f"  paired delta median      : {statistics.median(deltas):+.2f}%")
        print(f"  UPSTREAM->FINAL median    : {statistics.median(ab):+.2f}%")
        print(f"  FINAL->UPSTREAM median    : {statistics.median(ba):+.2f}%")
        print(f"  paired delta range       : {min(deltas):+.2f}% .. {max(deltas):+.2f}%")

        summary[str(seq)] = {
            "upstream_median_of_medians": upstream_med,
            "final_median_of_medians": final_med,
            "aggregate_delta_pct": aggregate_delta,
            "paired_delta_median_pct": statistics.median(deltas),
            "upstream_then_final_delta_median_pct": statistics.median(ab),
            "final_then_upstream_delta_median_pct": statistics.median(ba),
            "paired_delta_range_pct": [min(deltas), max(deltas)],
        }

    out = {
        "config": {
            "comparison": "upstream_vs_final",
            "flash_attention_commit": "0251105a2fb19d2957484b7f023cd8c115286ced",
            "composable_kernel_commit": "c56c6750d0fc54ed771d532cc92c316423449614",
            "upstream_pyd_sha256": expected["UPSTREAM"],
            "final_pyd_sha256": expected["FINAL"],
            "pairs": PAIRS,
            "seqs": SEQS,
            "B": B, "H": H, "D": D,
            "warmup": WARMUP,
            "rounds": ROUNDS,
            "iters": ITERS,
        },
        "results": all_results,
        "summary": summary,
    }

    out_path = Path(args.output).resolve()
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("saved:", out_path)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]))
    parent()
