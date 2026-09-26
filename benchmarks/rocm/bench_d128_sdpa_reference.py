import json
import os
import statistics
import subprocess
import sys
import argparse
from pathlib import Path

SEQS = [1024, 2048, 4096]
REPEATS = 3
B, H, D = 1, 16, 128
WARMUP = 20
ROUNDS = 7
ITERS = 30
SEED = 20260923


def worker(seq: int) -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    import torch
    import torch.nn.functional as F

    torch.cuda.set_device(0)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED + seq)
    shape = (B, seq, H, D)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    dout = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)

    def op():
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
        ).transpose(1, 2)

    for _ in range(WARMUP):
        q.grad = k.grad = v.grad = None
        op().backward(dout)
    torch.cuda.synchronize()

    round_ms = []
    for _ in range(ROUNDS):
        events = []
        for _ in range(ITERS):
            q.grad = k.grad = v.grad = None
            out = op()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out.backward(dout)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        round_ms.append(statistics.mean(start.elapsed_time(end) for start, end in events))

    finite = all(torch.isfinite(x.grad).all().item() for x in (q, k, v))
    payload = {
        "seq": seq,
        "median_ms": statistics.median(round_ms),
        "mean_ms": statistics.mean(round_ms),
        "min_ms": min(round_ms),
        "max_ms": max(round_ms),
        "rounds_ms": round_ms,
        "finite_gradients": finite,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }
    print("@@JSON@@" + json.dumps(payload))


def run_worker(script: str, python: str, hip_visible_devices: str, seq: int) -> dict:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = hip_visible_devices
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    process = subprocess.run(
        [python, script, "--worker", str(seq)],
        env=env,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(f"SDPA S={seq} worker failed")
    for line in reversed(process.stdout.splitlines()):
        if line.startswith("@@JSON@@"):
            return json.loads(line[8:])
    raise RuntimeError(f"SDPA S={seq} worker returned no JSON")


def parent() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--hip-visible-devices", default="0")
    parser.add_argument("--output", default="d128_sdpa_reference.json")
    args = parser.parse_args()

    script = str(Path(__file__).resolve())
    results = {}
    summary = {}
    for seq in SEQS:
        runs = []
        for repeat in range(1, REPEATS + 1):
            result = run_worker(script, args.python, args.hip_visible_devices, seq)
            runs.append(result)
            print(f"S={seq} repeat={repeat}: {result['median_ms']:.4f} ms")
        medians = [run["median_ms"] for run in runs]
        results[str(seq)] = runs
        summary[str(seq)] = {
            "median_of_medians_ms": statistics.median(medians),
            "range_ms": [min(medians), max(medians)],
        }

    payload = {
        "config": {
            "implementation": "torch.nn.functional.scaled_dot_product_attention",
            "dispatch": "PyTorch default ROCm SDPA dispatch",
            "repeats": REPEATS,
            "seqs": SEQS,
            "B": B,
            "H": H,
            "D": D,
            "warmup": WARMUP,
            "rounds": ROUNDS,
            "iters": ITERS,
        },
        "results": results,
        "summary": summary,
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved:", output)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        parent()
