from __future__ import annotations

import os
import sys
import time

import torch


def _fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def main() -> int:
    print("=== PyTorch CUDA Smoke Test ===")
    print("Python:", sys.version.replace("\n", " "))
    print("Torch version:", torch.__version__)
    print("Torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nCUDA is NOT available to PyTorch.")
        print("Common causes:")
        print("- Installed CPU-only torch wheel")
        print("- Driver/toolkit mismatch for the installed torch wheel")
        print("- Running inside a container without GPU passthrough")
        print("- CUDA_VISIBLE_DEVICES is empty/invalid")
        print("\nEnvironment:")
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        return 1

    n = torch.cuda.device_count()
    print("CUDA device count:", n)

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(
            f"Device {i}: {props.name} | "
            f"cc {props.major}.{props.minor} | "
            f"total_mem {_fmt_bytes(props.total_memory)}"
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Warmup + correctness check
    print("\nRunning a small GPU matmul correctness test...")
    a = torch.randn((1024, 1024), device=device, dtype=torch.float32)
    b = torch.randn((1024, 1024), device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    t1 = time.time()

    # Compare against CPU for correctness (within tolerance)
    c_cpu = (a.detach().cpu() @ b.detach().cpu())
    max_abs_err = (c.detach().cpu() - c_cpu).abs().max().item()

    print(f"Matmul time (1024x1024): {(t1 - t0) * 1000:.2f} ms")
    print(f"Max abs error vs CPU: {max_abs_err:.6e}")

    if max_abs_err > 1e-3:
        print("FAIL: Numerical mismatch is larger than expected.")
        return 2

    # Memory stats
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print("\nCUDA memory after matmul:")
    print("  allocated:", _fmt_bytes(int(allocated)))
    print("  reserved :", _fmt_bytes(int(reserved)))

    # Simple kernel / reduction
    print("\nRunning a simple reduction on GPU...")
    x = torch.rand((10_000_000,), device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    s = x.sum()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Sum: {float(s):.6f} | time: {(t1 - t0) * 1000:.2f} ms")

    print("\nPASS: PyTorch can use CUDA and execute on the GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
