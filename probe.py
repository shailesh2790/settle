"""Stage 0 — measure the laptop, then print what it can build.

    python probe.py
Measures: GPU name, VRAM, bf16 support, bf16 matmul TFLOP/s, device memory bandwidth.
Then applies the capacity planner in common.plan() to those measured numbers."""
import time
import torch
from common import plan, plan_table


def measure():
    if not torch.cuda.is_available():
        print("No CUDA GPU visible. Everything below would run on CPU and be 20-100x slower.")
        return None
    d = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    vram = prop.total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if bf16 else torch.float16

    n = 4096
    a = torch.randn(n, n, device=d, dtype=dt); b = torch.randn(n, n, device=d, dtype=dt)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.perf_counter(); it = 20
    for _ in range(it): a @ b
    torch.cuda.synchronize(); tflops = 2 * n ** 3 * it / (time.perf_counter() - t) / 1e12
    del a, b

    free, _ = torch.cuda.mem_get_info()
    nbytes = int(min(free * 0.3, 1.5e9)) // 4 * 4
    x = torch.empty(nbytes // 4, device=d, dtype=torch.float32); y = torch.empty_like(x)
    for _ in range(3): y.copy_(x)
    torch.cuda.synchronize(); t = time.perf_counter(); it = 20
    for _ in range(it): y.copy_(x)
    torch.cuda.synchronize(); bw = 2 * nbytes * it / (time.perf_counter() - t) / 1e9
    del x, y; torch.cuda.empty_cache()
    return {"name": prop.name, "vram_gb": vram, "bf16": bf16, "tflops": tflops, "bw_gbs": bw}


if __name__ == "__main__":
    m = measure()
    if m:
        print(f"GPU            {m['name']}")
        print(f"VRAM           {m['vram_gb']:.1f} GB")
        print(f"bf16           {'yes' if m['bf16'] else 'no (use fp16)'}")
        print(f"matmul         {m['tflops']:.1f} TFLOP/s  (measured, {('bf16' if m['bf16'] else 'fp16')})")
        print(f"bandwidth      {m['bw_gbs']:.0f} GB/s  (measured device copy)\n")
        print(plan_table(plan(m["vram_gb"], m["bw_gbs"], m["tflops"])))
        print("\nAR ceiling = batch-1 decode upper bound from bandwidth alone; real HF generate lands at "
              "roughly 30-60% of it. LoRA Mtok/h assumes 35% of measured matmul throughput.")
