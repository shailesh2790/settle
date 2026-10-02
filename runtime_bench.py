"""Week 1 of the pocket runtime: accuracy, speed and ENERGY per correct answer on a local 4-bit model.

Runs the MBPP test set (257 tasks, same prompts, same Docker sandbox as bench_decode.py / phase2.py) through
either the llama.cpp server (4-bit GGUF) or Hugging Face transformers (bf16, our setup so far), and records
GPU energy for every answer from the GPU's own energy counter (NVML).

    python runtime_bench.py --engine llamacpp --k 8 --out runs/runtime/llamacpp_q4.jsonl
    python runtime_bench.py --engine hf --k 8 --limit 60 --out runs/runtime/hf_bf16.jsonl

Energy covers the GPU only (the CPU is not metered on this laptop), which flatters the Python-heavy hf engine.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import pynvml, requests
from common import load_mbpp, user_message, extract_code, check

ROOT = os.path.dirname(os.path.abspath(__file__))
GGUF = os.path.join(ROOT, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
SERVER = os.path.join(ROOT, "tools", "llama.cpp", "llama-server.exe")
SYSTEM = "You are a careful Python programmer."
PORT = 8090
pynvml.nvmlInit(); GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
energy_j = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(GPU) / 1000.0


def idle_watts(secs=10):
    e0 = energy_j(); time.sleep(secs); return (energy_j() - e0) / secs


# ---------------------------------------------------------------- engines
class LlamaCpp:
    name = "llama.cpp Q4_K_M"

    def __init__(self, k):
        self.proc = subprocess.Popen([SERVER, "-m", GGUF, "-ngl", "99", "--port", str(PORT), "-c", str(2048 * max(1, k)),
                                      "--parallel", str(max(1, k)), "--log-disable"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200: break
            except requests.RequestException: pass
            time.sleep(0.5)
        else: raise SystemExit("llama-server did not start")
        self.pool = ThreadPoolExecutor(max(1, k))

    def _one(self, task, temperature, seed, max_new):
        r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", timeout=300, json={
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task)}],
            "temperature": temperature, "top_p": 0.95 if temperature > 0 else 1.0, "seed": seed, "max_tokens": max_new}).json()
        return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]

    def generate(self, task, k, temperature, max_new=512):
        if k == 1: return [self._one(task, temperature, 0, max_new)]
        return list(self.pool.map(lambda s: self._one(task, temperature, s, max_new), range(k)))   # k requests in parallel

    def close(self): self.proc.terminate()


class HF:
    name = "transformers bf16"

    def __init__(self, k):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()

    def generate(self, task, k, temperature, max_new=512):
        text = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task)}],
                                            tokenize=False, add_generation_prompt=True)
        ids = self.tok([text], return_tensors="pt").to("cuda")
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k) if temperature > 0 else dict(do_sample=False)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id, **kw)
        res = []
        for row in out[:, ids["input_ids"].shape[1]:]:
            eos = (row == self.tok.eos_token_id).nonzero(); n = int(eos[0]) if len(eos) else int(row.shape[0])
            res.append((self.tok.decode(row[:n], skip_special_tokens=True), n))
        return res

    def close(self): pass


# ---------------------------------------------------------------- benchmark
def measured(fn):
    e0, t0 = energy_j(), time.perf_counter(); out = fn(); return out, time.perf_counter() - t0, energy_j() - e0


def main(a):
    tasks = load_mbpp(a.limit)
    done = {json.loads(l)["task"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
    if sys.platform == "win32":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    idle = idle_watts(); print(f"GPU idle power: {idle:.1f} W", flush=True)
    eng = (LlamaCpp if a.engine == "llamacpp" else HF)(a.k)
    checker = ThreadPoolExecutor(4)
    try:
        with open(a.out, "a", encoding="utf-8") as log:
            for i, task in enumerate(tasks):
                if task["id"] in done: continue
                (g,), gs, gj = measured(lambda: eng.generate(task, 1, 0.0))
                row = {"task": task["id"], "engine": eng.name, "idle_w": round(idle, 2),
                       "greedy_hidden": check(task, extract_code(g[0], task))[0], "greedy_secs": round(gs, 3),
                       "greedy_joules": round(gj, 2), "greedy_tokens": g[1]}
                if a.k > 1:
                    samples, bs, bj = measured(lambda: eng.generate(task, a.k, a.temperature))
                    codes = [extract_code(t, task) for t, _ in samples]
                    vis = list(checker.map(lambda c: check(task, c, visible_only=True)[0], codes))
                    pick = vis.index(True) if any(vis) else 0
                    hid = list(checker.map(lambda c: check(task, c)[0], codes))
                    row.update(k=a.k, bestof_hidden=hid[pick], any_hidden=any(hid), bestof_secs=round(bs, 3),
                               bestof_joules=round(bj, 2), bestof_tokens=sum(n for _, n in samples))
                log.write(json.dumps(row) + "\n"); log.flush()
                if (i + 1) % 25 == 0: print(f"{i + 1}/{len(tasks)}", flush=True)
    finally:
        eng.close()
    summarize(a.out)


def summarize(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows); idle = rows[0]["idle_w"]; f = lambda k: sum(r[k] for r in rows if k in r)
    print(f"\n{path}: {rows[0]['engine']}, {n} tasks, GPU idle {idle:.1f} W")
    for mode, ok, secs, j, tok in (("one greedy answer", "greedy_hidden", "greedy_secs", "greedy_joules", "greedy_tokens"),
                                   (f"best of {rows[0].get('k', 8)}", "bestof_hidden", "bestof_secs", "bestof_joules", "bestof_tokens")):
        if ok not in rows[0]: continue
        solved = f(ok); s = f(secs); joules = f(j); above = joules - idle * s
        print(f"  {mode:<18} solved {solved / n * 100:5.1f}%  {s / n:5.2f} s/task  {f(tok) / s:6.1f} tok/s  "
              f"{joules / n:6.1f} J/task  {joules / max(1, solved):6.1f} J per correct answer ({above / max(1, solved):6.1f} above idle)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["llamacpp", "hf"], default="llamacpp")
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int); ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    summarize(a.out) if a.summary else main(a)
