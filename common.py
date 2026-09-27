"""Shared utilities for the Settle-LM laptop lab."""
import os, re, subprocess, sys, tempfile, textwrap

# ------------------------------------------------------------------ benchmarks
def load_humaneval(limit=None):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval")["test"]
    out = [{"id": r["task_id"], "kind": "humaneval", "prompt": r["prompt"], "entry": r["entry_point"],
            "test": r["test"]} for r in ds]
    return out[:limit] if limit else out


def load_mbpp(limit=None):
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized")["test"]
    out = []
    for r in ds:
        text = r.get("prompt") or r.get("text")
        out.append({"id": f"mbpp/{r['task_id']}", "kind": "mbpp", "prompt": text,
                    "tests": list(r["test_list"]), "imports": list(r.get("test_imports") or [])})
    return out[:limit] if limit else out


def user_message(task):
    if task["kind"] == "humaneval":
        return ("Complete the following Python function. Return the complete function, including the "
                "signature, in a single ```python code block.\n\n" + task["prompt"])
    # MBPP: only the FIRST test is shown; the rest stay hidden for honest scoring
    return (f"{task['prompt']}\nYour code should pass this test:\n{task['tests'][0]}\n"
            "Return the function in a single ```python code block.")


def repair_message(error):
    return ("Running your code against the test above failed with:\n\n" + error.strip()[-1500:] +
            "\n\nFix the function. Return the complete corrected function in a single ```python code block.")


FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(response, task=None):
    blocks = FENCE.findall(response)
    code = max(blocks, key=len) if blocks else response
    code = code.strip("\n")
    if task and task["kind"] == "humaneval" and f"def {task['entry']}" not in code:
        code = task["prompt"] + code                       # model returned only the body
    return code


# ------------------------------------------------------------------ execution
def build_program(task, code, visible_only=False):
    if task["kind"] == "humaneval":
        return f"{code}\n\n{task['test']}\n\ncheck({task['entry']})\n"
    tests = task["tests"][:1] if visible_only else task["tests"]
    return "\n".join(task.get("imports", [])) + "\n" + code + "\n\n" + "\n".join(tests) + "\n"


SANDBOX = os.environ.get("SETTLE_SANDBOX", "docker")          # "docker" (default) or "none"
SANDBOX_NAME, SANDBOX_IMAGE = "settle-sandbox", "python:3.11-slim"
_sandbox_ready = False


def _ensure_sandbox():
    """One long-lived locked-down container; each program runs in it via `docker exec`.
    No network, read-only root, small /tmp, nobody user, no capabilities, 512 MB, 1 CPU, 64 processes."""
    global _sandbox_ready
    if _sandbox_ready:
        return
    state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", SANDBOX_NAME], capture_output=True, text=True)
    if state.stdout.strip() != "true":
        subprocess.run(["docker", "rm", "-f", SANDBOX_NAME], capture_output=True)
        r = subprocess.run(["docker", "run", "-d", "--name", SANDBOX_NAME, "--network", "none", "--read-only",
                            "--tmpfs", "/tmp:rw,size=64m", "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
                            "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                            "--user", "65534:65534", SANDBOX_IMAGE, "sleep", "infinity"], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("Could not start the Docker sandbox (is Docker Desktop running?). "
                             "Set SETTLE_SANDBOX=none to run generated code unsandboxed.\n" + r.stderr)
    _sandbox_ready = True


def run_program(src, timeout=8.0):
    """Run untrusted generated code with a timeout. Returns (passed, error_tail).
    Default: inside the Docker sandbox above. SETTLE_SANDBOX=none: a plain subprocess on this machine."""
    if SANDBOX == "docker":
        _ensure_sandbox()
        t = max(1, int(round(timeout)))
        try:
            p = subprocess.run(["docker", "exec", "-i", "-w", "/tmp", SANDBOX_NAME, "timeout", "-s", "KILL", str(t),
                                "python", "-I", "-"], input=src, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=t + 20)
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: exceeded {timeout}s"
        if p.returncode == 137:
            return False, f"TimeoutError: exceeded {timeout}s (or killed by the sandbox memory limit)"
        return p.returncode == 0, (p.stderr or p.stdout)[-2000:]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            p = subprocess.run([sys.executable, "-X", "utf8", path], cwd=d, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
            return p.returncode == 0, (p.stderr or p.stdout)[-2000:]
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: exceeded {timeout}s"


def check(task, code, visible_only=False, timeout=8.0):
    return run_program(build_program(task, code, visible_only), timeout)


# ------------------------------------------------------------------ capacity planner
def plan(vram_gb, bw_gbs, tflops, sizes_b=(0.1, 0.5, 1.5, 3.0, 7.0), mfu=0.35, reserve_gb=1.2):
    """What fits and how fast, from three measured numbers. Returns a list of dict rows.
    Rules of thumb (approximate):
      inference weights: bf16 2 B/param, int8 1, int4 ~0.56 (incl. scales)
      AR decode ceiling: bandwidth / weight bytes (batch 1, ignores KV reads)
      LoRA (bf16 base):  2 B/param + ~25% activations/LoRA/optimizer  + reserve
      QLoRA (4-bit base): 0.56 B/param + ~35%                        + reserve
      full fine-tune:     16 B/param (bf16 weights+grads, fp32 Adam moments + master) + reserve
      training compute:   ~6P FLOPs/token full, ~4P with LoRA (no weight grads for the base)"""
    rows = []
    for p in sizes_b:
        P = p * 1e9
        w16, w8, w4 = 2 * P / 1e9, P / 1e9, 0.56 * P / 1e9
        fits = lambda gb: gb + reserve_gb <= vram_gb * 0.95
        row = {
            "params_B": p,
            "infer_bf16": fits(w16), "infer_int4": fits(w4),
            "ar_tok_s_bf16": bw_gbs / w16, "ar_tok_s_int4": bw_gbs / w4,
            "lora": fits(w16 * 1.25), "qlora": fits(w4 * 1.35), "full_ft": fits(16 * P / 1e9),
            "lora_Mtok_per_h": tflops * 1e12 * mfu / (4 * P) * 3600 / 1e6,
            "full_Mtok_per_h": tflops * 1e12 * mfu / (6 * P) * 3600 / 1e6,
        }
        rows.append(row)
    return rows


def plan_table(rows):
    yn = lambda b: "yes" if b else "no"
    lines = [f"{'size':>6} | {'infer bf16':>10} | {'infer int4':>10} | {'AR ceil bf16':>12} | {'AR ceil int4':>12} | "
             f"{'LoRA':>4} | {'QLoRA':>5} | {'full FT':>7} | {'LoRA Mtok/h':>11}"]
    for r in rows:
        lines.append(f"{r['params_B']:>5}B | {yn(r['infer_bf16']):>10} | {yn(r['infer_int4']):>10} | "
                     f"{r['ar_tok_s_bf16']:>9.0f} t/s | {r['ar_tok_s_int4']:>9.0f} t/s | {yn(r['lora']):>4} | "
                     f"{yn(r['qlora']):>5} | {yn(r['full_ft']):>7} | {r['lora_Mtok_per_h']:>11.1f}")
    return "\n".join(lines)
