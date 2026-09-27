"""Stage 1 + 2 — does block diffusion pay off on YOUR laptop, before you train anything?

    python bench_decode.py --bench mbpp --limit 100                     # quick
    python bench_decode.py --bench both --thresholds 0.8 0.9 0.95       # full sweep
    python bench_decode.py --bench mbpp --repair-rounds 2               # stage 2: verifier loop

Compares, at batch size 1 (the laptop reality):
  AR   Qwen/Qwen2.5-1.5B-Instruct                 greedy HF generate
  DIFF Efficient-Large-Model/Fast_dLLM_v2_1.5B    block diffusion, same parent weights, ~1B tokens of conversion
Reports pass@1, latency, tokens/s and peak VRAM. MBPP shows the model ONE test; scoring uses all three,
so the repair loop can only use information a real user would have.

WARNING: this executes model-generated code. Run it inside a container or a throwaway VM."""
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import load_humaneval, load_mbpp, user_message, repair_message, extract_code, check

AR_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DIFF_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
SYSTEM = "You are a careful Python programmer."


# trust_remote_code runs the repo's modeling.py: pin the revision whose code was reviewed (2026-09-26)
DIFF_REVISION = "25093b6f63300adfd57f72145083c8a528fe4f16"


def load(model_id, remote):
    rev = DIFF_REVISION if model_id == DIFF_ID else None
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=remote, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                                 trust_remote_code=remote, revision=rev).eval()
    return tok, model


@torch.no_grad()
def generate(kind, tok, model, messages, max_new, threshold, small_block):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to(model.device)["input_ids"]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if kind == "ar":
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    else:
        out = model.generate(ids, tokenizer=tok, max_new_tokens=max_new, small_block_size=small_block,
                             threshold=threshold)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new = out[0][ids.shape[1]:]
    eos = (new == tok.eos_token_id).nonzero()
    n_new = int(eos[0]) if len(eos) else int(new.shape[0])
    return tok.decode(new, skip_special_tokens=True), n_new, dt


def load_done(path):
    """Rows already in the results file, keyed by (model, threshold, repair setting, task), so a crashed
    or interrupted run resumes where it stopped instead of starting over."""
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[(r["model"], r["threshold"], r.get("repair_cfg", 0), r["task"])] = r
    return done


def run(kind, tok, model, tasks, args, threshold, log, done):
    stats = {"pass": 0, "pass_repair": 0, "n": 0, "tokens": 0, "secs": 0.0, "resumed": 0}
    for task in tasks:
        prev = done.get((kind, threshold, args.repair_rounds, task["id"]))
        if prev:                                               # already measured in an earlier run
            stats["n"] += 1; stats["pass"] += prev["pass"]; stats["pass_repair"] += prev["pass_after_repair"]
            stats["tokens"] += prev.get("total_tokens", prev["new_tokens"]); stats["secs"] += prev.get("total_secs", prev["secs"])
            stats["resumed"] += 1
            continue
        t_tokens, t_secs = 0, 0.0
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task)}]
        resp, n_new, dt = generate(kind, tok, model, msgs, args.max_new, threshold, args.small_block)
        code = extract_code(resp, task)
        ok, err = check(task, code)
        stats["n"] += 1; stats["pass"] += ok; stats["tokens"] += n_new; stats["secs"] += dt
        t_tokens, t_secs = n_new, dt
        final_ok, repairs, vis0, vis_final = ok, 0, None, None
        # Stage 2: verifier loop driven ONLY by the visible test (MBPP), never by the hidden ones.
        # The visible result is logged for every MBPP answer so we can measure how well it predicts
        # the hidden tests, i.e. whether the check a user has is a trustworthy judge.
        if task["kind"] == "mbpp":
            vis0, vis_err = check(task, code, visible_only=True)
            vis_final, cur_code, cur_resp = vis0, code, resp
            while not vis_final and repairs < args.repair_rounds:
                msgs += [{"role": "assistant", "content": cur_resp}, {"role": "user", "content": repair_message(vis_err)}]
                cur_resp, n2, dt2 = generate(kind, tok, model, msgs, args.max_new, threshold, args.small_block)
                cur_code = extract_code(cur_resp, task); repairs += 1
                stats["tokens"] += n2; stats["secs"] += dt2; t_tokens += n2; t_secs += dt2
                vis_final, vis_err = check(task, cur_code, visible_only=True)
            if repairs:
                final_ok, _ = check(task, cur_code)
        stats["pass_repair"] += final_ok
        log.write(json.dumps({"model": kind, "threshold": threshold, "task": task["id"], "pass": ok,
                              "pass_after_repair": final_ok, "repair_rounds": repairs, "visible_pass": vis0,
                              "visible_pass_final": vis_final, "new_tokens": n_new,
                              "secs": round(dt, 3), "repair_cfg": args.repair_rounds, "total_tokens": t_tokens,
                              "total_secs": round(t_secs, 3)}) + "\n"); log.flush()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["humaneval", "mbpp", "both"], default="mbpp")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["ar", "diff"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.9])
    ap.add_argument("--small-block", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--repair-rounds", type=int, default=0)
    ap.add_argument("--out", default="bench_results.jsonl")
    args = ap.parse_args()
    if sys.platform == "win32":              # keep the laptop awake for the run only (reverts on exit)
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    tasks = []
    if args.bench in ("humaneval", "both"): tasks += load_humaneval(args.limit)
    if args.bench in ("mbpp", "both"): tasks += load_mbpp(args.limit)
    rows = []
    done = load_done(args.out)
    if done: print(f"resuming: {len(done)} task results already in {args.out}", flush=True)
    with open(args.out, "a", encoding="utf-8") as log:
        for kind in args.models:
            tok, model = load(AR_ID if kind == "ar" else DIFF_ID, remote=(kind == "diff"))
            torch.cuda.reset_peak_memory_stats()
            for th in ([None] if kind == "ar" else args.thresholds):
                s = run(kind, tok, model, tasks, args, th, log, done)
                rows.append((kind, th, s, torch.cuda.max_memory_allocated() / 1e9))
            del model; torch.cuda.empty_cache()

    print(f"\n{'model':<6} {'thresh':>6} {'pass@1':>7} {'+repair':>8} {'tok/s':>7} {'s/task':>7} {'VRAM GB':>8}")
    for kind, th, s, vram in rows:
        n = max(1, s["n"])
        print(f"{kind:<6} {('-' if th is None else th):>6} {s['pass']/n*100:>6.1f}% {s['pass_repair']/n*100:>7.1f}% "
              f"{s['tokens']/max(s['secs'],1e-9):>7.1f} {s['secs']/n:>7.2f} {vram:>8.2f}")
    if any(s["resumed"] for _, _, s, _ in rows):
        print("(some rows resumed from an earlier run; VRAM is peak for this run only)")
    print("\nDecision gate: diffusion pass@1 within 5 points of AR AND >= 2x tokens/s at batch 1 -> build on it.")
