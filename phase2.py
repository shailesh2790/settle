"""Phase 2: a small model learns from its own verified work (expert iteration), on a laptop.

    python phase2.py bestof  --k 8 --out runs/phase2/bestof_base.jsonl             # sampling baseline on the test set
    python phase2.py collect --k 8 --out runs/phase2/collect_r1.jsonl              # verified solutions to TRAINING problems
    python phase2.py train   --data runs/phase2/collect_r1.jsonl --out runs/phase2/adapter_r1
    python phase2.py bestof  --k 8 --greedy --adapter runs/phase2/adapter_r1 --out runs/phase2/bestof_r1.jsonl

Splits: training problems are MBPP train + validation (task ids 511-974); the test set is MBPP sanitized
test (ids 11-510). They do not overlap. Training answers are kept only if they pass ALL of that problem's
tests; on the test set a submitted answer is chosen using only the one visible test, as a user would.
Generated code runs in the Docker sandbox (common.run_program). Every mode resumes from its --out file.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import load_mbpp, user_message, extract_code, check

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a careful Python programmer."
POOL = ThreadPoolExecutor(4)                       # sandbox checks in parallel


def keep_awake():
    if sys.platform == "win32":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def load_train_tasks():
    from datasets import load_dataset
    full, out = load_dataset("mbpp", "full"), []
    for split in ("train", "validation"):
        for r in full[split]:
            setup = r["test_setup_code"].strip()
            out.append({"id": f"mbpp/{r['task_id']}", "kind": "mbpp", "prompt": r["text"],
                        "tests": list(r["test_list"]), "imports": [setup] if setup else []})
    return out


def messages(task):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task)}]


def load_model(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.eval()


@torch.no_grad()
def sample(tok, model, task, k, temperature, max_new=512):
    """k answers to one task. temperature 0 = one greedy answer."""
    text = tok.apply_chat_template(messages(task), tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to(model.device)
    kw = dict(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k) if temperature > 0 else dict(do_sample=False)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **kw)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    res = []
    for row in out[:, ids["input_ids"].shape[1]:]:
        eos = (row == tok.eos_token_id).nonzero()
        n = int(eos[0]) if len(eos) else int(row.shape[0])
        res.append((tok.decode(row[:n], skip_special_tokens=True), n))
    return res, dt


def checks(task, codes, visible_only):
    return list(POOL.map(lambda c: check(task, c, visible_only=visible_only)[0], codes))


def done_ids(path):
    if not os.path.exists(path): return set()
    return {json.loads(l)["task"] for l in open(path, encoding="utf-8") if l.strip()}


# ---------------------------------------------------------------- modes
def bestof(a):
    """Test set: sample k, submit the first answer that passes the visible test (else the first sample)."""
    tasks = load_mbpp(a.limit); done = done_ids(a.out)
    tok, model = load_model(a.adapter)
    with open(a.out, "a", encoding="utf-8") as log:
        for i, task in enumerate(tasks):
            if task["id"] in done: continue
            samples, dt = sample(tok, model, task, a.k, a.temperature)
            codes = [extract_code(t, task) for t, _ in samples]
            vis = checks(task, codes, visible_only=True)
            hid = checks(task, codes, visible_only=False)
            pick = vis.index(True) if any(vis) else 0
            row = {"task": task["id"], "k": a.k, "adapter": a.adapter, "visible_pass": sum(vis),
                   "submitted_hidden": hid[pick], "sample0_hidden": hid[0], "any_hidden": any(hid),
                   "tokens": sum(n for _, n in samples), "gen_secs": round(dt, 2)}
            if a.greedy:
                [(g, gn)], gdt = sample(tok, model, task, 1, 0)
                row.update(greedy_hidden=check(task, extract_code(g, task))[0], greedy_secs=round(gdt, 2), greedy_tokens=gn)
            log.write(json.dumps(row) + "\n"); log.flush()
            if (i + 1) % 25 == 0: print(f"{i + 1}/{len(tasks)}", flush=True)
    summarize_bestof(a.out)


def summarize_bestof(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows); f = lambda key: sum(r[key] for r in rows if key in r)
    print(f"\n{path}: {n} test tasks, k={rows[0]['k']}, adapter={rows[0]['adapter']}")
    if any("greedy_hidden" in r for r in rows):
        print(f"  greedy answer                           {f('greedy_hidden') / n * 100:5.1f}%   {sum(r.get('greedy_secs', 0) for r in rows) / n:.1f} s/task")
    print(f"  one sampled answer                      {f('sample0_hidden') / n * 100:5.1f}%")
    print(f"  best of {rows[0]['k']}, chosen by the visible test     {f('submitted_hidden') / n * 100:5.1f}%   {f('gen_secs') / n:.1f} s/task")
    print(f"  any of {rows[0]['k']} passes (oracle ceiling)         {f('any_hidden') / n * 100:5.1f}%")


def collect(a):
    """Training problems: keep sampled answers that pass ALL tests (deduplicated, up to --keep per problem)."""
    tasks = load_train_tasks()
    if a.limit: tasks = tasks[: a.limit]
    done = done_ids(a.out); tok, model = load_model(a.adapter)
    with open(a.out, "a", encoding="utf-8") as log:
        for i, task in enumerate(tasks):
            if task["id"] in done: continue
            samples, dt = sample(tok, model, task, a.k, a.temperature)
            codes = [extract_code(t, task) for t, _ in samples]
            ok = checks(task, codes, visible_only=False)
            keep, seen = [], set()
            for c, good in zip(codes, ok):
                norm = "\n".join(l.rstrip() for l in c.strip().splitlines())
                if good and norm not in seen: seen.add(norm); keep.append(c)
            log.write(json.dumps({"task": task["id"], "passed": sum(ok), "k": a.k, "solutions": keep[: a.keep],
                                  "user": user_message(task), "gen_secs": round(dt, 2)}) + "\n"); log.flush()
            if (i + 1) % 25 == 0: print(f"{i + 1}/{len(tasks)}", flush=True)
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8") if l.strip()]
    solved = sum(1 for r in rows if r["solutions"])
    print(f"\n{a.out}: {len(rows)} training problems, {solved} with at least one verified solution "
          f"({solved / len(rows) * 100:.1f}%), {sum(len(r['solutions']) for r in rows)} verified solutions")


def train(a):
    """LoRA fine-tune on verified solutions; loss only on the answer tokens."""
    from peft import LoraConfig, get_peft_model
    ex = []
    for path in a.data:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            for code in r["solutions"]:
                ex.append((r["user"], f"```python\n{code.strip()}\n```"))
    random.Random(0).shuffle(ex)
    print(f"{len(ex)} verified examples from {len(a.data)} file(s)", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(ex) // a.accum; step = 0; t0 = time.time(); model.train()
    for ep in range(a.epochs):
        for i, (user, answer) in enumerate(ex):
            prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(answer, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
            ids = torch.tensor([(p_ids + a_ids)[: a.max_len]], device="cuda")
            labels = ids.clone(); labels[0, : len(p_ids)] = -100          # learn the answer, not the prompt
            loss = model(input_ids=ids, labels=labels).loss / a.accum
            loss.backward()
            if (i + 1) % a.accum == 0:
                for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / 10)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    print(f"epoch {ep + 1} step {step}/{total} loss {loss.item() * a.accum:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
    model.save_pretrained(a.out)
    json.dump({"base": BASE, "data": a.data, "examples": len(ex), "epochs": a.epochs, "rank": a.rank, "lr": a.lr},
              open(os.path.join(a.out, "phase2_config.json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bestof", "collect", "train", "summary"])
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--adapter"); ap.add_argument("--limit", type=int); ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--keep", type=int, default=4); ap.add_argument("--out")
    ap.add_argument("--data", nargs="+"); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    if a.out and a.mode != "train": os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keep_awake()
    {"bestof": bestof, "collect": collect, "train": train, "summary": lambda a: summarize_bestof(a.out)}[a.mode](a)
