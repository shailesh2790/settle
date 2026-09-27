"""Turn a bench_decode.py results file into site/data/<name>.json for the site's Code page.

    .venv/Scripts/python tools/export_bench.py runs/bench/stage1.jsonl stage1

Keeps one row per (model, task) (the last one written, which is what a resumed run reports),
adds a one-line description of each task, and precomputes the summary the page shows.
"""
import json, math, os, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common import load_humaneval, load_mbpp


def describe(task):
    if task["kind"] == "humaneval":
        doc = task["prompt"].split('"""')[1] if '"""' in task["prompt"] else task["prompt"].split("'''")[1] if "'''" in task["prompt"] else ""
        first = " ".join(doc.strip().split("\n")[0].split())
        return f"{task['entry']}(): {first}"[:140]
    return " ".join(task["prompt"].split())[:140]


def main(src, name):
    rows = {}
    for line in open(src, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); rows[(r["model"], r["task"])] = r
    info = {t["id"]: t for t in load_humaneval() + load_mbpp()}
    models = sorted({m for m, _ in rows})
    tasks = []
    for tid in sorted({t for _, t in rows}, key=lambda s: (s.split("/")[0], int(s.split("/")[1]))):
        t = {"id": tid, "bench": "HumanEval" if tid.startswith("HumanEval") else "MBPP", "desc": describe(info[tid])}
        for m in models:
            r = rows.get((m, tid))
            if r: t[m] = {"pass": bool(r["pass"]), "tokens": r["new_tokens"], "secs": round(r["secs"], 2)}
        tasks.append(t)

    def agg(sel):
        out = {}
        for m in models:
            rs = [t[m] for t in sel if m in t]
            secs = sum(r["secs"] for r in rs); toks = sum(r["tokens"] for r in rs)
            out[m] = {"n": len(rs), "pass": round(100 * sum(r["pass"] for r in rs) / max(1, len(rs)), 1),
                      "tok_s": round(toks / max(secs, 1e-9), 1), "secs": round(secs / max(1, len(rs)), 2),
                      "tokens": round(toks / max(1, len(rs)))}
        return out

    summary = {b: agg([t for t in tasks if t["bench"] == b]) for b in ("HumanEval", "MBPP")}
    summary["All"] = agg(tasks)
    paired = [t for t in tasks if all(m in t for m in models)]
    both = sum(t["ar"]["pass"] and t["diff"]["pass"] for t in paired)
    ar_only = sum(t["ar"]["pass"] and not t["diff"]["pass"] for t in paired)
    diff_only = sum(t["diff"]["pass"] and not t["ar"]["pass"] for t in paired)
    neither = len(paired) - both - ar_only - diff_only
    z = (ar_only - diff_only) / math.sqrt(ar_only + diff_only) if ar_only + diff_only else 0.0
    out = {"name": name, "models": {"ar": "Qwen2.5-1.5B-Instruct (autoregressive)", "diff": "Fast-dLLM v2 1.5B (block diffusion)"},
           "settings": "batch 1, greedy, diffusion threshold 0.9, small block 8, max 512 new tokens, RTX 3060 Laptop 6 GB on AC; "
                       "generated code tested in a Docker sandbox; MBPP shows the model 1 test and scores all 3",
           "summary": summary, "paired": {"n": len(paired), "both": both, "ar_only": ar_only, "diff_only": diff_only,
                                           "neither": neither, "mcnemar_z": round(z, 2)},
           "tasks": tasks}
    dst = os.path.join(ROOT, "site", "data", f"{name}.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"wrote {dst}: {len(tasks)} tasks, {os.path.getsize(dst) / 1024:.0f} KB; paired {out['paired']}")
    for b, s in summary.items(): print(f"  {b:<9} " + "  ".join(f"{m} {v['pass']}% {v['tok_s']} tok/s {v['secs']} s" for m, v in s.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "stage1")
