# Settle: reasoning on a laptop

Small recurrent reasoners that solve problems by refining an answer in place, instead of writing
it one token at a time. They check their own answers, spend extra effort only when unsure, and
train and run on one laptop (RTX 3060, 6 GB). Everything here was measured on held-out problems.

**Live demo (runs in your browser):** https://settle-three-ochre.vercel.app

| Result | Where |
|---|---|
| Sudoku: a 233k-parameter model reaches 72% of hard held-out Sudoku-Extreme puzzles with verified search (48% by thinking alone) after 3 hours of laptop training; it never returns a wrong grid | [docs/SETTLE.md](docs/SETTLE.md) |
| Mazes: retraining the maze model to say "no path" makes its search reliable (calibration AUROC 1.00) and 3-7x cheaper, 58% on 41x41 mazes vs 25% | [arena/README.md](arena/README.md) |
| Code: block-diffusion decoding and test-driven repair on a 1.5B model, benchmarked on a laptop (both fall short of their gates) | this README, below |
| Full write-up | [docs/Settle_Project_Report.docx](docs/Settle_Project_Report.docx) |

**Prior work this builds on.** The maze model follows the "thinking longer" recurrent networks of
Schwarzschild, Bansal et al. (*End-to-end Algorithm Synthesis with Recurrent Networks*, 2022).
The Sudoku benchmark is Sudoku-Extreme from HRM (Sapient, 2025); the strongest small-model result on it
is TRM (Samsung, 2025). The Sudoku model's message passing echoes Recurrent Relational Networks
(Palm et al., 2018). What is new here is the laptop budget, verified search, the calibration
scorecard, and the "no path" result.

Results are from single runs, with small samples on the largest mazes (12 instances). MIT licensed.

---

# Settle-LM laptop lab

Four stages, cheapest first. Each ends in a number that decides whether the next stage is worth doing.

| Stage | What | Runs on | Time | Decides |
|---|---|---|---|---|
| 0 | `python probe.py` — measure VRAM, bf16 TFLOP/s, bandwidth; print what fits | laptop | 1 min | your real budget |
| 1 | `python bench_decode.py --bench both` — AR Qwen2.5-1.5B vs Fast-dLLM v2 1.5B (block diffusion, same parent), batch 1 | laptop | 2-4 h | does diffusion pay off at batch 1 on YOUR GPU? |
| 2 | `python bench_decode.py --bench mbpp --repair-rounds 2` — verifier loop driven only by the visible test | laptop | 2-3 h | how much does test feedback add? |
| 3 | `python bd_train.py --preset laptop --max-steps 20` then full run; `python bd_sample.py --bench mbpp` | laptop smoke, A100 real | overnight / ~8-24 h | can you do the conversion yourself, and does remask-refine beat regenerate? |

**Decision gate after Stage 1:** diffusion pass@1 within 5 points of AR **and** at least 2x tokens/s at batch 1
-> build on it. Published speedups (1.5-1.8x on A100/H100) were measured at large batch sizes;
batch 1 on a laptop is untested territory, which is exactly why Stage 1 comes first.

## Results

**Stage 0** (`probe.py`): 6.4 GB VRAM, 18.1 TFLOP/s bf16, 291 GB/s. A 1.5B model fits in bf16.

**Stage 1** (2026-09-27, `runs/bench/stage1.jsonl`, 421 tasks per model, batch 1, AC power, greedy, threshold 0.9):

| Benchmark | Model | pass@1 | tokens/s | tokens/task | s/task |
|---|---|---|---|---|---|
| HumanEval (164) | AR Qwen2.5-1.5B-Instruct | 50.6% | 14.3 | 201 | 14.05 |
| HumanEval (164) | Fast-dLLM v2 1.5B | 42.1% | 33.8 | 258 | 7.63 |
| MBPP (257) | AR | 50.2% | 14.3 | 81 | 5.64 |
| MBPP (257) | Fast-dLLM v2 | 39.3% | 25.4 | 194 | 7.63 |
| **All (421)** | **AR** | **50.4%** | **14.3** | | **8.92** |
| **All (421)** | **Fast-dLLM v2** | **40.4%** | **28.6** | | **7.63** |

**Gate: fail.** Diffusion is 10 points below AR (the gate allows 5). The gap is statistically
significant: on the same 421 tasks, 82 pass only with AR and 40 only with diffusion (McNemar
z = 3.8). Speed passes on tokens/s (2.0x) but that overstates it: diffusion writes 1.3-2.4x more
tokens per answer, so wall-clock per task is 1.8x faster on HumanEval and 35% *slower* on MBPP.
Both models are bound by Python per-step overhead (GPU 25-40% busy), not by the GPU.

**Stage 2** (2026-09-27, `runs/bench/stage2.jsonl`, MBPP 257, AR Qwen2.5-1.5B-Instruct, up to 2 repairs driven
only by the one visible test):

| | Tasks | Rate |
|---|---|---|
| pass@1 (first answer) | 129 / 257 | 50.2% |
| after repair | 136 / 257 | **52.9%** (+2.7 pts) |
| answers that failed the visible test and got repair attempts | 113 | |
| repairs that made the visible test pass | 9 / 113 | 8% |
| repairs that fixed the task (hidden tests) | 7 / 113 | 6% |
| repairs that broke a passing task | 0 | |
| time per task (generation) | 5.0 s first try | 15.1 s with repair (3x) |

**Gate: fail.** The target was +10 points at similar wall-clock; repair gives +2.7 points for 3x the time.
The 1.5B model mostly cannot use an error message to fix its own code.

**The visible test as a judge:** when it passes, the hidden tests pass 90% of the time (129/144); when it
fails, the hidden tests never pass (0/113). It is a useful but leaky check: one accepted answer in ten is
still wrong. A search on code that trusts it will inherit that 10%.

## Phase 2: learning from its own verified work (round 1)

`phase2.py`, `phase2_pipeline.ps1`, results in `runs/phase2/` (2026-09-28). Training problems: MBPP train +
validation (464, task ids 511-974); test: MBPP sanitized test (257, ids 11-510); no overlap. Qwen2.5-1.5B-Instruct
sampled 8 answers per training problem (temperature 0.8); answers passing ALL of that problem's tests were
kept (325 problems solved, 972 verified solutions) and used for a LoRA fine-tune (rank 16, 2 epochs).

| Same 257 test tasks | Before | After | Paired change |
|---|---|---|---|
| One greedy answer | 50.2% | 49.8% | +25 / -26 tasks, not significant |
| One sampled answer | 45.1% | 48.6% | +34 / -25, not significant (z = 1.2) |
| **Best of 8, chosen by the visible test** | **70.8%** | 69.3% | +20 / -24, not significant |
| Any of 8 passes (ceiling) | 74.7% | 75.1% | +18 / -17 |
| Time for 8 samples | 14.7 s | 11.0 s | answers ~18% shorter |

**What worked:** sampling 8 answers and submitting the first that passes the visible test adds **+20.6 points**
over one greedy answer (50.2% -> 70.8%) for 2.7x the time, far more than repair (+2.7 points for 3x the time).
**What did not:** one round of fine-tuning on its own verified solutions did not make the model better; it only
made answers shorter. The verified solutions came mostly from problems it could already solve: 44% of the 972
examples were from problems solved by 6-8 of 8 samples, only 15% from problems solved by 1-2 of 8 (the frontier),
because easy problems yield more distinct correct answers. So they taught it little that was new.

## Pocket runtime, week 1: a local 4-bit model, measured in joules per correct answer

`runtime_bench.py`, `runtime_week1.ps1`, results in `runs/runtime/` (2026-10-02). Same MBPP test tasks, prompts
and Docker sandbox as above. llama.cpp (build 11344, CUDA 12.4) runs Qwen2.5-1.5B-Instruct quantised to 4 bits
(Q4_K_M, 1.1 GB instead of 3.1 GB). Energy is read from the GPU's own counter (NVML); the CPU is not metered,
which flatters the Python-heavy transformers engine.

| Same 100 tasks | transformers bf16 | llama.cpp 4-bit | Change |
|---|---|---|---|
| One answer: solved | 54.0% | 59.0% | +13 / -8 tasks, not significant |
| One answer: seconds per task | 5.23 | 0.49 | 10.7x faster |
| One answer: GPU energy per correct answer | 416 J | 83 J | 5.0x less |
| Best of 8: solved | 68.0% | 79.0% | +17 / -6 tasks (z = 2.3) |
| Best of 8: seconds per task | 14.72 | 3.86 | 3.8x faster |
| Best of 8: GPU energy per correct answer | 1,726 J | 466 J | 3.7x less |

On all 257 tasks the 4-bit runtime solves 56.8% with one answer and 72.4% with best of 8 (bf16 runs: 50.2% and
70.8%). Quantising did not cost accuracy; the small gains are more likely from decoding and chat-template
differences between the engines than from the 4-bit weights. Raw generation speed (`llama-bench`): 181 tokens/s.

Best of 8 adds about 15 points but costs about 5.6x the energy per correct answer, so it should be spent only when
the cheap answer fails its check: that routing is week 2.

## Capacity plan (assumed RTX 3060 laptop specs: 6 GB, ~336 GB/s, ~20 TFLOP/s bf16 — replace with `probe.py` output)

| size | infer bf16 | infer int4 | AR ceiling bf16 | LoRA | QLoRA | full FT | LoRA Mtok/h |
|---|---|---|---|---|---|---|---|
| 0.1B | yes | yes | 1,680 t/s | yes | yes | yes | 63 |
| 0.5B | yes | yes | 336 t/s | yes | yes | no | 12.6 |
| 1.5B | yes | yes | 112 t/s | yes (tight) | yes | no | 4.2 |
| 3B | no | yes | 56 t/s | no | yes | no | 2.1 |
| 7B | no | yes | 24 t/s | no | no | no | 0.9 |

Reference point: Fast-dLLM v2 converted a 1.5B model with ~1B tokens. At ~12.6M tokens/hour a laptop
overnight run gives a 0.5B model ~10-15% of that budget: enough to see whether the loss and pass@1 move,
not enough for a finished model. An A100 does the full budget for 0.5B in roughly 8 hours.

## Files

- `bd_masks.py` — block-diffusion attention masks and token-shift indexing (NumPy, unit-tested)
- `common.py` — HumanEval/MBPP loading, prompts, code extraction, subprocess test runner, capacity planner
- `probe.py`, `bench_decode.py`, `bd_train.py`, `bd_sample.py` — the four stages
- `test_logic.py` — runs anywhere without a GPU

The Settle reasoners this lab builds on (maze, Sudoku, the website) are documented in
[docs/SETTLE.md](docs/SETTLE.md).

## Status and safety

The GPU-independent logic is unit-tested. The torch scripts compile but were written without GPU access:
run each with a tiny limit first (`--limit 5`, `--max-steps 20`). The benchmarks execute model-generated
code inside a Docker container (`settle-sandbox`: no network, read-only root, `nobody` user, 512 MB, 1 CPU,
hard timeout), so Docker Desktop must be running. `test_logic.py` checks those limits hold.
`SETTLE_SANDBOX=none` falls back to a plain subprocess, which is not a sandbox.
`bench_decode.py` pins Fast-dLLM v2 to the revision whose `modeling.py` was reviewed (`trust_remote_code`).
