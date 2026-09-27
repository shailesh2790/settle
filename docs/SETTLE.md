# Settle: reasoners that think by iterating

Small weight-tied recurrent networks that solve problems by refining a latent state in place,
instead of emitting tokens. The same small update runs every step. Harder problems take more
steps, not more parameters. Everything here trains on one laptop and runs in a browser.

Live site: **https://settle-three-ochre.vercel.app** (maze demo, Sudoku lab).

## 1. Maze reasoner (`settle.py`)

A weight-tied recurrent network (11,953 parameters) that solves mazes by iterating in latent
space. It updates every cell in parallel and halts when its answer is stable.

    h0      = relu(conv(x))
    a_t     = relu(conv([h_t, x]))        # recall: problem re-injected every step
    h_{t+1} = tanh(h_t + conv(a_t))       # bounded state, same weights every step
    answer  = conv(h_T)

It uses pure NumPy with hand-derived backprop (gradient-checked). It trained on one CPU core in
about 18 minutes.

Results on held-out mazes (trained on 9x9 to 15x15):

| Maze | Solved exactly | Mean thinking steps |
|---|---|---|
| 11x11 | 100% | 22.5 |
| 17x17 | 100% | 43.5 |
| 21x21 | 92%  | 83.8 |
| 31x31 | 73%  | 209.5 |
| 41x41 | 17% (2/12) | ~795 |

Forcing 400 steps at 21x21 gives the same accuracy as halting (18/20), so longer thinking no
longer hurts.

    pip install numpy
    python settle.py gradcheck            # verify the hand-written gradients
    python settle.py eval 5 8 10 15       # evaluate the shipped weights
    python settle.py train 2000           # retrain from scratch (writes settle_weights.json)
    python settle.py train 1000 resume    # continue training

### In the browser (`web/`)

`web/settle.js` is a direct JavaScript port of `encode`/`step`/`readout`. It matches NumPy to
about 1e-5 and halts on the same step. `build_web.py` bundles it with the weights into a single
self-contained file, `web/dist/settle.html`, which works offline. It also writes the site's
`site/maze.html`.

    python build_web.py                               # rebuild from settle_weights.json
    python build_web.py settle_torch_weights.json     # ship a model trained with settle_torch.py
    python tests/make_parity_fixture.py && node tests/parity.test.js   # JS vs NumPy check

### PyTorch port (`settle_torch.py`)

This is the same model with batched adaptive halting and AdamW training. It reads and writes
the same JSON weights.

    python settle_torch.py parity                     # must match settle.py (1.7e-5)
    python settle_torch.py eval 5 8 10 15 20
    python settle_torch.py train --C 64 --steps 20000 --nmin 4 --nmax 10

## 2. Constraint problems: Settle on factor graphs (`settle_graph.py`)

The same recurrence, moved from pixels to constraint groups. A problem is a set of variables and
the groups they belong to (for Sudoku: rows, columns, boxes). Each step, every group pools its
members, and each member hears back "the rest of my group". The model is told which cells share
a group but never the rule. It learns all-different reasoning from solved examples.

    rest   = sum(group) - me          # the rest of my row / column / box
    msg    = relu(R·rest + S·me + type)
    h      = norm(h + MLP([h, puzzle, msg]))
    answer = argmax(out(h))           # all 81 cells at once

Because answers can be checked, inference stops when the answer **verifies**. When thinking
stalls, **verified search** branches on the cell the model is least sure of, tries its top
values, and lets all branches settle together as one GPU batch. Any answer returned has passed
the checker.

Data: Sudoku-Extreme (the HRM/TRM benchmark), `test.csv` plus the first ~533k rows of
`train.csv` from huggingface.co/datasets/sapientinc/sudoku-extreme, in `data/`. Loading and
augmentation live in `tasks/sudoku.py`.

    python -m venv .venv && .venv/Scripts/pip install numpy torch --index-url https://download.pytorch.org/whl/cu126
    .venv/Scripts/python settle_graph.py train --minutes 60 --batch 128
    .venv/Scripts/python settle_graph.py eval --ckpt settle_sudoku.pt --n 200 --search
    .venv/Scripts/python settle_graph.py solve --ckpt settle_sudoku.pt --search "<81 chars, . for blanks>"
    .venv/Scripts/python settle_graph.py export --ckpt settle_sudoku.pt   # -> site/models/

### Results: 10-minute model

233k parameters, 10 minutes on an RTX 3060 laptop GPU, 200 random Sudoku-Extreme test puzzles
(`runs/factor_10min_eval.txt`):

| Difficulty (classical-solver backtracks) | Puzzles | Thinking only | + verified search |
|---|---|---|---|
| none (pure propagation) | 28 | 50.0% | 96.4% |
| 1-10 | 45 | 2.2% | 51.1% |
| 11-50 | 107 | 0.0% | 23.4% |
| >50 | 20 | 0.0% | 5.0% |
| **all** | 200 | **7.5%** | **38.0%** |

In 10 minutes the network learns constraint propagation but not guessing. Search supplies the
guessing, at about 1.7 s per puzzle. For reference, TRM (7M parameters, trained for days on a
datacenter GPU) reports 87% with no search. Pairwise peer edges (`--arch pairwise`) reached the
same validation accuracy per step as the factor graph but run about 30% slower.

### Results: 3-hour model (`runs/long_best.pt`, live on the site)

The same 233k-parameter model, trained for 180 minutes on the laptop GPU (best checkpoint at
step 215,606, validation 53.1%). Same 200 held-out puzzles as above (`runs/long_best_eval.txt`):

| Difficulty (classical-solver backtracks) | Puzzles | Thinking only | + verified search | 10-min model (thinking / + search) |
|---|---|---|---|---|
| none (pure propagation) | 28 | 100% | 100% | 50% / 96% |
| 1-10 | 45 | 46.7% | 73.3% | 2% / 51% |
| 11-50 | 107 | 38.3% | 64.5% | 0% / 23% |
| >50 | 20 | 30.0% | 70.0% | 0% / 5% |
| **all** | 200 | **48.0%** | **72.0%** | 7.5% / 38% |

Longer training taught the network to guess, not just deduce: it solves puzzles that need up to
50+ backtracks by thinking alone. Search is also cheaper (0.68 s per searched puzzle vs 1.7 s)
because its branches are better. Validation during training climbed steadily from 13.5% (6 min)
to 19.5% (57 min) to 53% (167 min), so it had not plateaued.

### Laptop-safe training (`train_long.ps1`)

    .\train_long.ps1 -Minutes 180 -Out runs\long.pt     # resumes if runs\long.pt exists
    New-Item runs\STOP                                   # stop cleanly, checkpoint saved

- It trains only on AC power. On battery it checkpoints and waits until you plug back in.
- It keeps the PC awake while training (the Windows setting is untouched and reverts on exit).
- After a GPU driver crash it resumes from the checkpoint with the same learning-rate schedule
  (at most 8 times), running one training process at a time.
- `--minutes` is the total budget across restarts. The best validation checkpoint is saved as
  `*_best.pt`.
- Launch long runs with `Start-Process pwsh -ArgumentList "-NoExit","-File","train_long.ps1",...`
  so they survive the terminal or editor session that started them.

## 3. Website (`site/`, Vercel)

Three static pages. Every model runs in the visitor's browser, so there are no servers or GPUs
to pay for.

    site/index.html           home
    site/maze.html            generated by build_web.py (edit web/app.html instead)
    site/sudoku.html          Sudoku lab: samples by difficulty, paste or type a puzzle,
                              confidence per cell, verified search
    site/sudoku-engine.js     FactorSettle inference + search in plain JS (matches PyTorch to ~1e-5)
    site/sudoku-worker.js     runs the engine off the main thread
    site/models/              exported model, shown in the page's model card
    site/data/                sample puzzles (50 per difficulty band, held out)

Ship a new model or page edits:

    .\deploy.ps1 -Ckpt runs\long_best.pt   # export, parity tests, build, deploy to production
    .\deploy.ps1                            # redeploy after editing site/*.html
    .\deploy.ps1 -Ckpt x.pt -Preview        # try it on a preview URL first

The deploy stops if the JavaScript engines no longer match the Python models. The results
tables on `index.html` and `sudoku.html` are plain text: update them from `settle_graph.py eval`
when a new model ships. To test locally, run `cd site; python -m http.server` and open
http://localhost:8000. The Sudoku worker does not run from `file://`.

## Laptop notes (RTX 3060 6 GB, Windows)

- Commit memory (RAM plus page file) is the binding limit, not VRAM. When it runs out you see
  `CUDA error: unknown error`, a CUDA OOM despite free VRAM, or "paging file is too small"
  (os error 1455) when loading model weights. Long uptime lets stale editor, Node and browser
  processes pile up. A restart freed about 28 GB.
- Run one GPU job at a time.
- The GPU throttles on battery, so measure speed on AC only.

## Where to take it next

1. Train longer or wider: the 3-hour Sudoku run was still improving. For the maze, widen to 64-128 channels for the
   maze and train longer on 9x9-21x21: expect 41x41+ extrapolation.
2. Data efficiency: train Sudoku on 1,000 puzzles and let the model grow its own training set
   from answers it verified (the "no huge dataset" test).
3. Add an external associative memory (modern Hopfield / kNN) as the knowledge store.
4. A discrete-diffusion output head for text: parallel refinement of a whole answer instead of
   left-to-right. The Settle-LM lab in the main README benchmarks this with existing models.
