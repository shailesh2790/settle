// Runs the Sudoku model off the main thread. Messages:
//   in  {type:"solve", q:number[81], maxSteps, search, beam, settleSteps, maxRounds, delayMs}
//   out {type:"ready", meta} | {type:"step", ...} | {type:"round", ...} | {type:"done", ...} | {type:"error", message}
importScripts("sudoku-engine.js");
const E = self.SudokuEngine;
let P = null;

const ready = (async () => {
  try {
    const [meta, bin] = await Promise.all([
      fetch("models/sudoku_model.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`model manifest: HTTP ${r.status}`); return r.json(); }),
      fetch("models/sudoku_model.bin", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`model weights: HTTP ${r.status}`); return r.arrayBuffer(); }),
    ]);
    P = E.loadModel(meta, bin);
    postMessage({ type: "ready", meta: Object.assign({}, meta, { tensors: undefined }) });
  } catch (e) {
    postMessage({ type: "error", message: "Could not load the model. " + e.message });
  }
})();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.type !== "solve") return;
  await ready; if (!P) return;
  const q = Uint8Array.from(m.q), t0 = performance.now();
  if (E.givensConflict(q)) { postMessage({ type: "done", ok: false, reason: "conflict", ms: 0 }); return; }

  // Phase 1: plain thinking, streamed step by step.
  const s = E.createSolver(P, q); let prev = null, pred = null, ok = false;
  for (let t = 1; t <= m.maxSteps; t++) {
    const ts = performance.now();
    const lg = s.step();
    pred = E.argmaxPred(q, lg); ok = E.isSolved(q, pred);
    let changed = 0; if (prev) for (let i = 0; i < 81; i++) if (pred[i] !== prev[i]) changed++;
    prev = pred;
    postMessage({ type: "step", t, probs: E.softmaxRows(lg), pred, ok, changed, stepMs: performance.now() - ts,
                  violations: ok ? 0 : E.violations(q, pred) });
    if (ok) break;
    if (m.delayMs) await sleep(m.delayMs);
  }
  if (ok || !m.search) {
    postMessage({ type: "done", ok, pred, steps: s.t, rounds: 0, ms: performance.now() - t0, reason: ok ? "verified" : "cap" });
    return;
  }

  // Phase 2: verified search. Each round reports the best surviving branch.
  const r = E.search(P, q, { beam: m.beam, settleSteps: m.settleSteps, maxRounds: m.maxRounds }, (rnd, states, total) => {
    postMessage({ type: "round", round: rnd, beam: states.length, best: Array.from(states[0].q), totalSteps: total, ms: performance.now() - t0 });
  });
  postMessage({ type: "done", ok: r.ok, pred: r.pred || pred, steps: s.t, rounds: r.rounds, searchSteps: r.totalSteps,
                ms: performance.now() - t0, reason: r.ok ? "search" : "gave-up" });
};
