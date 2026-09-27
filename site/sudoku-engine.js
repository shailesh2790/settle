// Sudoku engine: a line-for-line port of FactorSettle (settle_graph.py) + verified search.
// Runs in a Web Worker, a page, or Node. Plain float32 JavaScript: no GPU, no server.
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- the constraint groups
  const UNITS = [];
  for (let r = 0; r < 9; r++) UNITS.push([...Array(9)].map((_, c) => r * 9 + c));                 // rows    (type 0)
  for (let c = 0; c < 9; c++) UNITS.push([...Array(9)].map((_, r) => r * 9 + c));                 // columns (type 1)
  for (let br = 0; br < 3; br++) for (let bc = 0; bc < 3; bc++) {                                  // boxes   (type 2)
    const u = []; for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) u.push((br * 3 + r) * 9 + bc * 3 + c); UNITS.push(u);
  }
  const MEM_VAR = [], MEM_GRP = [], MEM_TYP = [];
  UNITS.forEach((u, g) => u.forEach((v) => { MEM_VAR.push(v); MEM_GRP.push(g); MEM_TYP.push(Math.floor(g / 9)); }));
  const M = MEM_VAR.length, NV = 81, NG = 27, T = 3;
  const SLOT_N = new Float32Array(NV * T);
  for (let m = 0; m < M; m++) SLOT_N[MEM_VAR[m] * T + MEM_TYP[m]]++;

  // ---------------------------------------------------------------- weights
  // Linear weights are stored transposed (in x out) so the inner loop runs over contiguous outputs.
  function loadModel(meta, buffer) {
    const all = new Float32Array(buffer), W = {};
    for (const [name, t] of Object.entries(meta.tensors)) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      W[name] = { shape: t.shape, data: all.subarray(t.offset, t.offset + n) };
    }
    const lin = (name, bias = true) => {
      const w = W[name + ".weight"], [out, inp] = w.shape, wt = new Float32Array(inp * out);
      for (let o = 0; o < out; o++) for (let i = 0; i < inp; i++) wt[i * out + o] = w.data[o * inp + i];
      return { wt, b: bias ? W[name + ".bias"].data : null, inp, out };
    };
    return {
      meta, D: meta.D, H: meta.H, C: meta.n_classes,
      embed: W["embed.weight"].data, hInit: W["h_init"].data,
      pool: lin("pool", false), rest: lin("rest"), self_: lin("self_", false), typeEmb: W["type_emb.weight"].data,
      msgOut: lin("msg_out"), upd0: lin("upd.0"), upd2: lin("upd.2"), out: lin("out"),
      lnW: W["norm.weight"].data, lnB: W["norm.bias"].data,
    };
  }

  // y[r] = x[r] @ Wt + b for `rows` rows. x: rows*inp, y: rows*out.
  function linear(L, x, rows, y) {
    const { wt, b, inp, out } = L;
    for (let r = 0; r < rows; r++) {
      const yo = r * out, xo = r * inp;
      if (b) for (let o = 0; o < out; o++) y[yo + o] = b[o]; else y.fill(0, yo, yo + out);
      for (let i = 0; i < inp; i++) {
        const v = x[xo + i]; if (v === 0) continue;
        const wo = i * out;
        for (let o = 0; o < out; o++) y[yo + o] += v * wt[wo + o];
      }
    }
    return y;
  }

  function layerNorm(P, x, rows) {
    const D = P.D;
    for (let r = 0; r < rows; r++) {
      const o = r * D; let mu = 0, v = 0;
      for (let i = 0; i < D; i++) mu += x[o + i];
      mu /= D;
      for (let i = 0; i < D; i++) { const d = x[o + i] - mu; v += d * d; }
      const inv = 1 / Math.sqrt(v / D + 1e-5);
      for (let i = 0; i < D; i++) x[o + i] = (x[o + i] - mu) * inv * P.lnW[i] + P.lnB[i];
    }
  }

  // erf-based GELU (torch.nn.GELU default). erf via Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7).
  function gelu(x) {
    const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
    const e = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
    return 0.5 * x * (1 + (x >= 0 ? e : -e));
  }

  // ---------------------------------------------------------------- one puzzle's thinking
  function createSolver(P, grid) {
    const { D, H, C } = P;
    const x = new Float32Array(NV * D), h = new Float32Array(NV * D);
    for (let v = 0; v < NV; v++) for (let i = 0; i < D; i++) {
      x[v * D + i] = P.embed[grid[v] * D + i]; h[v * D + i] = x[v * D + i] + P.hInit[i];
    }
    layerNorm(P, h, NV);
    const g = new Float32Array(NV * H), sh = new Float32Array(NV * H), S = new Float32Array(NG * H);
    const rest = new Float32Array(M * H), msg = new Float32Array(M * H), agg = new Float32Array(NV * T * H);
    const mo = new Float32Array(NV * D), cat = new Float32Array(NV * 3 * D), u1 = new Float32Array(NV * 2 * D), u2 = new Float32Array(NV * D);
    const logits = new Float32Array(NV * C);
    let t = 0;

    function step() {
      linear(P.pool, h, NV, g);                                     // what each cell contributes to its groups
      linear(P.self_, h, NV, sh);
      S.fill(0);
      for (let m = 0; m < M; m++) { const so = MEM_GRP[m] * H, go = MEM_VAR[m] * H; for (let k = 0; k < H; k++) S[so + k] += g[go + k]; }
      for (let m = 0; m < M; m++) {                                 // the rest of my group, seen from me
        const so = MEM_GRP[m] * H, go = MEM_VAR[m] * H, ro = m * H;
        for (let k = 0; k < H; k++) rest[ro + k] = S[so + k] - g[go + k];
      }
      linear(P.rest, rest, M, msg);
      agg.fill(0);
      for (let m = 0; m < M; m++) {
        const v = MEM_VAR[m], ty = MEM_TYP[m], mo_ = m * H, so = v * H, to = ty * H, ao = (v * T + ty) * H;
        for (let k = 0; k < H; k++) { const val = msg[mo_ + k] + sh[so + k] + P.typeEmb[to + k]; if (val > 0) agg[ao + k] += val; }
      }
      for (let s = 0; s < NV * T; s++) { const inv = 1 / SLOT_N[s]; for (let k = 0; k < H; k++) agg[s * H + k] *= inv; }
      linear(P.msgOut, agg, NV, mo);
      for (let v = 0; v < NV; v++) {
        cat.set(h.subarray(v * D, v * D + D), v * 3 * D);
        cat.set(x.subarray(v * D, v * D + D), v * 3 * D + D);
        cat.set(mo.subarray(v * D, v * D + D), v * 3 * D + 2 * D);
      }
      linear(P.upd0, cat, NV, u1);
      for (let i = 0; i < u1.length; i++) u1[i] = gelu(u1[i]);
      linear(P.upd2, u1, NV, u2);
      for (let i = 0; i < h.length; i++) h[i] += u2[i];
      layerNorm(P, h, NV);
      t++;
      return linear(P.out, h, NV, logits);
    }
    return { step, get t() { return t; }, logits };
  }

  // ---------------------------------------------------------------- checking
  function isSolved(q, pred) {
    for (let i = 0; i < 81; i++) if (q[i] && q[i] !== pred[i]) return false;
    for (const u of UNITS) { let seen = 0; for (const v of u) seen |= 1 << pred[v]; if (seen !== 0b1111111110) return false; }
    return true;
  }
  function givensConflict(q) {
    for (const u of UNITS) { let seen = 0; for (const v of u) { const d = q[v]; if (!d) continue; if (seen & (1 << d)) return true; seen |= 1 << d; } }
    return false;
  }
  function violations(q, pred) {
    let n = 0;
    for (const u of UNITS) { const c = new Array(10).fill(0); for (const v of u) c[pred[v]]++; for (let d = 1; d <= 9; d++) if (c[d] > 1) n += c[d] - 1; }
    for (let i = 0; i < 81; i++) if (q[i] && q[i] !== pred[i]) n++;
    return n;
  }
  function softmaxRows(logits) {
    const p = new Float32Array(81 * 9);
    for (let v = 0; v < 81; v++) {
      let mx = -Infinity; for (let c = 0; c < 9; c++) mx = Math.max(mx, logits[v * 9 + c]);
      let s = 0; for (let c = 0; c < 9; c++) { p[v * 9 + c] = Math.exp(logits[v * 9 + c] - mx); s += p[v * 9 + c]; }
      for (let c = 0; c < 9; c++) p[v * 9 + c] /= s;
    }
    return p;
  }
  function argmaxPred(q, logits) {
    const pred = new Uint8Array(81);
    for (let v = 0; v < 81; v++) {
      if (q[v]) { pred[v] = q[v]; continue; }
      let b = 0; for (let c = 1; c < 9; c++) if (logits[v * 9 + c] > logits[v * 9 + b]) b = c;
      pred[v] = b + 1;
    }
    return pred;
  }

  // Think until verified or maxSteps. onStep(t, logits) may return false to stop early.
  function think(P, q, maxSteps, onStep, checkEvery = 1) {
    const s = createSolver(P, q); let pred = null, ok = false;
    for (let t = 1; t <= maxSteps; t++) {
      const lg = s.step();
      if (t % checkEvery === 0 || t === maxSteps) { pred = argmaxPred(q, lg); ok = isSolved(q, pred); }
      if (onStep && onStep(t, lg, pred, ok) === false) break;
      if (ok) break;
    }
    return { pred, ok, steps: s.t, probs: softmaxRows(s.logits) };
  }

  // Same algorithm as settle_graph.search: branch on the least-sure blank cell, keep a beam.
  function search(P, q, opts = {}, onRound) {
    const { beam = 8, branch = 3, settleSteps = 32, maxRounds = 30 } = opts;
    let states = [{ q: Uint8Array.from(q), score: 0 }]; let totalSteps = 0;
    for (let rnd = 0; rnd <= maxRounds; rnd++) {
      const kids = [];
      for (const st of states) {
        const r = think(P, st.q, settleSteps, null, 4); totalSteps += r.steps;
        if (r.ok) return { pred: r.pred, ok: true, rounds: rnd, totalSteps };
        let cell = -1, best = 2;
        for (let v = 0; v < 81; v++) {
          if (st.q[v]) continue;
          let top = 0; for (let c = 0; c < 9; c++) top = Math.max(top, r.probs[v * 9 + c]);
          if (top < best) { best = top; cell = v; }
        }
        if (cell < 0) continue;
        const order = [...Array(9).keys()].sort((a, b) => r.probs[cell * 9 + b] - r.probs[cell * 9 + a]).slice(0, branch);
        for (const c of order) {
          const pv = r.probs[cell * 9 + c]; if (pv < 1e-4) break;
          const k = Uint8Array.from(st.q); k[cell] = c + 1;
          if (!givensConflict(k)) kids.push({ q: k, score: st.score + Math.log(pv + 1e-9), cell, value: c + 1 });
        }
      }
      if (!kids.length) break;
      kids.sort((a, b) => b.score - a.score); states = kids.slice(0, beam);
      if (onRound && onRound(rnd + 1, states, totalSteps) === false) break;
    }
    return { pred: null, ok: false, rounds: maxRounds, totalSteps };
  }

  const api = { loadModel, createSolver, think, search, isSolved, violations, givensConflict, softmaxRows, argmaxPred, UNITS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.SudokuEngine = api;
})(typeof self !== "undefined" ? self : this);
