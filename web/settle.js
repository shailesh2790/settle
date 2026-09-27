// Settle inference engine — a line-for-line port of encode/step/readout/solve in settle.py.
// Runs in any browser or in Node. No dependencies, no GPU, no server.
// Tensors are Float32Array in HWC layout; weights are HWIO (3,3,Cin,Cout) exactly as saved.
(function (root) {
  "use strict";

  function loadWeights(o) {
    const p = { C: o.C };
    for (const k of ["wIn", "bIn", "w1", "b1", "w2", "b2", "wOut", "bOut"])
      p[k] = { shape: o[k].shape, data: Float32Array.from(o[k].data) };
    p.nParams = ["wIn", "bIn", "w1", "b1", "w2", "b2", "wOut", "bOut"].reduce((s, k) => s + p[k].data.length, 0);
    return p;
  }

  // 3x3 same-padding conv. x: H*W*Cin, w: 3*3*Cin*Cout, b: Cout -> out: H*W*Cout (pre-activation).
  function conv3(x, H, W, Cin, w, b, Cout, out) {
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < W; xx++) {
        const o = (y * W + xx) * Cout;
        for (let co = 0; co < Cout; co++) out[o + co] = b[co];
        for (let dy = 0; dy < 3; dy++) {
          const sy = y + dy - 1;
          if (sy < 0 || sy >= H) continue;
          for (let dx = 0; dx < 3; dx++) {
            const sx = xx + dx - 1;
            if (sx < 0 || sx >= W) continue;
            const xi = (sy * W + sx) * Cin, wt = (dy * 3 + dx) * Cin;
            for (let ci = 0; ci < Cin; ci++) {
              const v = x[xi + ci];
              if (v === 0) continue;                 // ReLU activations are sparse: skip zeros
              const wr = (wt + ci) * Cout;
              for (let co = 0; co < Cout; co++) out[o + co] += v * w[wr + co];
            }
          }
        }
      }
    }
    return out;
  }

  // X: H*W*3 (wall, start, goal). Returns a stateful solver you can step one "thought" at a time.
  function createSolver(p, X, H, W) {
    const C = p.C, N = H * W;
    const h = new Float32Array(N * C), hx = new Float32Array(N * (C + 3));
    const z = new Float32Array(N * C), u = new Float32Array(N * C), logit = new Float32Array(N);

    conv3(X, H, W, 3, p.wIn.data, p.bIn.data, C, h);
    for (let i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0;

    let t = 0, active = 0;
    function step() {
      for (let i = 0; i < N; i++) {                                    // concat [h, x]
        hx.set(h.subarray(i * C, i * C + C), i * (C + 3));
        hx[i * (C + 3) + C] = X[i * 3]; hx[i * (C + 3) + C + 1] = X[i * 3 + 1]; hx[i * (C + 3) + C + 2] = X[i * 3 + 2];
      }
      conv3(hx, H, W, C + 3, p.w1.data, p.b1.data, C, z);              // a = relu(conv([h, x]))
      let on = 0;
      for (let i = 0; i < z.length; i++) { if (z[i] < 0) z[i] = 0; else if (z[i] > 0) on++; }
      active = on / z.length;
      conv3(z, H, W, C, p.w2.data, p.b2.data, C, u);                    // h = tanh(h + conv(a))
      for (let i = 0; i < h.length; i++) h[i] = Math.tanh(h[i] + u[i]);
      t++;
      return readout();
    }
    function readout() {
      conv3(h, H, W, C, p.wOut.data, p.bOut.data, 1, logit);
      return logit;
    }
    return { step, readout, get t() { return t; }, get active() { return active; }, h, logit };
  }

  // Mirrors settle.solve: think until the thresholded answer is unchanged for `patience` steps.
  function solve(p, X, H, W, maxIter = 400, patience = 4) {
    const s = createSolver(p, X, H, W);
    let prev = null, stable = 0, pred;
    for (let t = 1; t <= maxIter; t++) {
      const lg = s.step();
      pred = new Uint8Array(lg.length);
      for (let i = 0; i < lg.length; i++) pred[i] = lg[i] > 0 ? 1 : 0;
      if (prev && pred.every((v, i) => v === prev[i])) stable++; else stable = 0;
      prev = pred;
      if (stable >= patience) break;
    }
    return { pred, steps: s.t, logit: s.logit };
  }

  // ------------------------------------------------------------ mazes (same algorithm as settle.make_maze)
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeMaze(n, rand = Math.random) {
    const S = 2 * n + 1, g = new Uint8Array(S * S).fill(1), seen = new Uint8Array(n * n);
    seen[0] = 1; g[1 * S + 1] = 0;
    const stack = [[0, 0]];
    while (stack.length) {
      const [cx, cy] = stack[stack.length - 1];
      const nb = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        .map(([dx, dy]) => [cx + dx, cy + dy, dx, dy])
        .filter(([x, y]) => x >= 0 && x < n && y >= 0 && y < n && !seen[y * n + x]);
      if (!nb.length) { stack.pop(); continue; }
      const [nx, ny, dx, dy] = nb[Math.floor(rand() * nb.length)];
      seen[ny * n + nx] = 1;
      g[(2 * cy + 1 + dy) * S + 2 * cx + 1 + dx] = 0;
      g[(2 * ny + 1) * S + 2 * nx + 1] = 0;
      stack.push([nx, ny]);
    }
    let a, b;
    for (let k = 0; k < 50; k++) {
      a = [Math.floor(rand() * n), Math.floor(rand() * n)]; b = [Math.floor(rand() * n), Math.floor(rand() * n)];
      if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) >= Math.max(2, n / 2)) break;
    }
    return { S, grid: g, start: [2 * a[1] + 1, 2 * a[0] + 1], goal: [2 * b[1] + 1, 2 * b[0] + 1] };
  }

  // BFS ground truth. Returns Uint8Array path mask, or null if unreachable.
  function shortestPath(g, S, s, t) {
    const prev = new Int32Array(S * S).fill(-2), q = [s[0] * S + s[1]];
    prev[q[0]] = -1;
    for (let i = 0; i < q.length; i++) {
      const u = q[i];
      if (u === t[0] * S + t[1]) break;
      const uy = Math.floor(u / S), ux = u % S;
      for (const [dy, dx] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const vy = uy + dy, vx = ux + dx, v = vy * S + vx;
        if (vy >= 0 && vy < S && vx >= 0 && vx < S && !g[v] && prev[v] === -2) { prev[v] = u; q.push(v); }
      }
    }
    let u = t[0] * S + t[1];
    if (prev[u] === -2) return null;
    const m = new Uint8Array(S * S);
    while (u !== -1) { m[u] = 1; u = prev[u]; }
    return m;
  }

  function encodeInput(g, S, s, t) {
    const X = new Float32Array(S * S * 3);
    for (let i = 0; i < S * S; i++) X[i * 3] = g[i];
    X[(s[0] * S + s[1]) * 3 + 1] = 1; X[(t[0] * S + t[1]) * 3 + 2] = 1;
    return X;
  }

  const api = { loadWeights, conv3, createSolver, solve, makeMaze, shortestPath, encodeInput, mulberry32 };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Settle = api;
})(typeof self !== "undefined" ? self : this);
