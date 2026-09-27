// node tests/sudoku_parity.test.js: the browser engine must match PyTorch FactorSettle.
const fs = require("fs"), path = require("path");
const E = require("../site/sudoku-engine.js");
const dir = path.join(__dirname, "../site/models");
const meta = JSON.parse(fs.readFileSync(path.join(dir, "sudoku_model.json")));
const buf = fs.readFileSync(path.join(dir, "sudoku_model.bin"));
const P = E.loadModel(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "sudoku_fixture.json")));
let fail = 0;
for (const c of cases) {
  const q = Uint8Array.from(c.q), s = E.createSolver(P, q);
  let lg; const t0 = Date.now(); for (let i = 0; i < 16; i++) lg = s.step(); const ms = (Date.now() - t0) / 16;
  let err = 0; for (let i = 0; i < lg.length; i++) err = Math.max(err, Math.abs(lg[i] - c.logits16[i]));
  const r = E.think(P, q, 128, null, 1);
  const samePred = r.pred.every((v, i) => v === c.pred[i]);
  const ok = err < 2e-3 && r.ok === c.ok && r.steps === c.steps && samePred; fail += !ok;
  console.log(`max|dlogit| after 16 steps ${err.toExponential(2)}  verified js ${r.ok}/torch ${c.ok}  steps js ${r.steps}/torch ${c.steps}  same answer ${samePred}  ${ms.toFixed(1)} ms/step  ${ok ? "OK" : "FAIL"}`);
}
const bad = Uint8Array.from(cases[0].pred); [bad[0], bad[1]] = [bad[1], bad[0]];
console.log("checker:", E.isSolved(Uint8Array.from(cases[0].q), Uint8Array.from(cases[0].pred)) === cases[0].ok, "rejects swapped:", !E.isSolved(new Uint8Array(81), bad));
process.exit(fail ? 1 : 0);
