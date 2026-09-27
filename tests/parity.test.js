// node tests/parity.test.js — JS engine must match NumPy logits and halting behaviour.
const fs = require("fs"), path = require("path");
const S = require("../web/settle.js");
const WEIGHTS = process.argv[2] || "settle_weights.json", FIXTURE = process.argv[3] || "parity_fixture.json";
const p = S.loadWeights(JSON.parse(fs.readFileSync(path.join(__dirname, "..", WEIGHTS))));
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, FIXTURE)));
let fail = 0;
for (const c of cases) {
  const X = Float32Array.from(c.X), solver = S.createSolver(p, X, c.S, c.S);
  let lg; for (let i = 0; i < 10; i++) lg = solver.step();
  let maxErr = 0; for (let i = 0; i < lg.length; i++) maxErr = Math.max(maxErr, Math.abs(lg[i] - c.logit10[i]));
  const t0 = Date.now(), r = S.solve(p, X, c.S, c.S, 300);
  const same = r.pred.every((v, i) => v === c.pred[i]);
  const ok = maxErr < 1e-3 && same && r.steps === c.steps; fail += !ok;
  console.log(`${c.S}x${c.S}: max|dlogit| after 10 steps ${maxErr.toExponential(2)}  pred match ${same}  steps js ${r.steps} / numpy ${c.steps}  (${Date.now() - t0} ms)  ${ok ? "OK" : "FAIL"}`);
}
// JS maze generator + BFS sanity
const rand = S.mulberry32(1); let solved = 0;
for (let k = 0; k < 20; k++) {
  const m = S.makeMaze(8, rand), X = S.encodeInput(m.grid, m.S, m.start, m.goal);
  const gt = S.shortestPath(m.grid, m.S, m.start, m.goal), r = S.solve(p, X, m.S, m.S, 200);
  solved += r.pred.every((v, i) => !!(v && !m.grid[i]) === !!gt[i]);
}
console.log(`JS-generated 17x17 mazes solved: ${solved}/20`);
process.exit(fail || solved < 18 ? 1 : 0);
