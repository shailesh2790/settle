"""Run anywhere, no GPU:  python test_logic.py"""
import numpy as np
from bd_masks import train_mask, logit_index, infer_mask, infer_logit_index, sample_block_noise, block_ids
from common import extract_code, run_program, build_program, plan

L, B = 12, 4
m = train_mask(L, B)
bid = block_ids(L, B)

# noisy queries: same-block noisy keys, earlier-block clean keys, never later clean keys, never own clean block
for i in range(L):
    for j in range(L):
        assert m[i, j] == (bid[i] == bid[j]), ("noisy->noisy", i, j)
        assert m[i, L + j] == (bid[j] < bid[i]), ("noisy->clean", i, j)
        assert not m[L + i, j], "clean must never see noisy"
        assert m[L + i, L + j] == (j <= i), "clean is causal"

# token shift: the row that predicts token j must never be able to see token j's clean value
for j in range(1, L):
    r = logit_index(j, L, B)
    if j % B == 0:
        assert r == L + j - 1                     # clean AR row
        assert not m[r, L + j] and not m[r, j]    # cannot see clean j (causal) nor the noisy copy
    else:
        assert r == j - 1                         # noisy row in the same block
        assert bid[r] == bid[j]
        assert not m[r, L + j], "noisy row must not see clean copy of its own block"

# inference mask and shift agree with training semantics
n_ctx = 7; im = infer_mask(n_ctx, B)
assert im[n_ctx:, :].all() and not im[:n_ctx, n_ctx:].any()
assert infer_logit_index(0, n_ctx) == n_ctx - 1 and infer_logit_index(3, n_ctx) == n_ctx + 2

# noise: only response tokens, never position 0, at least one masked
rng = np.random.default_rng(0)
resp = np.zeros(L, bool); resp[5:11] = True
for _ in range(200):
    nz = sample_block_noise(resp, B, rng)
    assert nz.any() and not (nz & ~resp).any() and not nz[0]

# code extraction
r = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
assert extract_code(r) == "def f(x):\n    return x + 1"
he = {"kind": "humaneval", "prompt": "def g(a):\n    \"\"\"doc\"\"\"\n", "entry": "g", "test": "def check(c):\n    assert c(2) == 4\n"}
assert extract_code("```python\n    return a * 2\n```", he).startswith("def g(a):")

# sandboxed runner: pass, fail, timeout
assert run_program("assert 1 + 1 == 2\n")[0]
uni = "a " + chr(0x279E) + " b " + chr(0xE9) + " " + chr(0x4E2D)      # non-ASCII, as in MBPP examples
assert run_program(f"s = {uni!r}\nassert len(s) == 9\nprint(s)\n")[0], \
    "generated code with non-ASCII characters must run (crashed Stage 1 once)"
ok, err = run_program("assert 1 == 2\n"); assert not ok and "AssertionError" in err
ok, err = run_program("while True: pass\n", timeout=1.0); assert not ok and "Timeout" in err
assert run_program(build_program(he, "def g(a):\n    return a * 2\n"))[0]
import common
if common.SANDBOX == "docker":                  # the sandbox must actually contain generated code
    ok, err = run_program("import socket; socket.create_connection(('1.1.1.1', 53), 2)\n"); assert not ok, "network must be blocked"
    ok, err = run_program("open('/etc/owned', 'w').write('x')\n"); assert not ok, "root filesystem must be read-only"
    ok, err = run_program("import os; assert os.getuid() == 65534\n"); assert ok, "must run as nobody: " + err
    ok, err = run_program("x = bytearray(900 * 1024 * 1024)\n"); assert not ok, "memory must be capped"
    assert run_program("open('/tmp/scratch', 'w').write('ok')\n")[0], "/tmp must stay writable"
mb = {"kind": "mbpp", "tests": ["assert h(1) == 2", "assert h(5) == 99"], "imports": []}
assert run_program(build_program(mb, "def h(x):\n    return x + 1\n", visible_only=True))[0]
assert not run_program(build_program(mb, "def h(x):\n    return x + 1\n"))[0]   # hidden test catches it

# planner sanity on a known card: 6 GB, 336 GB/s, 20 TFLOP/s
rows = {r["params_B"]: r for r in plan(6.0, 336, 20)}
assert rows[1.5]["infer_bf16"] and not rows[7.0]["infer_bf16"] and rows[7.0]["infer_int4"]
assert rows[0.1]["full_ft"] and not rows[0.5]["full_ft"]
print("all logic tests passed")
