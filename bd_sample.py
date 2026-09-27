"""Sample from a converted block-diffusion adapter, and repair by REMASKING instead of regenerating.

    python bd_sample.py --adapter bd_adapter --prompt "Write a function that returns the n-th Fibonacci number."
    python bd_sample.py --adapter bd_adapter --bench mbpp --limit 50 --refine-rounds 2

Each block of B tokens starts fully masked. Every step, all masked positions are predicted at once;
positions whose confidence >= threshold are committed (at least one per step). When the visible test
fails, the lowest-confidence fraction of generated tokens is remasked and re-denoised block by block,
conditioned on everything before it — an edit, not a rewrite.
No KV cache here (clarity over speed): use bench_decode.py with Fast-dLLM v2 for speed numbers."""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bd_masks import infer_mask, infer_logit_index, additive


class BDSampler:
    def __init__(self, adapter, device="cuda"):
        cfg = json.load(open(os.path.join(adapter, "bd_config.json")))
        self.B, self.mask_id = cfg["block"], cfg["mask_id"]
        self.tok = AutoTokenizer.from_pretrained(adapter)
        base = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, adapter).merge_and_unload().to(device).eval()
        self.body, self.head, self.dev = self.model.model, self.model.lm_head, device
        self.nfe = 0

    @torch.no_grad()
    def _denoise(self, ctx, block, threshold, conf):
        """Fill the masked positions of `block` given clean context `ctx` (1-D LongTensors)."""
        n_ctx, B = ctx.shape[0], block.shape[0]
        m = torch.from_numpy(additive(infer_mask(n_ctx, B))).to(self.dev, torch.bfloat16)[None, None]
        pos = torch.arange(n_ctx + B, device=self.dev)[None]
        while (block == self.mask_id).any():
            seq = torch.cat([ctx, block])[None]
            h = self.body(input_ids=seq, attention_mask=m, position_ids=pos).last_hidden_state[0]
            self.nfe += 1
            masked = (block == self.mask_id).nonzero().flatten()
            rows = torch.tensor([infer_logit_index(int(k), n_ctx) for k in masked], device=self.dev)
            probs = torch.softmax(self.head(h[rows]).float(), -1)
            c, ids = probs.max(-1)
            commit = c >= threshold
            if not commit.any():
                commit[c.argmax()] = True
            block[masked[commit]] = ids[commit]
            conf[masked[commit]] = c[commit]
        return block, conf

    @torch.no_grad()
    def generate(self, messages, max_new=512, threshold=0.9):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ctx = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.dev)
        n_prompt, confs = ctx.shape[0], []
        for _ in range(max_new // self.B):
            block = torch.full((self.B,), self.mask_id, device=self.dev)
            conf = torch.zeros(self.B, device=self.dev)
            block, conf = self._denoise(ctx, block, threshold, conf)
            ctx = torch.cat([ctx, block]); confs.append(conf)
            if (block == self.tok.eos_token_id).any():
                break
        gen, conf = ctx[n_prompt:], torch.cat(confs)
        return n_prompt, ctx, gen, conf

    @torch.no_grad()
    def refine(self, ctx_full, n_prompt, conf, frac=0.2, threshold=0.9):
        """Remask the lowest-confidence `frac` of generated tokens; re-denoise block by block."""
        gen = ctx_full[n_prompt:].clone()
        k = max(1, int(frac * gen.shape[0]))
        low = torch.topk(-conf, k).indices
        gen[low] = self.mask_id
        conf = conf.clone(); conf[low] = 0
        for s in range(0, gen.shape[0], self.B):
            blk = gen[s: s + self.B]
            if (blk == self.mask_id).any():
                ctx = torch.cat([ctx_full[:n_prompt], gen[:s]])
                blk, c = self._denoise(ctx, blk.clone(), threshold, conf[s: s + self.B].clone())
                gen[s: s + self.B], conf[s: s + self.B] = blk, c
        return torch.cat([ctx_full[:n_prompt], gen]), conf

    def decode(self, gen):
        ids = gen.tolist()
        if self.tok.eos_token_id in ids:
            ids = ids[: ids.index(self.tok.eos_token_id)]
        return self.tok.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="bd_adapter")
    ap.add_argument("--prompt"); ap.add_argument("--bench", choices=["mbpp"]); ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.9); ap.add_argument("--refine-rounds", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=384)
    a = ap.parse_args()
    S = BDSampler(a.adapter)
    sysmsg = {"role": "system", "content": "You are a careful Python programmer."}
    if a.prompt:
        t = time.time(); npf, full, gen, conf = S.generate([sysmsg, {"role": "user", "content": a.prompt}], a.max_new, a.threshold)
        n = gen.shape[0]
        print(S.decode(gen)); print(f"\n{n} tokens, {S.nfe} forward passes ({n/max(S.nfe,1):.2f} tokens/pass), {time.time()-t:.1f}s")
    else:
        from common import load_mbpp, user_message, extract_code, check
        tasks = load_mbpp(a.limit); p0 = p1 = 0; toks = passes = 0
        for task in tasks:
            S.nfe = 0
            npf, full, gen, conf = S.generate([sysmsg, {"role": "user", "content": user_message(task)}], a.max_new, a.threshold)
            code = extract_code(S.decode(gen), task); ok, _ = check(task, code); p0 += ok
            for _ in range(a.refine_rounds):
                if check(task, code, visible_only=True)[0]:
                    break
                full, conf = S.refine(full, npf, conf, threshold=a.threshold)
                code = extract_code(S.decode(full[npf:]), task)
            p1 += check(task, code)[0]; toks += gen.shape[0]; passes += S.nfe
        n = len(tasks)
        print(f"MBPP pass@1 {p0/n*100:.1f}%  after remask-refine {p1/n*100:.1f}%  tokens/forward-pass {toks/max(passes,1):.2f}")
