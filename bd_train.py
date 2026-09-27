"""Stage 3 — convert a small autoregressive coder into a block-diffusion model with LoRA.

    python bd_train.py --preset laptop                 # RTX 3060 6 GB: smoke-scale conversion, ~overnight
    python bd_train.py --preset a100                   # Colab A100: real conversion run

Recipe (Fast-dLLM-v2 / BD3-LM family, simplified):
  * two-copy layout [noisy | clean] with the block mask from bd_masks.train_mask()
  * token shift: masked token j is predicted from row j-1 (clean copy for block-first tokens)
  * per-block masking ratio t ~ U(0.05, 1) over RESPONSE tokens only; prompt stays clean
  * optional AR loss on the clean copy keeps next-token ability intact
  * mask token = an existing unused special token, so the vocabulary never has to be resized
Untested on GPU in the authoring sandbox: the mask/index logic is unit-tested (test_logic.py);
run  --max-steps 20  first and check that the loss falls."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from bd_masks import train_mask, logit_indices, sample_block_noise, additive

PRESETS = {
    "laptop": dict(base="Qwen/Qwen2.5-Coder-0.5B-Instruct", seq=512, block=32, batch=2, accum=8, steps=3000,
                   lr=2e-4, rank=32, ckpt=True),
    "a100": dict(base="Qwen/Qwen2.5-Coder-1.5B-Instruct", seq=2048, block=32, batch=8, accum=4, steps=20000,
                 lr=1e-4, rank=64, ckpt=True),
}
MASK_CANDIDATES = ["<|fim_pad|>", "<|fim_middle|>", "<|reserved_special_token_0|>"]


def pick_mask_id(tok):
    for t in MASK_CANDIDATES:
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            return i, t
    raise SystemExit("No unused special token found for [MASK]; add one and resize embeddings.")


def examples(args, tok):
    from datasets import load_dataset
    ds = load_dataset(args.data, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    for r in ds:
        p, a = r.get(args.prompt_field), r.get(args.response_field)
        if not p or not a:
            continue
        msgs = [{"role": "system", "content": "You are a careful Python programmer."}, {"role": "user", "content": p}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        ri = tok(a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(pi) >= args.seq - args.block:
            continue
        ids = (pi + ri)[: args.seq]
        resp = np.zeros(args.seq, dtype=bool); resp[len(pi): len(ids)] = True
        ids = ids + [tok.pad_token_id or tok.eos_token_id] * (args.seq - len(ids))
        yield np.array(ids, dtype=np.int64), resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="laptop")
    ap.add_argument("--data", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    ap.add_argument("--prompt-field", default="problem"); ap.add_argument("--response-field", default="solution")
    ap.add_argument("--ar-weight", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--out", default="bd_adapter")
    cli = ap.parse_args()
    args = argparse.Namespace(**{**PRESETS[cli.preset], **vars(cli)})
    steps = args.max_steps or args.steps
    L, B, dev = args.seq, args.block, "cuda"

    tok = AutoTokenizer.from_pretrained(args.base)
    mask_id, mask_tok = pick_mask_id(tok)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation=args.attn).to(dev)
    if args.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    core = model.get_base_model()                       # Qwen2ForCausalLM with LoRA layers injected
    body, head = core.model, core.lm_head

    mask4d = torch.from_numpy(additive(train_mask(L, B))).to(dev, torch.bfloat16)[None, None]
    pos = torch.cat([torch.arange(L), torch.arange(L)]).to(dev)[None]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    rng = np.random.default_rng(0); it = examples(args, tok)
    t0, seen, step = time.time(), 0, 0
    model.train()
    while step < steps:
        for _ in range(args.accum):
            batch = [next(it) for _ in range(args.batch)]
            clean = torch.tensor(np.stack([b[0] for b in batch]), device=dev)
            noisy_np = np.stack([sample_block_noise(b[1], B, rng) for b in batch])
            noisy = clean.clone(); noisy[torch.from_numpy(noisy_np).to(dev)] = mask_id
            h = body(input_ids=torch.cat([noisy, clean], 1), attention_mask=mask4d.expand(len(batch), -1, -1, -1),
                     position_ids=pos.expand(len(batch), -1)).last_hidden_state          # (b, 2L, d)
            # diffusion loss: masked tokens, token-shifted rows
            bi, rows, tgt = [], [], []
            for k in range(len(batch)):
                js = np.flatnonzero(noisy_np[k])
                bi += [k] * len(js); rows += logit_indices(js, L, B).tolist(); tgt += batch[k][0][js].tolist()
            logits = head(h[torch.tensor(bi, device=dev), torch.tensor(rows, device=dev)]).float()
            loss_d = F.cross_entropy(logits, torch.tensor(tgt, device=dev))
            loss = loss_d
            # AR loss on the clean copy (row L+j-1 predicts token j), subsampled to bound memory
            if args.ar_weight > 0:
                bi, rows, tgt = [], [], []
                for k in range(len(batch)):
                    js = np.flatnonzero(batch[k][1]); js = js[js > 0]
                    if len(js) > 256: js = rng.choice(js, 256, replace=False)
                    bi += [k] * len(js); rows += (L + js - 1).tolist(); tgt += batch[k][0][js].tolist()
                loss_a = F.cross_entropy(head(h[torch.tensor(bi, device=dev), torch.tensor(rows, device=dev)]).float(),
                                         torch.tensor(tgt, device=dev))
                loss = loss + args.ar_weight * loss_a
            (loss / args.accum).backward()
            seen += len(batch) * L
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups: g["lr"] = args.lr * sched(step)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 10 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} diff {loss_d.item():.3f} ar {loss_a.item() if args.ar_weight>0 else 0:.3f} "
                  f"{seen/el:,.0f} tok/s  {seen/1e6:.1f}M tok  VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
        if step % 500 == 0 or step == steps:
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
            json.dump({"base": args.base, "block": B, "mask_id": int(mask_id), "mask_token": mask_tok, "seq": L,
                       "steps": step, "tokens_seen": seen}, open(os.path.join(args.out, "bd_config.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
