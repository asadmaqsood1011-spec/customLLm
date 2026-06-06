"""Benchmark a trained checkpoint.

    python benchmark.py             # perplexity, throughput, MFU
    python benchmark.py --gpt2      # also compare bits-per-byte vs pretrained GPT-2

Three metrics:
  - Perplexity: intrinsic quality on held-out val (our tokenizer; vocab-specific).
  - Throughput + MFU: training tokens/sec and Model FLOPs Utilization on the GPU.
  - Bits-per-byte: tokenizer-independent score, so it's a *fair* comparison to
    GPT-2 on the exact same text. (GPT-2 is loaded only to score it; our model
    is still 100% from scratch.)
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from bpe import BPETokenizer
from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
LN2 = math.log(2)


@torch.no_grad()
def eval_nats(logits_fn, ids, block_size, device):
    """Sum of negative log-likelihood (nats) over `ids`, plus token count.

    Scores each token given the preceding tokens within non-overlapping
    context windows of `block_size`.
    """
    total_nats, n_tokens = 0.0, 0
    for i in range(0, len(ids) - 1, block_size):
        x = torch.tensor([ids[i:i + block_size]], dtype=torch.long, device=device)
        y = torch.tensor([ids[i + 1:i + 1 + block_size]], dtype=torch.long, device=device)
        m = min(x.shape[1], y.shape[1])
        if m == 0:
            break
        x, y = x[:, :m], y[:, :m]
        logits = logits_fn(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nats += nll.item()
        n_tokens += y.numel()
    return total_nats, n_tokens


def benchmark_speed(model, cfg, device, batch_size, block_size, ctx, iters=30):
    """Training throughput (tokens/sec) and MFU from timed fwd+bwd steps."""
    model.train()
    x = torch.randint(0, cfg.vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, block_size), device=device)

    def step():
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(5):          # warmup
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters

    tokens_per_iter = batch_size * block_size
    tps = tokens_per_iter / dt

    # nanoGPT FLOPs estimate: 6*N (params) + attention term, per token
    N = sum(p.numel() for p in model.parameters())
    L, H = cfg.n_layer, cfg.n_head
    Q, T = cfg.n_embd // cfg.n_head, cfg.block_size
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_iter = flops_per_token * block_size * batch_size
    return tps, flops_per_iter / dt, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt2", action="store_true", help="compare bits-per-byte vs pretrained GPT-2")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu_peak_flops", type=float, default=58e12,
                   help="bf16 matmul peak. Default ~RTX 4070 bf16 w/ fp32 accumulate")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" \
        else torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)

    tok = BPETokenizer().load(os.path.join(DATA_DIR, "tokenizer.json"))
    ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"device={device} dtype={dtype} | params={model.num_params()/1e6:.2f}M "
          f"| ckpt iter {ckpt['iter']} val_loss {ckpt['val_loss']:.4f}\n")

    # held-out val tokens (the exact split train.py never trained on)
    val_ids = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16).astype(np.int64).tolist()
    val_text = tok.decode(val_ids)
    n_bytes = len(val_text.encode("utf-8"))

    # --- 1. perplexity + bits-per-byte (our model) ---
    def ours_logits(x):
        with ctx:
            return model(x)[0].float()
    nats, n_tok = eval_nats(ours_logits, val_ids, cfg.block_size, device)
    ppl = math.exp(nats / n_tok)
    bpb = nats / n_bytes / LN2
    print("== Quality (held-out val) ==")
    print(f"  val tokens     : {n_tok:,}  ({n_bytes:,} bytes)")
    print(f"  perplexity     : {ppl:.2f}   (per token, our {cfg.vocab_size}-vocab BPE)")
    print(f"  bits-per-byte  : {bpb:.4f}\n")

    # --- 2. throughput + MFU ---
    tps, flops_achieved, dt = benchmark_speed(model, cfg, device, args.batch_size, cfg.block_size, ctx)
    mfu = flops_achieved / args.gpu_peak_flops
    print("== Speed (training, fwd+bwd) ==")
    print(f"  step time      : {dt*1000:.1f} ms  (batch {args.batch_size} x ctx {cfg.block_size})")
    print(f"  throughput     : {tps:,.0f} tokens/sec")
    print(f"  achieved FLOPs : {flops_achieved/1e12:.2f} TFLOP/s")
    print(f"  MFU            : {mfu*100:.1f}%  (vs {args.gpu_peak_flops/1e12:.0f} TFLOP/s peak)\n")

    # --- 3. GPT-2 comparison (bits-per-byte, fair across tokenizers) ---
    if args.gpt2:
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        except ImportError:
            print("== GPT-2 comparison skipped: pip install transformers ==")
            return
        print("== Fair comparison vs pretrained GPT-2 (bits-per-byte, lower=better) ==")
        gtok = GPT2TokenizerFast.from_pretrained("gpt2")
        gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        g_ids = gtok.encode(val_text)

        def gpt2_logits(x):
            with ctx:
                return gpt2(x).logits.float()
        g_nats, g_tok = eval_nats(gpt2_logits, g_ids, 1024, device)
        g_bpb = g_nats / n_bytes / LN2
        print(f"  GPT-2 (124M, general)   bits-per-byte: {g_bpb:.4f}")
        print(f"  ours  ({model.num_params()/1e6:.0f}M, in-domain) bits-per-byte: {bpb:.4f}")
        better = "ours WINS" if bpb < g_bpb else "GPT-2 wins"
        print(f"  -> {better} on in-domain text "
              f"({abs(g_bpb-bpb)/max(g_bpb,bpb)*100:.0f}% gap)")


if __name__ == "__main__":
    main()
