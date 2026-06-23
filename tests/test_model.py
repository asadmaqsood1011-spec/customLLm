"""Tests for the from-scratch GPT model: shapes, masking, GQA, and the KV cache.

The KV-cache equivalence test is the important one: it proves that cached,
one-token-at-a-time decoding produces the same logits as a single full forward
pass, which only holds if RoPE positions, the causal mask and the cache are all
wired correctly.
"""

import torch

from model import GPT, GPTConfig, build_rope_cache, rotate_half


def small_cfg(**kw):
    base = dict(
        vocab_size=64, block_size=32, n_layer=2, n_head=4, n_kv_head=4,
        n_embd=32, dropout=0.0,
    )
    base.update(kw)
    return GPTConfig(**base)


def make_model(**kw):
    torch.manual_seed(0)
    return GPT(small_cfg(**kw)).eval()


def test_forward_output_shape():
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (2, 16))
    logits, loss = m(idx)
    assert logits.shape == (2, 16, m.cfg.vocab_size)
    assert loss is None


def test_loss_is_scalar_with_targets():
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, m.cfg.vocab_size, (2, 16))
    _, loss = m(idx, targets=tgt)
    assert loss.ndim == 0 and loss.item() > 0


def test_weight_tying_shared_storage():
    m = make_model()
    assert m.tok_emb.weight is m.head.weight


def test_num_params_counts_tied_head_once():
    m = make_model()
    # if tying were broken, the head matrix (vocab*n_embd) would be double counted
    manual = sum(p.numel() for p in set(m.parameters()))
    assert m.num_params() == manual


def test_context_overflow_raises():
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (1, m.cfg.block_size + 1))
    try:
        m(idx)
        assert False, "expected context overflow assertion"
    except AssertionError as e:
        assert "context overflow" in str(e)


def test_rope_cache_shape():
    cos, sin = build_rope_cache(block_size=10, head_dim=8, theta=10000.0)
    assert cos.shape == (10, 8) and sin.shape == (10, 8)


def test_rotate_half_is_quarter_turn():
    # rotate_half applied twice negates the vector (a 180 degree rotation)
    x = torch.randn(2, 4)
    assert torch.allclose(rotate_half(rotate_half(x)), -x)


def test_causal_mask_future_tokens_dont_leak():
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (1, 8))
    base, _ = m(idx)
    # change the LAST token; logits at earlier positions must be unchanged
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % m.cfg.vocab_size
    other, _ = m(idx2)
    assert torch.allclose(base[:, :-1], other[:, :-1], atol=1e-5)


def test_gqa_shapes_with_shared_kv_heads():
    m = make_model(n_head=4, n_kv_head=2)  # n_rep = 2
    idx = torch.randint(0, m.cfg.vocab_size, (2, 12))
    logits, _ = m(idx)
    assert logits.shape == (2, 12, m.cfg.vocab_size)


def test_features_returns_backbone_hidden_states():
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (2, 10))
    feats = m.features(idx)
    assert feats.shape == (2, 10, m.cfg.n_embd)


def test_kv_cache_matches_full_forward():
    """Incremental cached decoding == one full forward pass, position by position."""
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (1, 12))

    full, _ = m._forward(idx)  # (1, T, V)

    caches, start = None, 0
    for t in range(idx.size(1)):
        step_logits, caches = m._forward(
            idx[:, t:t + 1], past_caches=caches, start_pos=start, use_cache=True
        )
        start += 1
        assert torch.allclose(step_logits[:, -1], full[:, t], atol=1e-5)


def test_generate_extends_sequence_deterministically():
    m = make_model()
    idx = torch.randint(0, m.cfg.vocab_size, (1, 4))
    torch.manual_seed(123)
    out_a = m.generate(idx, max_new_tokens=6, temperature=1.0, top_k=5)
    torch.manual_seed(123)
    out_b = m.generate(idx, max_new_tokens=6, temperature=1.0, top_k=5)
    assert out_a.shape == (1, 10)
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a[:, :4], idx)
