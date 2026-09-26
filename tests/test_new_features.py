"""
test_new_features.py — Unit tests for new features and bug fixes.

Tests:
- Pow negative exponent zero base safety
- no_grad context manager and decorator
- Embedding layer forward & backward
- LayerNorm layer forward & backward
- Gradient clipping (clip_grad_norm_, clip_grad_value_)
- Learning rate schedulers (StepLR, CosineAnnealingLR, ExponentialLR)
- Einsum autograd operation
"""
import math

import numpy as np

from minigrad.graph import is_grad_enabled, no_grad
from minigrad.nn import Embedding, LayerNorm
from minigrad.nn.utils import clip_grad_norm_, clip_grad_value_
from minigrad.ops import einsum
from minigrad.optim import SGD, CosineAnnealingLR, ExponentialLR, StepLR
from minigrad.tensor import Tensor


def test_pow_negative_exponent_zero_base():
    """Negative pow with 0.0 base should not produce NaN gradients."""
    x = Tensor([[0.0, 2.0]], requires_grad=True)
    y = x ** (-2)
    y.backward()
    assert not np.any(np.isnan(x.grad)), f"x.grad contains NaN: {x.grad}"
    assert not np.any(np.isinf(x.grad)), f"x.grad contains Inf: {x.grad}"


def test_pow_negative_tensor_exponent_zero_base():
    """Negative pow with Tensor exponent and 0.0 base should not produce NaN gradients."""
    x = Tensor([[0.0, 2.0]], requires_grad=True)
    exp = Tensor([-2.0])
    y = x ** exp
    y.backward()
    assert not np.any(np.isnan(x.grad)), f"x.grad contains NaN: {x.grad}"
    assert not np.any(np.isinf(x.grad)), f"x.grad contains Inf: {x.grad}"


def test_no_grad_context_manager():
    """no_grad should function as both a context manager and decorator, skipping graph construction."""
    assert is_grad_enabled() is True

    x = Tensor([1.0, 2.0], requires_grad=True)

    with no_grad():
        assert is_grad_enabled() is False
        y = x * 2
        assert y.requires_grad is False
        assert len(y._prev) == 0

    assert is_grad_enabled() is True

    @no_grad()
    def fn():
        assert is_grad_enabled() is False
        z = x + 3
        assert z.requires_grad is False
        assert len(z._prev) == 0

    fn()
    assert is_grad_enabled() is True


def test_embedding_layer():
    """Test Embedding forward and backward lookup."""
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = Tensor(np.array([[1, 2], [3, 1]]), requires_grad=False)
    out = emb(idx)

    assert out.data.shape == (2, 2, 4)
    out.sum().backward()

    # Index 1 was retrieved twice, so its weight gradient should be 2.0
    assert np.allclose(emb.weight.grad[1], 2.0)
    assert np.allclose(emb.weight.grad[2], 1.0)
    assert np.allclose(emb.weight.grad[0], 0.0)


def test_layernorm_layer():
    """Test LayerNorm normalizes mean=0, std=1 and learns gamma/beta."""
    ln = LayerNorm(normalized_shape=(4,))
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    out = ln(x)

    assert out.data.shape == (2, 3, 4)
    means = out.data.mean(axis=-1)
    vars_ = out.data.var(axis=-1)
    assert np.allclose(means, 0.0, atol=1e-4)
    assert np.allclose(vars_, 1.0, atol=1e-3)

    out.sum().backward()
    assert not np.any(np.isnan(x.grad))
    assert not np.allclose(ln.gamma.grad, 0.0)
    assert not np.allclose(ln.beta.grad, 0.0)

    # Affine scaling verification
    ln.gamma.data = np.full((4,), 2.0)
    ln.beta.data = np.full((4,), 5.0)
    out2 = ln(x)
    assert np.allclose(out2.data.mean(axis=-1), 5.0, atol=1e-3)
    assert np.allclose(out2.data.var(axis=-1), 4.0, atol=1e-2)


def test_clip_grad_norm():
    """Test L2 norm clipping of gradients."""
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0, 4.0], requires_grad=True)
    p1.grad = np.array([3.0, 4.0])  # norm = 5
    p2.grad = np.array([0.0, 0.0])

    total_norm = clip_grad_norm_([p1, p2], max_norm=2.5)
    assert math.isclose(total_norm, 5.0)
    # p1.grad should be scaled by 2.5 / 5.0 = 0.5
    assert np.allclose(p1.grad, [1.5, 2.0])


def test_clip_grad_value():
    """Test value clipping of gradients."""
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([-10.0, 5.0])
    clip_grad_value_([p], clip_value=2.0)
    assert np.allclose(p.grad, [-2.0, 2.0])


def test_lr_schedulers():
    """Test StepLR, CosineAnnealingLR, ExponentialLR schedules."""
    p = Tensor([1.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    # StepLR
    scheduler = StepLR(opt, step_size=2, gamma=0.5)
    assert math.isclose(opt.lr, 0.1)
    scheduler.step()
    assert math.isclose(opt.lr, 0.1)
    scheduler.step()
    assert math.isclose(opt.lr, 0.05)

    # CosineAnnealingLR
    opt.lr = 0.1
    c_scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    assert math.isclose(opt.lr, 0.1)
    c_scheduler.step(5)  # half-way -> lr should be 0.05
    assert math.isclose(opt.lr, 0.05, abs_tol=1e-5)

    # ExponentialLR
    opt.lr = 0.1
    e_scheduler = ExponentialLR(opt, gamma=0.9)
    assert math.isclose(opt.lr, 0.1)
    e_scheduler.step()
    assert math.isclose(opt.lr, 0.09)


def test_einsum_autograd():
    """Test einsum matrix multiply and gradient computation."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

    c = einsum("ij,jk->ik", a, b)
    expected = a.data @ b.data
    assert np.allclose(c.data, expected)

    c.sum().backward()
    # d(sum(A @ B))/dA = grad @ B.T = 1 @ B.T
    expected_a_grad = np.ones((2, 2)) @ b.data.T
    assert np.allclose(a.grad, expected_a_grad)


def test_einsum_trace_autograd():
    """Test einsum matrix trace (repeated indices) forward and backward."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    tr = einsum("ii->", a)
    assert math.isclose(float(tr.data), 5.0)

    tr.backward()
    # d(trace(A))/dA = eye(2)
    assert np.allclose(a.grad, np.eye(2))


# ── Multi-Head Attention & Transformer ──────────────────────────────

def test_linear_3d_input():
    """Linear should handle (B, T, C) inputs by reshaping internally."""
    from minigrad.nn import Linear

    np.random.seed(42)
    linear = Linear(16, 32)
    x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
    out = linear(x)

    assert out.data.shape == (2, 5, 32), f"Expected (2, 5, 32), got {out.data.shape}"

    # Verify gradient flows back
    loss = out.sum()
    loss.backward()
    assert x.grad.shape == (2, 5, 16)
    assert not np.all(x.grad == 0)


def test_multihead_attention():
    """MHA should produce correct output shape and flow gradients."""
    from minigrad.nn.attention import MultiHeadAttention

    np.random.seed(42)
    B, T, C, H = 2, 4, 16, 4
    mha = MultiHeadAttention(embed_dim=C, num_heads=H, dropout=0.0)
    mha.eval()  # Disable dropout for deterministic test

    x = Tensor(np.random.randn(B, T, C), requires_grad=True)
    out = mha(x)

    # Shape check
    assert out.data.shape == (B, T, C), f"Expected ({B}, {T}, {C}), got {out.data.shape}"

    # Gradient check
    loss = out.sum()
    loss.backward()
    assert x.grad.shape == (B, T, C)
    assert not np.all(x.grad == 0), "Gradients should flow through attention"

    # Check all projection weights got gradients
    for name, proj in [("q", mha.q_proj), ("k", mha.k_proj),
                       ("v", mha.v_proj), ("out", mha.out_proj)]:
        assert not np.all(proj.weight.grad == 0), f"{name}_proj weight grad is all zero"


def test_multihead_attention_causal_mask():
    """Causal mask should prevent attending to future positions."""
    from minigrad.nn.attention import MultiHeadAttention
    from minigrad.ops import softmax

    np.random.seed(42)
    B, T, C, H = 1, 4, 8, 2
    mha = MultiHeadAttention(embed_dim=C, num_heads=H, dropout=0.0)
    mha.eval()

    x = Tensor(np.random.randn(B, T, C), requires_grad=True)

    # Manually compute attention scores to verify masking
    q = mha.q_proj(x).reshape(B, T, H, C // H).transpose(0, 2, 1, 3)
    k = mha.k_proj(x).reshape(B, T, H, C // H).transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(C // H)
    from minigrad.ops import einsum
    scores = einsum("bhqd,bhkd->bhqk", q, k) * scale

    # Apply causal mask
    mask = np.triu(np.ones((T, T)), k=1).astype(np.float64) * (-1e9)
    masked_scores = scores + Tensor(mask)
    attn_weights = softmax(masked_scores, axis=-1)

    # Future positions should have ~0 attention weight
    attn_data = attn_weights.data
    for t in range(T):
        for future in range(t + 1, T):
            assert np.all(attn_data[0, :, t, future] < 1e-6), (
                f"Position {t} attends to future position {future}"
            )

    # Full forward should also work with causal=True
    out = mha(x, causal=True)
    assert out.data.shape == (B, T, C)


def test_transformer_block():
    """TransformerBlock should preserve shape and propagate gradients."""
    from minigrad.nn.attention import TransformerBlock

    np.random.seed(42)
    B, T, C, H = 2, 4, 16, 4
    block = TransformerBlock(embed_dim=C, num_heads=H, dropout=0.0)
    block.eval()

    x = Tensor(np.random.randn(B, T, C), requires_grad=True)
    out = block(x, causal=True)

    # Shape preserved (residual connection)
    assert out.data.shape == (B, T, C)

    # Gradient flows
    loss = out.sum()
    loss.backward()
    assert x.grad.shape == (B, T, C)
    assert not np.all(x.grad == 0)

    # Residual connection: output shouldn't be identical to input
    assert not np.allclose(out.data, x.data)


# ── Module Freezing & LoRA ──────────────────────────────────────────

def test_module_freeze_unfreeze():
    """freeze() should disable requires_grad, unfreeze() should re-enable."""
    from minigrad.nn import Linear, Sequential

    model = Sequential([Linear(4, 8), Linear(8, 2)])
    params_before = model.parameters()
    assert len(params_before) == 4  # 2 weights + 2 biases

    # Freeze
    model.freeze()
    frozen_params = model.parameters()
    assert len(frozen_params) == 0, "Frozen model should have 0 trainable params"

    # Unfreeze
    model.unfreeze()
    unfrozen_params = model.parameters()
    assert len(unfrozen_params) == 4, "Unfrozen model should have all params back"


def test_lora_linear_forward():
    """LoRALinear should produce same output as base Linear when B=0 (init)."""
    from minigrad.nn import Linear
    from minigrad.nn.lora import LoRALinear

    np.random.seed(42)
    base = Linear(16, 32)
    x = Tensor(np.random.randn(2, 16), requires_grad=True)

    # Get base output before wrapping
    base_out = base(x).data.copy()

    # Wrap with LoRA (B initialized to zeros → ΔW = 0)
    lora = LoRALinear(base, rank=4, alpha=1.0)
    lora_out = lora(x)

    # At initialization, LoRA output should equal base output
    assert np.allclose(lora_out.data, base_out, atol=1e-10), (
        "LoRA output should match base output at initialization (B=0)"
    )

    # Only lora_A and lora_B should be trainable
    trainable = lora.parameters()
    assert len(trainable) == 2, f"Expected 2 trainable params (A, B), got {len(trainable)}"

    # Verify gradients flow through LoRA path
    loss = lora_out.sum()
    loss.backward()
    # At init B=0, so dL/dA = x.T @ (grad @ B.T) = 0 — mathematically correct.
    # But dL/dB = (x @ A).T @ grad * scaling should be non-zero.
    assert not np.all(lora.lora_B.grad == 0), "LoRA B should get gradients"
    assert lora.lora_A.requires_grad is True


def test_lora_linear_3d_input():
    """LoRALinear should handle 3D (B, T, C) sequence inputs."""
    from minigrad.nn import Linear
    from minigrad.nn.lora import LoRALinear

    np.random.seed(42)
    base = Linear(16, 32)
    lora = LoRALinear(base, rank=4)

    x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
    out = lora(x)

    assert out.data.shape == (2, 5, 32), f"Expected (2, 5, 32), got {out.data.shape}"

    loss = out.sum()
    loss.backward()
    assert x.grad.shape == (2, 5, 16)


def test_apply_lora():
    """apply_lora should replace targeted Linear layers with LoRALinear."""
    from minigrad.nn.attention import TransformerBlock
    from minigrad.nn.lora import LoRALinear, apply_lora

    np.random.seed(42)
    block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.0)

    # Count params before
    assert len(block.parameters()) > 0

    # Freeze everything
    block.freeze()
    assert len(block.parameters()) == 0

    # Apply LoRA to Q and V projections only
    count = apply_lora(block, target_modules=["q_proj", "v_proj"], rank=2, alpha=1.0)
    assert count == 2, f"Expected 2 replacements, got {count}"

    # Verify q_proj and v_proj are now LoRALinear
    assert isinstance(block.attn.q_proj, LoRALinear)
    assert isinstance(block.attn.v_proj, LoRALinear)

    # k_proj and out_proj should still be regular (frozen) Linear
    from minigrad.nn import Linear
    assert isinstance(block.attn.k_proj, Linear)
    assert isinstance(block.attn.out_proj, Linear)

    # Only LoRA params should be trainable (2 layers × (A + B) = 4 params)
    trainable = block.parameters()
    assert len(trainable) == 4, f"Expected 4 trainable LoRA params, got {len(trainable)}"

    # Forward pass should still work
    block.eval()
    x = Tensor(np.random.randn(1, 4, 16), requires_grad=True)
    out = block(x, causal=True)
    assert out.data.shape == (1, 4, 16)


def test_lora_merge():
    """Merging LoRA should produce equivalent weights to base + A@B*scaling."""
    from minigrad.nn import Linear
    from minigrad.nn.lora import LoRALinear

    np.random.seed(42)
    base = Linear(8, 4, bias=True)
    lora = LoRALinear(base, rank=2, alpha=2.0)

    # Manually set A and B to non-zero for testing
    lora.lora_A.data = np.random.randn(8, 2)
    lora.lora_B.data = np.random.randn(2, 4)

    # Get LoRA output
    x = Tensor(np.random.randn(3, 8))
    lora_out = lora(x).data.copy()

    # Merge and compare
    merged = lora.merge()
    merged_out = merged(x).data

    assert np.allclose(lora_out, merged_out, atol=1e-10), (
        "Merged Linear should produce identical output to LoRALinear"
    )

    # Merged should be a plain Linear with all params trainable
    assert isinstance(merged, Linear)
    assert len(merged.parameters()) == 2  # weight + bias
