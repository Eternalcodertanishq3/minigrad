"""
tests/test_spanda.py — Unit Tests for Innovation 5: S.P.A.N.D.A. (स्पन्द)
Spike-Propagation Asynchronous Network Dynamics & Autograd.
"""
import numpy as np

from minigrad import (
    SPANDA,
    LIFCell,
    LIFLayer,
    RateDecoder,
    RateEncoder,
    SpandaTelemetry,
    SpikingLinear,
    SpikingSequential,
    Tensor,
    surrogate_spike,
)
from minigrad.nn import MSELoss
from minigrad.optim import Adam
from minigrad.spanda import SurrogateType

# ── 1. Surrogate Spike Operator & Surrogate Autograd ──────────────────

def test_surrogate_spike_forward_and_backward():
    """
    Verifies that surrogate_spike produces exact discrete binary spikes in forward,
    and smooth non-zero surrogate gradients in backward.
    """
    # Voltages around threshold 1.0
    v_data = np.array([0.5, 0.9, 1.0, 1.2, 2.0], dtype=np.float64)
    v = Tensor(v_data.copy(), requires_grad=True)

    # 1. Forward pass: discrete binary spikes {0.0, 1.0}
    s = surrogate_spike(v, v_th=1.0, surrogate=SurrogateType.FAST_SIGMOID, alpha=2.0)
    expected_s = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(s.numpy(), expected_s)

    # 2. Backward pass: Fast Sigmoid surrogate gradient
    # sigma'(x) = 1 / (1 + alpha * |x|)^2, where x = v - v_th
    loss = s.sum()
    loss.backward()

    x = v_data - 1.0
    expected_grad = 1.0 / ((1.0 + 2.0 * np.abs(x)) ** 2)
    np.testing.assert_allclose(v.grad, expected_grad, rtol=1e-5)

    # Test ArcTan and Gaussian surrogates
    for surr in [SurrogateType.ATAN, SurrogateType.GAUSSIAN, SurrogateType.BOXCAR]:
        v_test = Tensor([1.0], requires_grad=True)
        s_test = surrogate_spike(v_test, v_th=1.0, surrogate=surr, alpha=2.0)
        s_test.backward()
        assert v_test.grad[0] > 0.0  # Must have positive surrogate gradient at threshold!


# ── 2. Leaky Integrate-and-Fire (LIF) Cell Dynamics ───────────────────

def test_lif_cell_dynamics():
    """
    Verifies sub-threshold integration, threshold crossing, and refractory hard reset.
    """
    cell = LIFCell(beta=0.8, v_th=1.0, v_reset=0.0)

    # Step 1: Sub-threshold input current 0.6 -> V = 0.6, S = 0
    i1 = Tensor([0.6])
    s1, state1 = cell(i1, None)
    v1, _ = state1
    assert s1.item() == 0.0
    np.testing.assert_allclose(v1.numpy(), [0.6])

    # Step 2: Input current 0.6 with leak: V = 0.8 * 0.6 + 0.6 = 1.08 -> Spike fires! S = 1
    i2 = Tensor([0.6])
    s2, state2 = cell(i2, state1)
    v2, _ = state2
    assert s2.item() == 1.0
    np.testing.assert_allclose(v2.numpy(), [1.08])

    # Step 3: Refractory reset: because S=1, V resets to 0.0 before adding input 0.5:
    # V = 0.0 + 0.5 = 0.5, S = 0
    i3 = Tensor([0.5])
    s3, state3 = cell(i3, state2)
    v3, _ = state3
    assert s3.item() == 0.0
    np.testing.assert_allclose(v3.numpy(), [0.5])


# ── 3. Temporal Unrolling & BPTT Gradient Flow ────────────────────────

def test_temporal_unrolling_bptt():
    """
    Verifies that gradients flow back through multiple temporal time-steps (BPTT).
    """
    layer = LIFLayer(beta=0.7, v_th=1.0)
    # Temporal sequence across T=5 time-steps, B=1, D=1
    inputs = np.array([[[0.4]], [[0.4]], [[0.4]], [[0.4]], [[0.4]]], dtype=np.float64)
    current_seq = Tensor(inputs, requires_grad=True)

    spikes = layer(current_seq)
    assert spikes.shape == (5, 1, 1)

    loss = spikes.sum()
    loss.backward()

    # Gradients must be computed for all time steps
    assert current_seq.grad is not None
    assert np.all(np.isfinite(current_seq.grad))
    assert np.any(current_seq.grad != 0.0)


# ── 4. Temporal Rate Encoders & Decoders ──────────────────────────────

def test_rate_encoder_and_decoder():
    """
    Verifies Poisson/Bernoulli rate encoding and frequency decoding.
    """
    encoder = RateEncoder(num_steps=1000, seed=42)
    x = np.array([[0.2, 0.7, 0.9]])

    spikes = encoder.encode(x)
    assert spikes.shape == (1000, 1, 3)
    # Output must be binary {0.0, 1.0}
    assert np.all((spikes.numpy() == 0.0) | (spikes.numpy() == 1.0))

    # Decoded rate should approximate input continuous values
    decoded = RateDecoder.decode(spikes).numpy()
    np.testing.assert_allclose(decoded, x, atol=0.05)


# ── 5. SpikingLinear & SpikingSequential Forward Execution ────────────

def test_spiking_linear_and_sequential():
    """
    Verifies multi-layer spiking network forward pass and binary spike emission.
    """
    model = SpikingSequential([
        SpikingLinear(4, 8, beta=0.85, v_th=1.0),
        SpikingLinear(8, 3, beta=0.85, v_th=1.0),
    ])

    batch_size = 4
    x = Tensor(np.random.randn(batch_size, 4))

    # Forward across T=8 time steps
    num_steps = 8
    spikes = model(x, num_steps=num_steps)

    assert spikes.shape == (num_steps, batch_size, 3)
    # All emitted spikes must be binary floats
    assert np.all((spikes.numpy() == 0.0) | (spikes.numpy() == 1.0))


# ── 6. End-to-End Spiking Neural Network Training ─────────────────────

def test_spiking_neural_network_training():
    """
    Trains a Spiking Neural Network on non-linear XOR data across T=8 time steps
    using Surrogate Gradient BPTT and Adam.
    """
    np.random.seed(42)

    X_train = Tensor(np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]))
    Y_target = Tensor(np.array([
        [0.05],
        [0.95],
        [0.95],
        [0.05],
    ]))

    model = SpikingSequential([
        SpikingLinear(2, 12, beta=0.85, v_th=0.8),
        SpikingLinear(12, 1, beta=0.85, v_th=0.8),
    ])

    optimizer = Adam(model.parameters(), lr=0.08)
    loss_fn = MSELoss()

    initial_loss = None
    num_steps = 8

    for epoch in range(1, 51):
        optimizer.zero_grad()
        spikes = model(X_train, num_steps=num_steps)
        rate_pred = RateDecoder.decode(spikes)
        loss = loss_fn(rate_pred, Y_target)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    # Training must reduce loss significantly
    assert final_loss < initial_loss
    assert final_loss < 0.12


# ── 7. Neuromorphic Hardware Energy & SynOps Telemetry ────────────────

def test_neuromorphic_energy_telemetry():
    """
    Verifies that SpandaTelemetry computes high temporal event sparsity and
    neuromorphic energy efficiency metrics.
    """
    model = SpikingSequential([
        SpikingLinear(8, 16, beta=0.9, v_th=1.2),
        SpikingLinear(16, 4, beta=0.9, v_th=1.2),
    ])

    sample = Tensor(np.random.randn(5, 8))
    telem = SPANDA.evaluate_neuromorphic_energy(model, sample, num_steps=10)

    assert isinstance(telem, SpandaTelemetry)
    assert telem.num_steps == 10
    assert 0.0 <= telem.mean_firing_rate <= 1.0
    assert telem.mean_sparsity >= 0.50
    assert telem.dense_macs > 0
    assert telem.spiking_acs >= 0
    assert telem.energy_efficiency_gain > 1.0
    assert "S.P.A.N.D.A." in telem.summary()
