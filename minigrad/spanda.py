"""
minigrad/spanda.py — Innovation 5: S.P.A.N.D.A. (स्पन्द)
Spike-Propagation Asynchronous Network Dynamics & Autograd.

Neuromorphic Event-Driven Spiking Dynamics & Surrogate-Gradient SNNs:
1. Temporal Leaky Integrate-and-Fire (LIF) dynamics with exponential decay and refractory reset.
2. Surrogate-Gradient Autograd overcoming the non-differentiable Heaviside step barrier
   (Fast Sigmoid, ArcTan, Gaussian, Boxcar).
3. Temporal Rate/Direct Encoders and Rate/Membrane Decoders.
4. Neuromorphic Energy and Synaptic Operations (SynOps) Telemetry.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from minigrad import ops
from minigrad.nn.linear import Linear
from minigrad.nn.module import Module
from minigrad.tensor import Tensor

# ── Surrogate Gradient Functions ─────────────────────────────────────

class SurrogateType:
    FAST_SIGMOID = "fast_sigmoid"
    ATAN = "atan"
    GAUSSIAN = "gaussian"
    BOXCAR = "boxcar"


def surrogate_spike(
    v: Tensor,
    v_th: float = 1.0,
    surrogate: str = SurrogateType.FAST_SIGMOID,
    alpha: float = 2.0,
) -> Tensor:
    """
    Differentiable Spiking Operator with Surrogate Gradient Backpropagation.

    Forward pass:
        Exact non-linear Heaviside binary pulse: S = (V >= V_th) in {0.0, 1.0}

    Backward pass:
        Smooth surrogate gradient dS/dV = sigma'(V - V_th) overcoming the
        non-differentiable Dirac delta barrier.
    """
    out_data = (v.data >= v_th).astype(np.float64)
    out = Tensor(
        out_data,
        requires_grad=v.requires_grad,
        _children=(v,),
        _op=f"surrogate_spike({surrogate})",
    )

    if out.requires_grad:
        out_ref = weakref.ref(out)

        def _backward() -> None:
            o = out_ref()
            if o is None:
                return
            if v.requires_grad:
                x = v.data - v_th
                if surrogate == SurrogateType.FAST_SIGMOID:
                    # Zenke & Ganguli (2018): sigma'(x) = 1 / (1 + alpha * |x|)^2
                    grad_surr = 1.0 / ((1.0 + alpha * np.abs(x)) ** 2)
                elif surrogate == SurrogateType.ATAN:
                    # ArcTan surrogate: sigma'(x) = 1 / (pi * (1 + (pi * alpha * x)^2))
                    grad_surr = 1.0 / (np.pi * (1.0 + (np.pi * alpha * x) ** 2))
                elif surrogate == SurrogateType.GAUSSIAN:
                    # Gaussian surrogate: normal distribution with scale 1/alpha
                    sigma = 1.0 / max(alpha, 1e-4)
                    grad_surr = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (x / sigma) ** 2)
                elif surrogate == SurrogateType.BOXCAR:
                    # Piecewise linear boxcar window
                    grad_surr = np.where(np.abs(x) < 0.5 * alpha, 1.0 / max(alpha, 1e-4), 0.0)
                else:
                    raise ValueError(f"Unknown surrogate function: '{surrogate}'")

                v.grad += grad_surr * o.grad

        out._backward = _backward
    return out


# ── Leaky Integrate-and-Fire (LIF) Neuron ─────────────────────────────

class LIFCell(Module):
    """
    Single-Step Leaky Integrate-and-Fire (LIF) Neuron Cell.

    Dynamics at discrete time t:
        V_leak(t) = beta * V(t-1) * (1 - S(t-1))   [Membrane leak with refractory reset]
        V(t)      = V_leak(t) + I(t)              [Current integration]
        S(t)      = SurrogateSpike(V(t), V_th)    [Binary spike emission]
    """

    def __init__(
        self,
        beta: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        surrogate: str = SurrogateType.FAST_SIGMOID,
        alpha: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.v_th = v_th
        self.v_reset = v_reset
        self.surrogate = surrogate
        self.alpha = alpha
        self.detach_reset = detach_reset

    def forward(
        self,
        current: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Processes a single time-step input current.

        Args:
            current: Input current tensor I(t)
            state: Optional tuple (V_{t-1}, S_{t-1})
        Returns:
            Tuple of (Spike_t, (V_t, Spike_t))
        """
        if state is None:
            v_prev = Tensor(np.full_like(current.data, self.v_reset), requires_grad=False)
            s_prev = Tensor(np.zeros_like(current.data), requires_grad=False)
        else:
            v_prev, s_prev = state

        # Hard reset mask: if s_prev == 1, membrane potential resets to v_reset
        if self.detach_reset:
            reset_mask = Tensor(1.0 - s_prev.data, requires_grad=False)
        else:
            reset_mask = Tensor(1.0) - s_prev

        v_leak = (v_prev * self.beta) * reset_mask
        v_t = v_leak + current

        s_t = surrogate_spike(
            v_t,
            v_th=self.v_th,
            surrogate=self.surrogate,
            alpha=self.alpha,
        )

        return s_t, (v_t, s_t)


class LIFLayer(Module):
    """
    Multi-Step Temporal LIF Layer.

    Processes an input sequence across T time-steps:
        Input shape:  (T, B, D)
        Output shape: (T, B, D)
    """

    def __init__(
        self,
        beta: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        surrogate: str = SurrogateType.FAST_SIGMOID,
        alpha: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.cell = LIFCell(
            beta=beta,
            v_th=v_th,
            v_reset=v_reset,
            surrogate=surrogate,
            alpha=alpha,
            detach_reset=detach_reset,
        )
        self.last_spikes: Optional[np.ndarray] = None
        self.last_firing_rate: float = 0.0

    def forward(self, current_seq: Tensor) -> Tensor:
        """
        current_seq: Tensor of shape (T, B, D)
        """
        num_steps = current_seq.shape[0]
        spikes: List[Tensor] = []
        state: Optional[Tuple[Tensor, Tensor]] = None

        for t in range(num_steps):
            i_t = current_seq[t]
            s_t, state = self.cell(i_t, state)
            spikes.append(s_t)

        out = ops.stack(spikes, axis=0)
        self.last_spikes = out.data.copy()
        self.last_firing_rate = float(np.mean(out.data))
        return out


# ── Spiking Layers & Sequences ────────────────────────────────────────

class SpikingLinear(Module):
    """
    Fully Connected Synaptic Layer with LIF Spiking Neurons.

    Maps input currents across time steps T:
        I(t) = W * X(t) + b
        S(t), V(t) = LIF(I(t))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        beta: float = 0.9,
        v_th: float = 1.0,
        surrogate: str = SurrogateType.FAST_SIGMOID,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features, bias=bias)
        self.cell = LIFCell(beta=beta, v_th=v_th, surrogate=surrogate, alpha=alpha)
        self.last_spikes: Optional[np.ndarray] = None
        self.last_firing_rate: float = 0.0

    def forward(self, x: Tensor, num_steps: Optional[int] = None) -> Tensor:
        """
        Supports:
          1. Temporal input x of shape (T, B, in_features)
          2. Static input x of shape (B, in_features) with num_steps specified.
        Returns:
          Spike train of shape (T, B, out_features)
        """
        if x.ndim == 2:
            # Static input: repeated across num_steps
            t_steps = num_steps if num_steps is not None else 1
            # Dense projection computed once for static input
            current_t = self.linear(x)
            spikes: List[Tensor] = []
            state: Optional[Tuple[Tensor, Tensor]] = None

            for _ in range(t_steps):
                s_t, state = self.cell(current_t, state)
                spikes.append(s_t)

            out = ops.stack(spikes, axis=0)
            self.last_spikes = out.data.copy()
            self.last_firing_rate = float(np.mean(out.data))
            return out

        elif x.ndim == 3:
            # Temporal sequence: (T, B, in_features)
            t_steps = x.shape[0]
            spikes = []
            state = None

            for t in range(t_steps):
                current_t = self.linear(x[t])
                s_t, state = self.cell(current_t, state)
                spikes.append(s_t)

            out = ops.stack(spikes, axis=0)
            self.last_spikes = out.data.copy()
            self.last_firing_rate = float(np.mean(out.data))
            return out

        else:
            raise ValueError(f"SpikingLinear expects 2D or 3D tensor, got shape {x.shape}")


class SpikingSequential(Module):
    """
    Temporal Sequential Container for multi-layer SNN architectures.
    """

    def __init__(self, *args: Union[Module, Sequence[Module]]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.layers = list(args[0])
        else:
            self.layers = [layer for layer in args if isinstance(layer, Module)]

    def forward(self, x: Tensor, num_steps: Optional[int] = None) -> Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            if i == 0 and out.ndim == 2 and isinstance(layer, SpikingLinear):
                out = layer(out, num_steps=num_steps)
            else:
                out = layer(out)
        return out


# ── Temporal Encoders & Decoders ─────────────────────────────────────

class RateEncoder:
    """
    Poisson / Bernoulli Rate Encoder:
    Converts continuous values x in [0, 1] to binary temporal spike trains (T, B, D).
    """

    def __init__(self, num_steps: int = 10, seed: Optional[int] = None) -> None:
        self.num_steps = num_steps
        self.rng = np.random.RandomState(seed)

    def encode(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        arr = x.numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float64)
        clamped = np.clip(arr, 0.0, 1.0)
        # Generate T temporal slices
        shape = (self.num_steps,) + clamped.shape
        random_matrix = self.rng.uniform(0.0, 1.0, size=shape)
        spikes_np = (random_matrix < clamped).astype(np.float64)
        return Tensor(spikes_np, requires_grad=False)


class DirectEncoder:
    """
    Direct Latency Repetition Encoder:
    Repeats a static continuous tensor across T time-steps without stochastic sampling.
    """

    def __init__(self, num_steps: int = 10) -> None:
        self.num_steps = num_steps

    def encode(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        t = x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float64))
        repeated = np.repeat(np.expand_dims(t.data, axis=0), self.num_steps, axis=0)
        return Tensor(repeated, requires_grad=t.requires_grad)


class RateDecoder:
    """
    Spike-Rate Firing Frequency Decoder:
    Averages temporal spike trains over time: y = (1 / T) * sum_{t=1}^T S(t).
    """

    @staticmethod
    def decode(spikes: Tensor) -> Tensor:
        # spikes: (T, B, D)
        return spikes.mean(axis=0)


class MembraneDecoder(Module):
    """
    Non-Spiking Readout Layer:
    Accumulates membrane potential over T time-steps without emitting spikes,
    outputting final continuous logits V(T).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, beta: float = 0.9) -> None:
        super().__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.beta = beta

    def forward(self, spikes_seq: Tensor) -> Tensor:
        """
        spikes_seq: (T, B, in_features)
        Returns: (B, out_features) final accumulated membrane potential
        """
        num_steps = spikes_seq.shape[0]
        v: Optional[Tensor] = None

        for t in range(num_steps):
            current = self.linear(spikes_seq[t])
            if v is None:
                v = current
            else:
                v = v * self.beta + current

        return v if v is not None else Tensor(0.0)


# ── Neuromorphic Telemetry & Energy Model ────────────────────────────

@dataclass
class SpandaTelemetry:
    """
    Neuromorphic Hardware Energy & Synaptic Operations (SynOps) Diagnostics.
    Compares a T-step SNN against a single-pass (T=1) static ANN baseline using
    actual per-layer measured spike counts.

    Accounting (all analytical, 45nm FP32 constants; NOT hardware measurements):
      * ANN baseline:   one dense MAC per synapse per sample (B * in * out per layer).
      * SNN layer 0:    receives *analog* input, so its synaptic work is real MACs (not ACs).
      * SNN layer k>0:  event-driven. Every *input* spike (emitted by layer k-1) triggers
                        ``out_features`` accumulates, so ACs = (input spike count) * out_features.
      * Neuron updates: T * B * out per layer (leak multiply + threshold compare). Reported as a
                        range: charged as AC (optimistic) or as MAC (conservative).
    Memory-access energy, which usually dominates, is not modeled.
    """
    num_steps: int
    mean_firing_rate: float
    mean_sparsity: float
    dense_macs: int
    spiking_acs: int
    synops_reduction_ratio: float
    estimated_ann_energy_pj: float
    estimated_snn_energy_pj: float
    energy_efficiency_gain: float
    snn_macs: int = 0
    neuron_updates: int = 0
    energy_efficiency_gain_optimistic: float = 0.0

    def summary(self) -> str:
        return "\n".join([
            "S.P.A.N.D.A. Neuromorphic Hardware Telemetry (Single-Pass ANN vs T-Step SNN):",
            f"  Time-Steps (T):              {self.num_steps}",
            f"  Average Spike Firing Rate:   {self.mean_firing_rate * 100:.2f}% (measured across layers)",
            f"  Temporal Event Sparsity:     {self.mean_sparsity * 100:.2f}% (zero-quiescent)",
            f"  Single-Pass ANN MACs (T=1):  {self.dense_macs:,}",
            f"  SNN MACs (analog layer 0):   {self.snn_macs:,}",
            f"  SNN ACs (input-spike driven):{self.spiking_acs:,} (integrated over T={self.num_steps})",
            f"  SNN neuron updates:          {self.neuron_updates:,}",
            f"  SynOps Ratio (ANN / SNN):    {self.synops_reduction_ratio:.2f}x  (<1 means the SNN does MORE ops)",
            f"  Estimated ANN Energy (T=1):  {self.estimated_ann_energy_pj:.2f} pJ (@ 4.6 pJ/MAC)",
            f"  Estimated SNN Energy (T={self.num_steps}): {self.estimated_snn_energy_pj:.2f} pJ "
            f"(conservative: updates @ MAC cost; ACs @ 0.9 pJ)",
            f"  Energy Ratio (ANN / SNN):    {self.energy_efficiency_gain:.2f}x conservative, "
            f"{self.energy_efficiency_gain_optimistic:.2f}x optimistic (updates @ AC cost)",
            "  Note: analytical model only; memory-access energy is not modeled.",
        ])


# ── Unified SPANDA Namespace ─────────────────────────────────────────

class SPANDA:
    """
    S.P.A.N.D.A. — Spike-Propagation Asynchronous Network Dynamics & Autograd.
    """
    SurrogateType = SurrogateType
    surrogate_spike = staticmethod(surrogate_spike)
    LIFCell = LIFCell
    LIFLayer = LIFLayer
    SpikingLinear = SpikingLinear
    SpikingSequential = SpikingSequential
    RateEncoder = RateEncoder
    DirectEncoder = DirectEncoder
    RateDecoder = RateDecoder
    MembraneDecoder = MembraneDecoder

    @staticmethod
    def evaluate_neuromorphic_energy(
        model: Module,
        sample_input: Tensor,
        num_steps: int = 10,
    ) -> SpandaTelemetry:
        """
        Runs an evaluation pass and profiles neuromorphic event sparsity,
        synaptic operations (SynOps), and analytical energy estimates.

        Uses actual per-layer recorded spike trains and compares T-step SNN
        accumulate operations against a single-pass (T=1) static ANN baseline.
        """
        from minigrad.graph import no_grad

        with no_grad():
            if sample_input.ndim == 2:
                out = model(sample_input, num_steps=num_steps)
            else:
                num_steps = sample_input.shape[0]
                out = model(sample_input)

        # Inspect layers
        layers = model.layers if hasattr(model, "layers") else [model]
        batch_size = sample_input.shape[0] if sample_input.ndim == 2 else sample_input.shape[1]

        static_input = sample_input.ndim == 2
        total_dense_macs = 0
        total_snn_macs = 0
        total_spiking_acs = 0
        total_neuron_updates = 0
        total_spikes_emitted = 0.0
        total_spike_sites = 0

        # Spikes feeding the next layer: None means the layer receives analog (real-valued) input.
        prev_spikes: Optional[np.ndarray] = None
        if not static_input and bool(np.isin(sample_input.data, (0.0, 1.0)).all()):
            prev_spikes = sample_input.data

        for layer in layers:
            if isinstance(layer, SpikingLinear):
                w = layer.linear.weight.data
                in_f, out_f = w.shape
                single_pass = batch_size * (in_f * out_f)
                total_dense_macs += single_pass

                if prev_spikes is None:
                    # Analog input: genuine multiply-accumulates (computed once if the input is static).
                    total_snn_macs += single_pass * (1 if static_input else num_steps)
                else:
                    # Event-driven: every INPUT spike triggers out_f accumulates.
                    total_spiking_acs += int(round(float(np.sum(prev_spikes)) * out_f))

                if layer.last_spikes is not None:
                    total_spikes_emitted += float(np.sum(layer.last_spikes))
                    total_spike_sites += int(layer.last_spikes.size)
                    total_neuron_updates += int(layer.last_spikes.size)
                    prev_spikes = layer.last_spikes
                else:
                    prev_spikes = None
            elif isinstance(layer, LIFLayer) and layer.last_spikes is not None:
                total_spikes_emitted += float(np.sum(layer.last_spikes))
                total_spike_sites += int(layer.last_spikes.size)
                prev_spikes = layer.last_spikes

        if total_spike_sites > 0:
            mean_fr = total_spikes_emitted / total_spike_sites
        elif isinstance(out, Tensor) and out.ndim == 3:
            mean_fr = float(np.mean(out.data))
        else:
            mean_fr = 0.0

        sparsity = 1.0 - mean_fr

        if total_dense_macs == 0:
            total_dense_macs = max(int(sample_input.data.size), 1)
            total_snn_macs = total_dense_macs
            total_spiking_acs = 0

        # Analytical 45nm CMOS energy model (Horowitz 2014): 32-bit FP MAC ~ 4.6 pJ, AC ~ 0.9 pJ
        e_mac, e_ac = 4.6, 0.9
        ann_pj = total_dense_macs * e_mac
        snn_core_pj = total_snn_macs * e_mac + total_spiking_acs * e_ac
        snn_conservative_pj = max(snn_core_pj + total_neuron_updates * e_mac, 1e-3)
        snn_optimistic_pj = max(snn_core_pj + total_neuron_updates * e_ac, 1e-3)
        synops_ratio = total_dense_macs / max(total_snn_macs + total_spiking_acs, 1)

        return SpandaTelemetry(
            num_steps=num_steps,
            mean_firing_rate=mean_fr,
            mean_sparsity=sparsity,
            dense_macs=total_dense_macs,
            spiking_acs=total_spiking_acs,
            synops_reduction_ratio=synops_ratio,
            estimated_ann_energy_pj=ann_pj,
            estimated_snn_energy_pj=snn_conservative_pj,
            energy_efficiency_gain=ann_pj / snn_conservative_pj,
            snn_macs=total_snn_macs,
            neuron_updates=total_neuron_updates,
            energy_efficiency_gain_optimistic=ann_pj / snn_optimistic_pj,
        )

    SpandaTelemetry = SpandaTelemetry

