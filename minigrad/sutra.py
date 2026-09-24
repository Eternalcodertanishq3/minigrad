"""
sutra.py — S.U.T.R.A. (State-space Unified Time-continuous Runge-Kutta Adjoint)

Continuous-Depth Neural ODE Engine with O(1) Pontryagin Adjoint Autograd
in pure NumPy/Tensor.

Features:
- Solvers: Explicit Euler (1st order), classical RK4 (4th order), and
  adaptive Dormand-Prince 5(4) (Dopri5) with FSAL and error control.
- Continuous Adjoint State Method: Backpropagates through ODEs with strictly
  O(1) memory by solving the reverse augmented adjoint ODE system.
- Direct Unrolled Autograd Mode (use_adjoint=False) for numerical parity checks.
- NeuralODE Module: Seamless integration with minigrad.nn.Module, Sequential,
  and optimizers (Adam, SGD).
- Adaptive Depth Telemetry: Tracks step counts, vector field evaluations (NFE),
  and dynamic step size adaptation.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from minigrad.graph import no_grad
from minigrad.nn.module import Module
from minigrad.tensor import Tensor


# ── Telemetry Dataclass ──────────────────────────────────────────────

@dataclass
class AdaptiveStepTelemetry:
    """
    Telemetry and performance statistics for numerical ODE integration.
    """
    solver: str
    n_steps: int = 0
    n_evals: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    step_sizes: List[float] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    rtol: float = 1e-5
    atol: float = 1e-5

    def summary(self) -> str:
        mean_step = float(np.mean(self.step_sizes)) if self.step_sizes else 0.0
        min_step = float(np.min(self.step_sizes)) if self.step_sizes else 0.0
        max_step = float(np.max(self.step_sizes)) if self.step_sizes else 0.0
        return (
            f"S.U.T.R.A. Telemetry [{self.solver}]:\n"
            f"  Total Steps:      {self.n_steps} (Accepted: {self.n_accepted}, Rejected: {self.n_rejected})\n"
            f"  Function Evals:   {self.n_evals} (NFE)\n"
            f"  Step Size Range:  [{min_step:.4e}, {max_step:.4e}] (Mean: {mean_step:.4e})\n"
            f"  Tolerances:       rtol={self.rtol:.1e}, atol={self.atol:.1e}"
        )


# ── State Algebra Helpers ────────────────────────────────────────────

def _add_states(s1: List[np.ndarray], s2: List[np.ndarray]) -> List[np.ndarray]:
    return [a + b for a, b in zip(s1, s2)]


def _scale_state(s: List[np.ndarray], factor: float) -> List[np.ndarray]:
    return [a * factor for a in s]


def _axpy_state(s1: List[np.ndarray], alpha: float, s2: List[np.ndarray]) -> List[np.ndarray]:
    return [a + alpha * b for a, b in zip(s1, s2)]


def _linear_comb_states(coeffs: List[float], stages: List[List[np.ndarray]]) -> List[np.ndarray]:
    n_comp = len(stages[0])
    res: List[np.ndarray] = []
    for c_idx in range(n_comp):
        acc = np.zeros_like(stages[0][c_idx])
        for coeff, stage in zip(coeffs, stages):
            if coeff != 0.0:
                acc += coeff * stage[c_idx]
        res.append(acc)
    return res


# ── Dormand-Prince 5(4) Coefficients (Butcher Tableau) ──────────────

C_DOPRI5 = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]

A_DOPRI5 = [
    [],
    [1.0 / 5.0],
    [3.0 / 40.0, 9.0 / 40.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
]

E_DOPRI5 = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
]


def _compute_dopri5_error(
    err_list: List[np.ndarray],
    curr_list: List[np.ndarray],
    next_list: List[np.ndarray],
    rtol: float,
    atol: float,
    active_indices: Optional[Sequence[int]] = None,
) -> float:
    if active_indices is None:
        active_indices = range(len(err_list))

    total_elements = 0
    sum_sq = 0.0
    for idx in active_indices:
        err = err_list[idx]
        y_c = curr_list[idx]
        y_n = next_list[idx]
        scale = atol + rtol * np.maximum(np.abs(y_c), np.abs(y_n))
        ratio = err / np.maximum(scale, 1e-15)
        sum_sq += np.sum(ratio ** 2)
        total_elements += err.size

    return float(np.sqrt(sum_sq / max(total_elements, 1)))


# ── Vector Field Wrapper ─────────────────────────────────────────────

def _wrap_ode_func(func: Union[Module, Callable]) -> Callable[[float, Any], Any]:
    """
    Normalizes func into a standard signature f(t, y) where t is float and y is Tensor/ndarray.
    Supports Module, func(t, y), func(y, t), and autonomous func(y).
    """
    if isinstance(func, Module):
        try:
            sig = inspect.signature(func.forward)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            if len(params) == 1:
                return lambda t, y: func(y)
            elif len(params) >= 2:
                p0_name = params[0].name.lower()
                if p0_name in ("t", "time"):
                    return lambda t, y: func(t, y)
                else:
                    return lambda t, y: func(y, t)
        except Exception:
            return lambda t, y: func(y)

    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) == 1:
            return lambda t, y: func(y)
        elif len(params) >= 2:
            p0_name = params[0].name.lower()
            if p0_name in ("t", "time"):
                return lambda t, y: func(t, y)
            elif p0_name in ("y", "x", "h", "state"):
                return lambda t, y: func(y, t)
            else:
                return lambda t, y: func(t, y)
    except Exception:
        pass

    def wrapper(t: float, y: Any) -> Any:
        try:
            return func(t, y)
        except TypeError:
            try:
                return func(y, t)
            except TypeError:
                return func(y)

    return wrapper


def _eval_func_np(
    func: Callable[[float, Any], Any],
    t: float,
    y_data: np.ndarray,
) -> np.ndarray:
    """
    Evaluates func in no_grad mode, returning a detached NumPy array.
    """
    with no_grad():
        out = func(t, Tensor(y_data, requires_grad=False))
    if isinstance(out, Tensor):
        return out.data.copy()
    if isinstance(out, (list, tuple)):
        items = [x.data if isinstance(x, Tensor) else x for x in out]
        return np.asarray(items, dtype=np.float64)
    if isinstance(out, np.ndarray):
        if out.dtype == object:
            items = [x.data if isinstance(x, Tensor) else x for x in out.flat]
            return np.asarray(items, dtype=np.float64).reshape(out.shape)
        return out.copy()
    return np.asarray(out, dtype=np.float64)


# ── Numerical Solvers for State Lists ────────────────────────────────

def _integrate_interval_euler(
    deriv_fn: Callable[[float, List[np.ndarray]], List[np.ndarray]],
    y_start: List[np.ndarray],
    t_start: float,
    t_end: float,
    n_steps: int,
    telemetry: Optional[AdaptiveStepTelemetry] = None,
) -> List[np.ndarray]:
    dt = (t_end - t_start) / max(n_steps, 1)
    y = [np.copy(arr) for arr in y_start]
    t = t_start

    for _ in range(n_steps):
        dy = deriv_fn(t, y)
        y = _axpy_state(y, dt, dy)
        t += dt
        if telemetry is not None:
            telemetry.n_steps += 1
            telemetry.n_evals += 1
            telemetry.n_accepted += 1
            telemetry.step_sizes.append(abs(dt))
            telemetry.time_points.append(t)

    return y


def _integrate_interval_rk4(
    deriv_fn: Callable[[float, List[np.ndarray]], List[np.ndarray]],
    y_start: List[np.ndarray],
    t_start: float,
    t_end: float,
    n_steps: int,
    telemetry: Optional[AdaptiveStepTelemetry] = None,
) -> List[np.ndarray]:
    dt = (t_end - t_start) / max(n_steps, 1)
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0
    y = [np.copy(arr) for arr in y_start]
    t = t_start

    for _ in range(n_steps):
        k1 = deriv_fn(t, y)
        s2 = _axpy_state(y, half_dt, k1)
        k2 = deriv_fn(t + half_dt, s2)
        s3 = _axpy_state(y, half_dt, k2)
        k3 = deriv_fn(t + half_dt, s3)
        s4 = _axpy_state(y, dt, k3)
        k4 = deriv_fn(t + dt, s4)

        comb = _linear_comb_states([1.0, 2.0, 2.0, 1.0], [k1, k2, k3, k4])
        y = _axpy_state(y, sixth_dt, comb)
        t += dt

        if telemetry is not None:
            telemetry.n_steps += 1
            telemetry.n_evals += 4
            telemetry.n_accepted += 1
            telemetry.step_sizes.append(abs(dt))
            telemetry.time_points.append(t)

    return y


def _integrate_interval_dopri5(
    deriv_fn: Callable[[float, List[np.ndarray]], List[np.ndarray]],
    y_start: List[np.ndarray],
    t_start: float,
    t_end: float,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_steps: int = 2000,
    h_init: Optional[float] = None,
    active_indices: Optional[Sequence[int]] = None,
    telemetry: Optional[AdaptiveStepTelemetry] = None,
) -> List[np.ndarray]:
    direction = 1.0 if t_end >= t_start else -1.0
    span_len = abs(t_end - t_start)
    if span_len < 1e-14:
        return [np.copy(arr) for arr in y_start]

    if h_init is not None:
        h = direction * abs(h_init)
    else:
        h = direction * min(0.05 * span_len, 0.05)
        if abs(h) < 1e-6:
            h = direction * 0.1 * span_len

    h_min = 1e-12 * span_len
    h_max = span_len

    y = [np.copy(arr) for arr in y_start]
    t = t_start

    # FSAL cache
    k1 = deriv_fn(t, y)
    n_evals = 1

    steps = 0
    while steps < max_steps:
        # Check completion
        if (direction > 0 and t >= t_end - 1e-12) or (direction < 0 and t <= t_end + 1e-12):
            break

        # Clamp step to hit t_end exactly
        if (direction > 0 and t + h > t_end) or (direction < 0 and t + h < t_end):
            h = t_end - t

        # Stages 2 through 7
        k_stages: List[List[np.ndarray]] = [k1]
        for stage_i in range(1, 7):
            coeffs = A_DOPRI5[stage_i]
            stage_state = _linear_comb_states(coeffs, k_stages[:stage_i])
            s_eval = _axpy_state(y, h, stage_state)
            t_eval = t + C_DOPRI5[stage_i] * h
            k_eval = deriv_fn(t_eval, s_eval)
            k_stages.append(k_eval)
            n_evals += 1

        # 5th-order next state is stage 7 argument (FSAL property)
        y_next = _axpy_state(y, h, _linear_comb_states(A_DOPRI5[6], k_stages[:6]))

        # Error vector: E = B5 - B4
        err_vec = _scale_state(_linear_comb_states(E_DOPRI5, k_stages), h)
        err_ratio = _compute_dopri5_error(
            err_vec, y, y_next, rtol=rtol, atol=atol, active_indices=active_indices
        )

        steps += 1
        if err_ratio <= 1.0 or abs(h) <= h_min:
            # Step accepted
            t += h
            y = y_next
            # FSAL: stage 7 evaluation k_stages[6] matches deriv at (t_next, y_next)
            k1 = k_stages[6]

            if telemetry is not None:
                telemetry.n_steps += 1
                telemetry.n_accepted += 1
                telemetry.step_sizes.append(abs(h))
                telemetry.time_points.append(t)

            factor = 0.9 * (1.0 / max(err_ratio, 1e-10)) ** 0.2
            factor = float(np.clip(factor, 0.2, 5.0))
            h_next = h * factor
            if abs(h_next) > h_max:
                h_next = direction * h_max
            h = h_next
        else:
            # Step rejected
            if telemetry is not None:
                telemetry.n_rejected += 1

            factor = 0.9 * (1.0 / max(err_ratio, 1e-10)) ** 0.2
            factor = float(np.clip(factor, 0.1, 0.5))
            h = h * factor
            if abs(h) < h_min:
                h = direction * h_min

    if telemetry is not None:
        telemetry.n_evals += n_evals

    return y


# ── Forward Integrator ───────────────────────────────────────────────

def _solve_ivp_np(
    func: Callable[[float, Any], Any],
    y0_data: np.ndarray,
    t_span: Sequence[float],
    method: str = "rk4",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    options: Optional[Dict[str, Any]] = None,
    telemetry: Optional[AdaptiveStepTelemetry] = None,
) -> List[np.ndarray]:
    """
    Integrates forward across a sequence of timestamps [t0, t1, ...] in NumPy mode.
    Returns list of state arrays at each timestamp in t_span.
    """
    opts = options or {}
    n_steps_default = int(opts.get("n_steps", 20))

    def deriv_fn(t: float, s: List[np.ndarray]) -> List[np.ndarray]:
        f_val = _eval_func_np(func, t, s[0])
        return [f_val]

    results: List[np.ndarray] = [y0_data.copy()]
    curr_state = [y0_data.copy()]

    for i in range(len(t_span) - 1):
        t_a = float(t_span[i])
        t_b = float(t_span[i + 1])

        if method == "euler":
            n_st = int(opts.get("n_steps", n_steps_default))
            curr_state = _integrate_interval_euler(
                deriv_fn, curr_state, t_a, t_b, n_steps=n_st, telemetry=telemetry
            )
        elif method == "rk4":
            n_st = int(opts.get("n_steps", n_steps_default))
            curr_state = _integrate_interval_rk4(
                deriv_fn, curr_state, t_a, t_b, n_steps=n_st, telemetry=telemetry
            )
        elif method in ("dopri5", "rk45", "adaptive"):
            curr_state = _integrate_interval_dopri5(
                deriv_fn,
                curr_state,
                t_a,
                t_b,
                rtol=rtol,
                atol=atol,
                max_steps=int(opts.get("max_steps", 2000)),
                h_init=opts.get("step_size", None),
                active_indices=[0],
                telemetry=telemetry,
            )
        else:
            raise ValueError(f"Unknown ODE solver method: '{method}'. Choose 'euler', 'rk4', or 'dopri5'.")

        results.append(curr_state[0].copy())

    return results


# ── Continuous Adjoint Backward Integrator ───────────────────────────

def _run_adjoint_backward(
    func: Callable[[float, Any], Any],
    params: List[Tensor],
    y_traj: List[np.ndarray],
    t_span: Sequence[float],
    grad_outputs: List[np.ndarray],
    method: str = "rk4",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Executes the continuous Pontryagin Adjoint State Method backwards across t_span.
    Augmented state: S(t) = [y(t), a(t), grad_p0(t), grad_p1(t), ...]
    Solves:
      dy/dt = f(t, y)
      da/dt = - a^T (∂f/∂y)
      d(grad_p)/dt = - a^T (∂f/∂p)
    """
    opts = options or {}
    n_steps_default = int(opts.get("n_steps", 20))
    n_params = len(params)

    # Augmented derivative function
    def aug_deriv_fn(t: float, s: List[np.ndarray]) -> List[np.ndarray]:
        y_curr = s[0]
        a_curr = s[1]

        y_tensor = Tensor(y_curr, requires_grad=True)
        # Evaluate f with autograd enabled for instantaneous single-step VJP
        f_tensor = func(t, y_tensor)
        f_data = f_tensor.data.copy()

        adj_tensor = Tensor(a_curr, requires_grad=False)
        # Instantaneous scalar projection: phi = sum(f_i * a_i)
        phi = (f_tensor * adj_tensor).sum()

        # Zero gradients on y_tensor and params
        y_tensor.zero_grad()
        for p in params:
            p.zero_grad()

        # Instantaneous single-step backward
        phi.backward()

        vjp_y = y_tensor.grad.copy() if y_tensor.grad is not None else np.zeros_like(y_curr)
        vjp_params = [
            (p.grad.copy() if p.grad is not None else np.zeros_like(p.data))
            for p in params
        ]

        # Reset parameter gradients
        for p in params:
            p.zero_grad()

        # Augmented dynamics: [f, -vjp_y, -vjp_p0, -vjp_p1, ...]
        ds: List[np.ndarray] = [f_data, -vjp_y]
        for vjp_p in vjp_params:
            ds.append(-vjp_p)
        return ds

    # Initialize at terminal time t_{K-1}
    last_idx = len(t_span) - 1
    curr_y = y_traj[last_idx].copy()
    curr_adj = grad_outputs[last_idx].copy()
    curr_param_grads = [np.zeros_like(p.data) for p in params]

    curr_state = [curr_y, curr_adj] + curr_param_grads

    # Integrate backwards from t_{i} down to t_{i-1}
    for i in range(last_idx, 0, -1):
        t_start = float(t_span[i])
        t_end = float(t_span[i - 1])

        if method == "euler":
            n_st = int(opts.get("n_steps", n_steps_default))
            curr_state = _integrate_interval_euler(
                aug_deriv_fn, curr_state, t_start, t_end, n_steps=n_st
            )
        elif method == "rk4":
            n_st = int(opts.get("n_steps", n_steps_default))
            curr_state = _integrate_interval_rk4(
                aug_deriv_fn, curr_state, t_start, t_end, n_steps=n_st
            )
        elif method in ("dopri5", "rk45", "adaptive"):
            curr_state = _integrate_interval_dopri5(
                aug_deriv_fn,
                curr_state,
                t_start,
                t_end,
                rtol=rtol,
                atol=atol,
                max_steps=int(opts.get("max_steps", 2000)),
                h_init=opts.get("step_size", None),
                active_indices=[0, 1],  # Error control on y and adjoint state
            )
        else:
            raise ValueError(f"Unknown solver: {method}")

        # Add incoming gradient at timestamp t_{i-1}
        if grad_outputs[i - 1] is not None:
            curr_state[1] += grad_outputs[i - 1]

    grad_y0 = curr_state[1]
    grad_params = curr_state[2:]
    return grad_y0, grad_params


# ── Direct Unrolled Autograd Integrator ──────────────────────────────

def _solve_ivp_unrolled_tensor(
    func: Callable[[float, Any], Any],
    y0: Tensor,
    t_span: Sequence[float],
    method: str = "rk4",
    options: Optional[Dict[str, Any]] = None,
) -> List[Tensor]:
    """
    Direct unrolled autograd through solver steps.
    Constructs the computation graph directly through discrete operations.
    """
    opts = options or {}
    n_steps = int(opts.get("n_steps", 20))

    results: List[Tensor] = [y0]
    y = y0

    for i in range(len(t_span) - 1):
        t_a = float(t_span[i])
        t_b = float(t_span[i + 1])
        dt = (t_b - t_a) / max(n_steps, 1)
        t = t_a

        if method == "euler":
            for _ in range(n_steps):
                f_val = func(t, y)
                if not isinstance(f_val, Tensor):
                    f_val = Tensor(f_val)
                y = y + f_val * dt
                t += dt
        elif method == "rk4":
            half_dt = 0.5 * dt
            sixth_dt = dt / 6.0
            for _ in range(n_steps):
                k1 = func(t, y)
                if not isinstance(k1, Tensor):
                    k1 = Tensor(k1)

                s2 = y + k1 * half_dt
                k2 = func(t + half_dt, s2)
                if not isinstance(k2, Tensor):
                    k2 = Tensor(k2)

                s3 = y + k2 * half_dt
                k3 = func(t + half_dt, s3)
                if not isinstance(k3, Tensor):
                    k3 = Tensor(k3)

                s4 = y + k3 * dt
                k4 = func(t + dt, s4)
                if not isinstance(k4, Tensor):
                    k4 = Tensor(k4)

                y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * sixth_dt
                t += dt
        else:
            raise ValueError(
                f"Direct unrolled mode only supports fixed-step 'euler' or 'rk4' (got '{method}'). "
                f"For adaptive solvers, set use_adjoint=True."
            )

        results.append(y)

    return results


# ── High-Level Functional API: odeint ────────────────────────────────

def odeint(
    func: Union[Module, Callable],
    y0: Union[Tensor, np.ndarray, Sequence[float], float],
    t_span: Union[Tuple[float, float], Sequence[float], np.ndarray, Tensor] = (0.0, 1.0),
    method: str = "rk4",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    use_adjoint: bool = True,
    options: Optional[Dict[str, Any]] = None,
    return_trajectory: Optional[bool] = None,
    return_telemetry: bool = False,
) -> Union[Tensor, Tuple[Tensor, AdaptiveStepTelemetry]]:
    """
    S.U.T.R.A. Functional Ordinary Differential Equation Integrator.

    Integrates dy/dt = func(t, y) from t_span[0] to t_span[-1].

    Args:
        func: Vector field module or callable f(t, y), f(y, t), or f(y).
        y0: Initial state (Tensor, array-like, or scalar).
        t_span: (t0, t1) 2-tuple, or sequence of evaluation time points [t0, t1, ..., tK].
        method: ODE solver ('euler', 'rk4', 'dopri5').
        rtol: Relative error tolerance for dopri5.
        atol: Absolute error tolerance for dopri5.
        use_adjoint: If True, uses Pontryagin Adjoint State Method with O(1) memory.
                     If False, uses direct unrolled autograd graph.
        options: Solver options (e.g. {'n_steps': 50, 'max_steps': 2000, 'step_size': 0.01}).
        return_trajectory: If True, returns states across all timestamps in t_span.
                           If False, returns only the terminal state.
                           If None, False for 2-tuple (t0, t1), True for longer sequences.
        return_telemetry: If True, returns (solution, telemetry_object).

    Returns:
        Tensor representing integrated state (or tuple of Tensor and AdaptiveStepTelemetry).
    """
    # 1. Normalize input tensor
    if not isinstance(y0, Tensor):
        y0_tensor = Tensor(np.asarray(y0, dtype=np.float64), requires_grad=False)
    else:
        y0_tensor = y0

    # 2. Normalize timestamps
    if isinstance(t_span, Tensor):
        t_seq = [float(x) for x in t_span.data.flat]
    elif isinstance(t_span, np.ndarray):
        t_seq = [float(x) for x in t_span.flat]
    else:
        t_seq = [float(x) for x in t_span]

    if len(t_seq) < 2:
        raise ValueError(f"t_span must contain at least 2 time points, got {t_span}")

    # Determine whether to return trajectory or terminal state
    if return_trajectory is None:
        return_trajectory = not (isinstance(t_span, tuple) and len(t_span) == 2)

    wrapped_func = _wrap_ode_func(func)
    telemetry = AdaptiveStepTelemetry(solver=method, rtol=rtol, atol=atol)

    # Collect trainable parameters if func is a Module
    trainable_params: List[Tensor] = []
    if isinstance(func, Module):
        trainable_params = [p for p in func.parameters() if p.requires_grad]
    elif hasattr(func, "parameters") and callable(getattr(func, "parameters")):
        trainable_params = [p for p in func.parameters() if getattr(p, "requires_grad", False)]

    needs_grad = y0_tensor.requires_grad or len(trainable_params) > 0

    # Case A: Forward-only without gradient tracking (runs NumPy solver directly for all methods)
    if not needs_grad:
        y_traj_np = _solve_ivp_np(
            wrapped_func,
            y0_tensor.data,
            t_seq,
            method=method,
            rtol=rtol,
            atol=atol,
            options=options,
            telemetry=telemetry,
        )
        if return_trajectory:
            out_data = np.stack(y_traj_np, axis=0)
        else:
            out_data = y_traj_np[-1].copy()

        out = Tensor(out_data, requires_grad=False, _op=f"sutra_odeint_{method}")
        if return_telemetry:
            return out, telemetry
        return out

    # Case B: Direct Unrolled Autograd Mode (use_adjoint=False)
    if not use_adjoint:
        traj_tensors = _solve_ivp_unrolled_tensor(
            wrapped_func,
            y0_tensor,
            t_seq,
            method=method,
            options=options,
        )
        if return_trajectory:
            stacked_data = np.stack([t.data for t in traj_tensors], axis=0)
            # Stack into single differentiable tensor if possible or return terminal
            out = traj_tensors[-1] if not return_trajectory else Tensor(stacked_data, requires_grad=needs_grad)
            # For trajectory unrolled, link to final or components
            out._prev = tuple(traj_tensors)
            out._op = f"unrolled_{method}"
        else:
            out = traj_tensors[-1]

        if return_telemetry:
            telemetry.n_steps = len(t_seq) - 1
            return out, telemetry
        return out

    # Case C: S.U.T.R.A. Continuous Adjoint Mode (use_adjoint=True, O(1) Memory)
    # Forward integration executes on NumPy arrays inside no_grad()
    y_traj_np = _solve_ivp_np(
        wrapped_func,
        y0_tensor.data,
        t_seq,
        method=method,
        rtol=rtol,
        atol=atol,
        options=options,
        telemetry=telemetry,
    )

    if return_trajectory:
        out_data = np.stack(y_traj_np, axis=0)
    else:
        out_data = y_traj_np[-1].copy()

    out = Tensor(
        out_data,
        requires_grad=needs_grad,
        _children=(y0_tensor, *trainable_params) if needs_grad else (),
        _op=f"sutra_odeint_{method}",
    )

    if needs_grad:
        def _backward() -> None:
            incoming_g = out.grad
            if return_trajectory:
                # incoming_g has shape (K, *y0.shape)
                grad_outputs = [incoming_g[i] for i in range(len(t_seq))]
            else:
                # incoming_g has shape (*y0.shape), corresponds to terminal state t_seq[-1]
                grad_outputs = [np.zeros_like(y_traj_np[i]) for i in range(len(t_seq) - 1)]
                grad_outputs.append(incoming_g)

            grad_y0, grad_params = _run_adjoint_backward(
                wrapped_func,
                trainable_params,
                y_traj_np,
                t_seq,
                grad_outputs,
                method=method,
                rtol=rtol,
                atol=atol,
                options=options,
            )

            # Accumulate gradient into y0
            if y0_tensor.requires_grad:
                unbroadcasted_g0 = Tensor._unbroadcast(grad_y0, y0_tensor.shape)
                y0_tensor.grad += unbroadcasted_g0

            # Accumulate gradients into parameters
            for p, gp in zip(trainable_params, grad_params):
                if p.requires_grad:
                    unbroadcasted_gp = Tensor._unbroadcast(gp, p.shape)
                    p.grad += unbroadcasted_gp

        out._backward = _backward

    if return_telemetry:
        return out, telemetry
    return out


# ── NeuralODE Module ─────────────────────────────────────────────────

class NeuralODE(Module):
    """
    Continuous-Depth Neural ODE Layer.

    Wraps a parameterized vector field func(t, y) and computes forward states
    via numerical integration, differentiated with O(1) memory via S.U.T.R.A.

    Usage:
        import minigrad.nn as nn
        from minigrad.sutra import NeuralODE

        f = nn.Sequential([nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2)])
        node = NeuralODE(f, t_span=(0.0, 1.0), solver='rk4')

        x = Tensor([[1.0, 2.0]], requires_grad=True)
        out = node(x)
        out.sum().backward()
    """

    def __init__(
        self,
        func: Union[Module, Callable],
        t_span: Union[Tuple[float, float], Sequence[float], np.ndarray, Tensor] = (0.0, 1.0),
        solver: str = "rk4",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        use_adjoint: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.func = func
        self.t_span = t_span
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self.options = options or {}
        self.telemetry: Optional[AdaptiveStepTelemetry] = None

    def forward(self, x: Tensor, t_span: Optional[Any] = None) -> Tensor:
        """
        Computes terminal state h(T) given initial condition h(0) = x.
        """
        span = t_span if t_span is not None else self.t_span
        res, telem = odeint(
            self.func,
            x,
            span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            use_adjoint=self.use_adjoint,
            options=self.options,
            return_trajectory=False,
            return_telemetry=True,
        )
        self.telemetry = telem
        return res

    def trajectory(self, x: Tensor, t_span: Optional[Any] = None) -> Tensor:
        """
        Computes states along the continuous trajectory across all timestamps in t_span.
        Returns Tensor of shape (len(t_span), *x.shape).
        """
        span = t_span if t_span is not None else self.t_span
        res, telem = odeint(
            self.func,
            x,
            span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            use_adjoint=self.use_adjoint,
            options=self.options,
            return_trajectory=True,
            return_telemetry=True,
        )
        self.telemetry = telem
        return res

    def __repr__(self) -> str:
        return (
            f"NeuralODE(func={self.func}, solver='{self.solver}', "
            f"t_span={self.t_span}, use_adjoint={self.use_adjoint})"
        )


# ── Unified SUTRA Namespace ──────────────────────────────────────────

class SUTRA:
    """
    S.U.T.R.A. — State-space Unified Time-continuous Runge-Kutta Adjoint
    Unified Engine Interface.
    """
    odeint = staticmethod(odeint)
    NeuralODE = NeuralODE
    AdaptiveStepTelemetry = AdaptiveStepTelemetry

    @staticmethod
    def solvers() -> List[str]:
        return ["euler", "rk4", "dopri5"]
