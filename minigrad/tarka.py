"""
minigrad/tarka.py — Innovation 4: T.A.R.K.A. (तर्क)
Tensorized Algebraic Reasoning and Knowledge-grounded Autograd.

Neuro-Symbolic Differentiable First-Order Logic & Axiomatic Autograd Constraints:
1. Continuous t-norm logic manifolds in [0, 1] (Product, Łukasiewicz, Gödel).
2. Differentiable first-order quantifiers (Universal forall, Existential exists)
   with targeted gradient flow via softmin / softmax aggregations.
3. Parameterized Neural Predicates and Neural Relations.
4. Axiomatic Semantic Loss functions injecting symbolic constraints
   (transitivity, symmetry, mutual exclusion) directly into the backprop graph.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from minigrad.tensor import Tensor
from minigrad.nn.module import Module
from minigrad import ops


# ── T-Norm Logic Systems ─────────────────────────────────────────────

class TNorm:
    """Supported continuous t-norm logic systems."""
    PRODUCT = "product"
    LUKASIEWICZ = "lukasiewicz"
    GODEL = "godel"


# ── Differentiable Logic Tensor Core ──────────────────────────────────

class LogicTensor:
    """
    A Continuous Degree-of-Truth Tensor representing propositions in [0, 1].

    Implements continuous t-norm fuzzy logic operators with full autograd support:
      - `&`  : Conjunction (A ∧ B)
      - `|`  : Disjunction (A ∨ B)
      - `~`  : Negation (¬A)
      - `>>` : Implication (A ⇒ B)
      - `^`  : Equivalence (A ⇔ B)
      - `.forall()`: Differentiable Universal Quantifier (∀x P(x))
      - `.exists()`: Differentiable Existential Quantifier (∃x P(x))
    """

    __slots__ = ("tensor", "tnorm")

    def __init__(
        self,
        data: Union[Tensor, np.ndarray, float, int, Sequence],
        tnorm: str = TNorm.PRODUCT,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(data, Tensor):
            self.tensor = data
        else:
            arr = np.asarray(data, dtype=np.float64)
            self.tensor = Tensor(arr, requires_grad=requires_grad)
        self.tnorm = tnorm.lower()

    @property
    def data(self) -> np.ndarray:
        return self.tensor.data

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self.tensor.grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx: Any) -> LogicTensor:
        return LogicTensor(self.tensor[idx], tnorm=self.tnorm)

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    def item(self) -> float:
        return float(self.tensor.item())

    # ── Logical Operations ───────────────────────────────────────────

    def __invert__(self) -> LogicTensor:
        """
        Continuous Negation: ¬A = 1 - A.
        Standard across Product, Łukasiewicz, and involutive Gödel systems.
        """
        neg_t = Tensor(1.0) - self.tensor
        return LogicTensor(neg_t, tnorm=self.tnorm)

    def __and__(self, other: Union[LogicTensor, Tensor, float, int]) -> LogicTensor:
        """
        Continuous Conjunction (t-norm): A ∧ B.
        - Product:     T(a, b) = a * b
        - Łukasiewicz: T(a, b) = max(0, a + b - 1)
        - Gödel:       T(a, b) = min(a, b)
        """
        o_t = other.tensor if isinstance(other, LogicTensor) else (
            other if isinstance(other, Tensor) else Tensor(float(other))
        )

        if self.tnorm == TNorm.PRODUCT:
            out_t = self.tensor * o_t
        elif self.tnorm == TNorm.LUKASIEWICZ:
            out_t = (self.tensor + o_t - 1.0).relu()
        elif self.tnorm == TNorm.GODEL:
            diff = self.tensor - o_t
            abs_diff = diff.relu() + (-diff).relu()
            out_t = (self.tensor + o_t - abs_diff) * 0.5
        else:
            raise ValueError(f"Unknown t-norm: {self.tnorm}")

        return LogicTensor(out_t, tnorm=self.tnorm)

    def __rand__(self, other: Union[Tensor, float, int]) -> LogicTensor:
        return self.__and__(other)

    def __or__(self, other: Union[LogicTensor, Tensor, float, int]) -> LogicTensor:
        """
        Continuous Disjunction (t-conorm / s-norm): A ∨ B.
        - Product:     S(a, b) = a + b - a * b
        - Łukasiewicz: S(a, b) = min(1, a + b)
        - Gödel:       S(a, b) = max(a, b)
        """
        o_t = other.tensor if isinstance(other, LogicTensor) else (
            other if isinstance(other, Tensor) else Tensor(float(other))
        )

        if self.tnorm == TNorm.PRODUCT:
            out_t = self.tensor + o_t - (self.tensor * o_t)
        elif self.tnorm == TNorm.LUKASIEWICZ:
            sum_t = self.tensor + o_t
            out_t = Tensor(1.0) - (Tensor(1.0) - sum_t).relu()
        elif self.tnorm == TNorm.GODEL:
            diff = self.tensor - o_t
            abs_diff = diff.relu() + (-diff).relu()
            out_t = (self.tensor + o_t + abs_diff) * 0.5
        else:
            raise ValueError(f"Unknown t-norm: {self.tnorm}")

        return LogicTensor(out_t, tnorm=self.tnorm)

    def __ror__(self, other: Union[Tensor, float, int]) -> LogicTensor:
        return self.__or__(other)

    def __rshift__(self, other: Union[LogicTensor, Tensor, float, int]) -> LogicTensor:
        """
        Continuous Implication: A ⇒ B.
        - Product (Reichenbach smooth implication): I(a, b) = 1 - a + a * b
        - Łukasiewicz: I(a, b) = min(1, 1 - a + b)
        - Gödel: Smooth lower-bounded implication
        """
        o_t = other.tensor if isinstance(other, LogicTensor) else (
            other if isinstance(other, Tensor) else Tensor(float(other))
        )

        if self.tnorm == TNorm.PRODUCT:
            # ¬A ∨ B = (1 - a) + b - (1 - a)*b = 1 - a + a*b
            out_t = Tensor(1.0) - self.tensor + (self.tensor * o_t)
        elif self.tnorm == TNorm.LUKASIEWICZ:
            # min(1, 1 - a + b) = 1 - relu(a - b)
            out_t = Tensor(1.0) - (self.tensor - o_t).relu()
        elif self.tnorm == TNorm.GODEL:
            # Smooth approximation: 1 - relu(a - b)
            out_t = Tensor(1.0) - (self.tensor - o_t).relu()
        else:
            raise ValueError(f"Unknown t-norm: {self.tnorm}")

        return LogicTensor(out_t, tnorm=self.tnorm)

    def __xor__(self, other: Union[LogicTensor, Tensor, float, int]) -> LogicTensor:
        """
        Continuous Equivalence: A ⇔ B = (A ⇒ B) ∧ (B ⇒ A).
        """
        forward = self >> other
        o_logic = other if isinstance(other, LogicTensor) else LogicTensor(other, tnorm=self.tnorm)
        backward = o_logic >> self
        return forward & backward

    # ── Differentiable First-Order Quantifiers ────────────────────────

    def forall(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        tau: float = 0.05,
    ) -> LogicTensor:
        """
        Differentiable Universal Quantifier: ∀x P(x).

        In discrete logic, ∀ is the conjunction over all elements (min).
        In T.A.R.K.A., ∀ is a smooth softmin weighted aggregation:
            ∀_τ(P) = Σ_i P_i * exp(-P_i / τ) / Σ_j exp(-P_j / τ)

        Gradient Property:
        When a single instance violates the rule (P_k -> 0 while others are 1),
        the softmin weight concentrates on instance k, actively steering
        the optimizer to fix the specific violating instance without disturbing
        the rest!
        """
        t = self.tensor
        if t.data.size <= 1 and axis is None:
            return self

        # Flatten if axis is None
        if axis is None:
            t = t.flatten()
            axis = -1

        # Compute stable softmin weights: softmax(-t / tau)
        scaled = t * (-1.0 / max(tau, 1e-6))
        weights = ops.softmax(scaled, axis=axis)

        # Weighted softmin output
        softmin_val = (weights * t).sum(axis=axis)
        return LogicTensor(softmin_val, tnorm=self.tnorm)

    def exists(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        tau: float = 0.05,
    ) -> LogicTensor:
        """
        Differentiable Existential Quantifier: ∃x P(x).

        In discrete logic, ∃ is the disjunction over all elements (max).
        In T.A.R.K.A., ∃ is a smooth softmax weighted aggregation:
            ∃_τ(P) = Σ_i P_i * exp(P_i / τ) / Σ_j exp(P_j / τ)
        """
        t = self.tensor
        if t.data.size <= 1 and axis is None:
            return self

        if axis is None:
            t = t.flatten()
            axis = -1

        # Compute stable softmax weights: softmax(t / tau)
        scaled = t * (1.0 / max(tau, 1e-6))
        weights = ops.softmax(scaled, axis=axis)

        # Weighted softmax output
        softmax_val = (weights * t).sum(axis=axis)
        return LogicTensor(softmax_val, tnorm=self.tnorm)

    # ── Evaluation & Semantic Loss Metrics ────────────────────────────

    def satisfaction(self) -> float:
        """Returns the mean truth satisfaction degree in [0, 1]."""
        return float(np.mean(self.tensor.numpy()))

    def violation(self) -> float:
        """Returns the mean violation degree in [0, 1]."""
        return float(1.0 - self.satisfaction())

    def semantic_loss(self, method: str = "linear", eps: float = 1e-7) -> Tensor:
        """
        Converts the truth satisfaction value into an autograd loss penalty:
        - "linear"        : L = 1.0 - Truth  (exact 0 loss when fully satisfied)
        - "cross_entropy" : L = -log(Truth + eps)
        - "squared"       : L = (1.0 - Truth)^2
        """
        # Aggregate to scalar truth if tensor is multi-dimensional
        truth = self.tensor.mean() if self.tensor.data.size > 1 else self.tensor

        if method == "linear":
            return Tensor(1.0) - truth
        elif method == "cross_entropy" or method == "log":
            return -(truth + eps).log()
        elif method == "squared":
            err = Tensor(1.0) - truth
            return err ** 2
        else:
            raise ValueError(f"Unknown loss method: {method}")

    def __repr__(self) -> str:
        return f"LogicTensor({self.tensor.data}, tnorm='{self.tnorm}')"


# ── Neural Predicates & Relations ─────────────────────────────────────

class NeuralPredicate(Module):
    """
    A Parameterized Neural Predicate P_θ(x) → [0, 1].

    Maps continuous input entities x to a continuous degree of truth
    that property P holds for x, smoothly squashed via Sigmoid.
    """

    def __init__(self, net: Module, tnorm: str = TNorm.PRODUCT) -> None:
        super().__init__()
        self.net = net
        self.tnorm = tnorm

    def forward(self, x: Union[Tensor, np.ndarray, Sequence]) -> LogicTensor:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64))
        logits = self.net(x)
        probs = logits.sigmoid()
        return LogicTensor(probs, tnorm=self.tnorm)


class NeuralRelation(Module):
    """
    A Parameterized Binary Neural Relation R_θ(x, y) → [0, 1].

    Maps pairs of entities (x, y) to a continuous degree of truth
    that binary relation R holds between x and y.
    """

    def __init__(self, net: Module, tnorm: str = TNorm.PRODUCT) -> None:
        super().__init__()
        self.net = net
        self.tnorm = tnorm

    def forward(
        self,
        x: Union[Tensor, np.ndarray, Sequence],
        y: Union[Tensor, np.ndarray, Sequence],
    ) -> LogicTensor:
        """
        Evaluates R(x, y).
        Supports:
          - Pairwise 1-to-1: x has shape (N, D), y has shape (N, D) -> returns (N, 1)
          - All-pairs matrix: when pairwise_matrix is called.
        """
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float64))
        if not isinstance(y, Tensor):
            y = Tensor(np.asarray(y, dtype=np.float64))

        # Concatenate along feature dimension
        xy = ops.concatenate([x, y], axis=-1)
        logits = self.net(xy)
        probs = logits.sigmoid()
        return LogicTensor(probs, tnorm=self.tnorm)

    def pairwise_matrix(self, entities: Union[Tensor, np.ndarray, Sequence]) -> LogicTensor:
        """
        Evaluates R(x_i, x_j) for all pairs (i, j) in the entity set.
        Returns an (N, N) LogicTensor truth matrix.
        """
        if not isinstance(entities, Tensor):
            entities = Tensor(np.asarray(entities, dtype=np.float64))

        n, d = entities.shape[0], entities.shape[1]
        # x_expanded: (N, 1, D) repeated to (N, N, D)
        # y_expanded: (1, N, D) repeated to (N, N, D)
        x_exp = entities.reshape(n, 1, d)
        y_exp = entities.reshape(1, n, d)

        # Broadcasted concatenation using tile / repeat
        x_grid = np.repeat(x_exp.data, n, axis=1)
        y_grid = np.repeat(y_exp.data, n, axis=0)
        xy_data = np.concatenate([x_grid, y_grid], axis=-1).reshape(n * n, 2 * d)

        xy_t = Tensor(xy_data, requires_grad=entities.requires_grad)
        logits = self.net(xy_t)
        probs = logits.sigmoid().reshape(n, n)
        return LogicTensor(probs, tnorm=self.tnorm)


# ── Built-In Differentiable Symbolic Axioms ──────────────────────────

class Axiom:
    """Base class for symbolic constraints and first-order axioms."""

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    def evaluate(self, *args: Any, **kwargs: Any) -> LogicTensor:
        raise NotImplementedError

    def loss(self, *args: Any, **kwargs: Any) -> Tensor:
        truth = self.evaluate(*args, **kwargs)
        return truth.semantic_loss(method="linear") * self.weight


class TransitivityAxiom(Axiom):
    """
    Differentiable Transitivity Constraint on Relation R:
      ∀x, y, z : (R(x, y) ∧ R(y, z)) ⇒ R(x, z)
    """

    def __init__(self, relation: NeuralRelation, name: str = "Transitivity", weight: float = 1.0) -> None:
        super().__init__(name=name, weight=weight)
        self.relation = relation

    def evaluate(self, entities: Union[Tensor, np.ndarray]) -> LogicTensor:
        """
        Evaluates transitivity on an entity set of size N.
        M: (N, N) relation truth matrix where M_ij = R(x_i, x_j).
        Premise:   (R_ij ∧ R_jk) -> shape (N, N, N)
        Consequent: R_ik         -> shape (N, 1, N)
        Rule: (Premise ⇒ Consequent).forall()
        """
        matrix = self.relation.pairwise_matrix(entities)
        m = matrix.tensor  # (N, N)
        n = m.shape[0]

        # m_ij: (N, N, 1), m_jk: (1, N, N)
        m_ij = m.reshape(n, n, 1)
        m_jk = m.reshape(1, n, n)
        # m_ik: (N, 1, N)
        m_ik = m.reshape(n, 1, n)

        # Premise: R_ij ∧ R_jk (Product norm: m_ij * m_jk)
        premise_t = m_ij * m_jk
        premise = LogicTensor(premise_t, tnorm=matrix.tnorm)

        consequent = LogicTensor(m_ik, tnorm=matrix.tnorm)

        # Implication across all N^3 triplets
        implication = premise >> consequent

        # Smooth universal quantifier across the entire (N, N, N) tensor
        return implication.forall(tau=0.05)


class SymmetryAxiom(Axiom):
    """
    Differentiable Symmetry Constraint on Relation R:
      ∀x, y : R(x, y) ⇒ R(y, x)
    """

    def __init__(self, relation: NeuralRelation, name: str = "Symmetry", weight: float = 1.0) -> None:
        super().__init__(name=name, weight=weight)
        self.relation = relation

    def evaluate(self, entities: Union[Tensor, np.ndarray]) -> LogicTensor:
        matrix = self.relation.pairwise_matrix(entities)
        m = matrix.tensor
        m_trans = m.transpose(1, 0)
        implication = matrix >> LogicTensor(m_trans, tnorm=matrix.tnorm)
        return implication.forall(tau=0.05)


class MutualExclusionAxiom(Axiom):
    """
    Differentiable Mutual Exclusion Constraint across K Predicates:
      ∀x : ¬(P_i(x) ∧ P_j(x))  for all i ≠ j
    """

    def __init__(
        self,
        predicates: Sequence[NeuralPredicate],
        name: str = "MutualExclusion",
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.predicates = predicates

    def evaluate(self, x: Union[Tensor, np.ndarray]) -> LogicTensor:
        """
        Evaluates pairwise disjointness on input batch x.
        """
        probs = [p(x) for p in self.predicates]
        k = len(probs)
        disjoint_rules: List[LogicTensor] = []

        for i in range(k):
            for j in range(i + 1, k):
                # ¬(P_i ∧ P_j)
                both = probs[i] & probs[j]
                disjoint = ~both
                disjoint_rules.append(disjoint)

        if not disjoint_rules:
            return LogicTensor(Tensor(1.0))

        # Conjunction of all disjointness pairs
        combined = disjoint_rules[0]
        for rule in disjoint_rules[1:]:
            combined = combined & rule

        return combined.forall(tau=0.05)


class CustomAxiom(Axiom):
    """
    Arbitrary User-Defined Continuous Logic Axiom.
    Evaluates rule_fn(*args, **kwargs) -> LogicTensor.
    """

    def __init__(
        self,
        rule_fn: Callable[..., LogicTensor],
        name: str = "CustomAxiom",
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.rule_fn = rule_fn

    def evaluate(self, *args: Any, **kwargs: Any) -> LogicTensor:
        res = self.rule_fn(*args, **kwargs)
        if not isinstance(res, LogicTensor):
            return LogicTensor(res)
        return res


# ── Diagnostic Telemetry ─────────────────────────────────────────────

@dataclass
class TarkaTelemetry:
    """Telemetry report of symbolic consistency and semantic loss."""
    axiom_satisfactions: Dict[str, float]
    mean_semantic_violation: float
    task_loss: float
    semantic_penalty: float
    total_loss: float

    def summary(self) -> str:
        lines = [
            "T.A.R.K.A. Neuro-Symbolic Telemetry:",
            f"  Task Loss:                 {self.task_loss:.6f}",
            f"  Semantic Penalty:          {self.semantic_penalty:.6f}",
            f"  Total Loss:                {self.total_loss:.6f}",
            f"  Mean Axiomatic Violation:  {self.mean_semantic_violation * 100:.2f}%",
            "  Axiom Satisfaction Breakdown:",
        ]
        for name, sat in self.axiom_satisfactions.items():
            lines.append(f"    - {name:<24}: {sat * 100:.2f}% satisfied")
        return "\n".join(lines)


# ── Semantic Loss Aggregator ─────────────────────────────────────────

class SemanticLoss(Module):
    """
    Neuro-Symbolic Objective Function:
      L_total = L_task + Σ_i λ_i * L_semantic(Axiom_i)

    Penalizes logical contradictions while optimizing supervised loss.
    """

    def __init__(
        self,
        axioms: Sequence[Axiom],
        task_loss_fn: Optional[Callable[[Tensor, Any], Tensor]] = None,
        loss_method: str = "linear",
    ) -> None:
        super().__init__()
        self.axioms = list(axioms)
        self.task_loss_fn = task_loss_fn
        self.loss_method = loss_method
        self.last_telemetry: Optional[TarkaTelemetry] = None

    def forward(
        self,
        task_pred: Optional[Tensor] = None,
        task_target: Optional[Any] = None,
        axiom_inputs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        axiom_inputs = axiom_inputs or {}

        # 1. Supervised Task Loss
        if self.task_loss_fn is not None and task_pred is not None and task_target is not None:
            task_l = self.task_loss_fn(task_pred, task_target)
            task_l_val = task_l.item()
        else:
            task_l = Tensor(0.0)
            task_l_val = 0.0

        # 2. Semantic Axiomatic Losses
        semantic_l = Tensor(0.0)
        satisfactions: Dict[str, float] = {}

        for axiom in self.axioms:
            # Look up input args for this specific axiom
            inp = axiom_inputs.get(axiom.name, axiom_inputs.get("default", None))
            if inp is not None:
                if isinstance(inp, tuple):
                    truth = axiom.evaluate(*inp)
                elif isinstance(inp, dict):
                    truth = axiom.evaluate(**inp)
                else:
                    truth = axiom.evaluate(inp)
            else:
                truth = axiom.evaluate()

            satisfactions[axiom.name] = truth.satisfaction()
            axiom_penalty = truth.semantic_loss(method=self.loss_method) * axiom.weight
            semantic_l = semantic_l + axiom_penalty

        total_l = task_l + semantic_l

        mean_violation = float(np.mean([1.0 - s for s in satisfactions.values()])) if satisfactions else 0.0

        self.last_telemetry = TarkaTelemetry(
            axiom_satisfactions=satisfactions,
            mean_semantic_violation=mean_violation,
            task_loss=task_l_val,
            semantic_penalty=semantic_l.item(),
            total_loss=total_l.item(),
        )

        return total_l


# ── Unified TARKA Namespace ──────────────────────────────────────────

class TARKA:
    """
    T.A.R.K.A. — Tensorized Algebraic Reasoning and Knowledge-grounded Autograd.
    """
    TNorm = TNorm
    LogicTensor = LogicTensor
    NeuralPredicate = NeuralPredicate
    NeuralRelation = NeuralRelation
    Axiom = Axiom
    TransitivityAxiom = TransitivityAxiom
    SymmetryAxiom = SymmetryAxiom
    MutualExclusionAxiom = MutualExclusionAxiom
    CustomAxiom = CustomAxiom
    SemanticLoss = SemanticLoss
    TarkaTelemetry = TarkaTelemetry
