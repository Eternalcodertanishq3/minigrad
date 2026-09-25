"""
tests/test_tarka.py — Unit Tests for Innovation 4: T.A.R.K.A. (तर्क)
Tensorized Algebraic Reasoning and Knowledge-grounded Autograd.
"""
import numpy as np

from minigrad import (
    LogicTensor,
    NeuralPredicate,
    NeuralRelation,
    SemanticLoss,
    Tensor,
)
from minigrad.tarka import (
    TNorm,
    TransitivityAxiom,
    SymmetryAxiom,
    MutualExclusionAxiom,
    CustomAxiom,
)
from minigrad.nn import Linear, Sequential, ReLU, MSELoss
from minigrad.optim import Adam


# ── 1. T-Norm Truth Tables & Exact Algebra ─────────────────────────────

def test_tnorm_truth_tables():
    """
    Verifies Product, Łukasiewicz, and Gödel logic operations against exact
    mathematical truth values on boundary and intermediate points.
    """
    # ── Product Logic (Default)
    a = LogicTensor([1.0, 1.0, 0.0, 0.5], tnorm=TNorm.PRODUCT)
    b = LogicTensor([1.0, 0.0, 0.0, 0.5], tnorm=TNorm.PRODUCT)

    # Conjunction: a * b
    conj = (a & b).numpy()
    np.testing.assert_allclose(conj, [1.0, 0.0, 0.0, 0.25])

    # Disjunction: a + b - a * b
    disj = (a | b).numpy()
    np.testing.assert_allclose(disj, [1.0, 1.0, 0.0, 0.75])

    # Negation: 1 - a
    neg = (~a).numpy()
    np.testing.assert_allclose(neg, [0.0, 0.0, 1.0, 0.5])

    # Implication: 1 - a + a * b
    impl = (a >> b).numpy()
    np.testing.assert_allclose(impl, [1.0, 0.0, 1.0, 0.75])

    # ── Łukasiewicz Logic
    a_luk = LogicTensor([0.7, 0.3], tnorm=TNorm.LUKASIEWICZ)
    b_luk = LogicTensor([0.5, 0.2], tnorm=TNorm.LUKASIEWICZ)

    # Conjunction: max(0, a + b - 1) -> [0.2, 0.0]
    conj_luk = (a_luk & b_luk).numpy()
    np.testing.assert_allclose(conj_luk, [0.2, 0.0])

    # Disjunction: min(1, a + b) -> [1.0, 0.5]
    disj_luk = (a_luk | b_luk).numpy()
    np.testing.assert_allclose(disj_luk, [1.0, 0.5])

    # ── Gödel Logic
    a_god = LogicTensor([0.8, 0.3], tnorm=TNorm.GODEL)
    b_god = LogicTensor([0.6, 0.9], tnorm=TNorm.GODEL)

    # Conjunction: min(a, b) -> [0.6, 0.3]
    conj_god = (a_god & b_god).numpy()
    np.testing.assert_allclose(conj_god, [0.6, 0.3])

    # Disjunction: max(a, b) -> [0.8, 0.9]
    disj_god = (a_god | b_god).numpy()
    np.testing.assert_allclose(disj_god, [0.8, 0.9])


# ── 2. Differentiable First-Order Quantifiers ─────────────────────────

def test_differentiable_quantifiers():
    """
    Verifies Universal (forall) softmin and Existential (exists) softmax aggregations
    and their targeted gradient distribution.
    """
    # Universal Quantifier (forall)
    all_true = LogicTensor([0.99, 0.98, 0.99, 0.97])
    assert all_true.forall(tau=0.05).item() > 0.95

    one_false = LogicTensor([1.0, 1.0, 1.0, 0.01])
    assert one_false.forall(tau=0.05).item() < 0.10  # Must collapse to near 0!

    # Targeted Gradient Flow: forall gradient must concentrate on the violating element
    p_data = np.array([0.9, 0.85, 0.15, 0.95], dtype=np.float64)
    p_tensor = Tensor(p_data.copy(), requires_grad=True)
    p_logic = LogicTensor(p_tensor)

    loss = p_logic.forall(tau=0.05).semantic_loss()
    loss.backward()

    # The violating element index 2 (value 0.15) should receive the dominant gradient!
    grads = p_tensor.grad
    assert grads[2] < grads[0]  # loss = 1 - forall, so dL/dp_2 is most negative!
    assert np.argmin(grads) == 2

    # Existential Quantifier (exists)
    all_false = LogicTensor([0.01, 0.02, 0.01])
    assert all_false.exists(tau=0.05).item() < 0.05

    one_true = LogicTensor([0.01, 0.01, 0.95])
    assert one_true.exists(tau=0.05).item() > 0.90


# ── 3. Smooth Autograd Backpropagation ────────────────────────────────

def test_logic_autograd_backpropagation():
    """
    Verifies that gradients flow smoothly through nested logical formulas
    and match numerical finite differences.
    """
    a = Tensor([0.7], requires_grad=True)
    b = Tensor([0.6], requires_grad=True)
    c = Tensor([0.4], requires_grad=True)

    la, lb, lc = LogicTensor(a), LogicTensor(b), LogicTensor(c)
    # Formula: (A ∧ B) ⇒ C
    formula = (la & lb) >> lc
    loss = formula.semantic_loss(method="linear")
    loss.backward()

    # Analytical formula under product logic:
    # A ∧ B = a * b
    # (a * b) ⇒ c = 1 - (a * b) + (a * b) * c
    # Loss = 1 - Implication = (a * b) - (a * b) * c = a * b * (1 - c)
    # dLoss/da = b * (1 - c) = 0.6 * (1 - 0.4) = 0.36
    # dLoss/db = a * (1 - c) = 0.7 * 0.6 = 0.42
    # dLoss/dc = -a * b = -0.42

    np.testing.assert_allclose(a.grad, [0.36], rtol=1e-4)
    np.testing.assert_allclose(b.grad, [0.42], rtol=1e-4)
    np.testing.assert_allclose(c.grad, [-0.42], rtol=1e-4)


# ── 4. Neural Predicates & Neural Relations ───────────────────────────

def test_neural_predicates_and_relations():
    """
    Verifies forward execution and shape handling of NeuralPredicate and NeuralRelation.
    """
    np.random.seed(42)
    # Unary Predicate: P(x) in [0, 1]
    pred_net = Sequential([Linear(4, 8), ReLU(), Linear(8, 1)])
    pred = NeuralPredicate(pred_net)

    x = np.random.randn(5, 4)
    out_p = pred(x)
    assert isinstance(out_p, LogicTensor)
    assert out_p.shape == (5, 1)
    assert np.all((out_p.numpy() >= 0.0) & (out_p.numpy() <= 1.0))

    # Binary Relation: R(x, y) in [0, 1]
    rel_net = Sequential([Linear(8, 8), ReLU(), Linear(8, 1)])
    rel = NeuralRelation(rel_net)

    entities = np.random.randn(6, 4)
    matrix = rel.pairwise_matrix(entities)
    assert isinstance(matrix, LogicTensor)
    assert matrix.shape == (6, 6)
    assert np.all((matrix.numpy() >= 0.0) & (matrix.numpy() <= 1.0))


# ── 5. Transitivity Axiom Optimization (Zero Data Training) ───────────

def test_transitivity_constraint_optimization():
    """
    Teaches a neural relation to satisfy transitivity purely from the symbolic
    TransitivityAxiom with 0 supervised labels!
    """
    np.random.seed(42)
    rel_net = Sequential([Linear(4, 8), ReLU(), Linear(8, 1)])
    rel = NeuralRelation(rel_net)
    trans_axiom = TransitivityAxiom(rel)

    # 4 synthetic entities: e0 < e1 < e2 < e3
    entities = Tensor(np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]), requires_grad=False)

    initial_satisfaction = trans_axiom.evaluate(entities).satisfaction()

    optimizer = Adam(rel.parameters(), lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        loss = trans_axiom.loss(entities)
        loss.backward()
        optimizer.step()

    final_satisfaction = trans_axiom.evaluate(entities).satisfaction()
    assert final_satisfaction > initial_satisfaction
    assert final_satisfaction > 0.85


# ── 6. Symmetry Axiom Optimization ───────────────────────────────────

def test_symmetry_constraint_optimization():
    """
    Verifies that SymmetryAxiom drives a relation to be symmetric: R(x, y) = R(y, x).
    """
    np.random.seed(123)
    rel_net = Sequential([Linear(4, 8), ReLU(), Linear(8, 1)])
    rel = NeuralRelation(rel_net)
    sym_axiom = SymmetryAxiom(rel)

    entities = Tensor(np.random.randn(4, 2), requires_grad=False)
    initial_satisfaction = sym_axiom.evaluate(entities).satisfaction()

    optimizer = Adam(rel.parameters(), lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        loss = sym_axiom.loss(entities)
        loss.backward()
        optimizer.step()

    final_satisfaction = sym_axiom.evaluate(entities).satisfaction()
    assert final_satisfaction > initial_satisfaction
    assert final_satisfaction > 0.90


# ── 7. Mutual Exclusion Regularization ────────────────────────────────

def test_mutual_exclusion_regularization():
    """
    Verifies that MutualExclusionAxiom prevents two predicates from being
    simultaneously active: ¬(P1(x) ∧ P2(x)).
    """
    np.random.seed(42)
    p1 = NeuralPredicate(Sequential([Linear(2, 4), ReLU(), Linear(4, 1)]))
    p2 = NeuralPredicate(Sequential([Linear(2, 4), ReLU(), Linear(4, 1)]))

    mutex_axiom = MutualExclusionAxiom([p1, p2])

    x = Tensor(np.random.randn(8, 2), requires_grad=False)
    initial_sat = mutex_axiom.evaluate(x).satisfaction()

    params = list(p1.parameters()) + list(p2.parameters())
    optimizer = Adam(params, lr=0.05)

    for _ in range(35):
        optimizer.zero_grad()
        loss = mutex_axiom.loss(x)
        loss.backward()
        optimizer.step()

    final_sat = mutex_axiom.evaluate(x).satisfaction()
    assert final_sat > initial_sat
    assert final_sat > 0.90


# ── 8. End-to-End Joint Supervised & Semantic Loss ────────────────────

def test_joint_semantic_loss_and_telemetry():
    """
    Verifies end-to-end training with SemanticLoss combining task MSE and
    custom axiomatic constraints, and verifies TarkaTelemetry output.
    """
    np.random.seed(42)
    net = Sequential([Linear(2, 4), ReLU(), Linear(4, 1)])
    pred = NeuralPredicate(net)

    # Custom rule: for positive inputs x >= 0, truth degree must be >= 0.8
    # Rule: P(x) >= 0.8 expressed as LogicTensor implication or truth bound
    def positive_prior_rule(x_batch: Tensor) -> LogicTensor:
        p_val = pred(x_batch)
        # Implication: target_bound => p_val
        bound = LogicTensor(Tensor(np.full_like(p_val.numpy(), 0.85)))
        return (bound >> p_val).forall()

    prior_axiom = CustomAxiom(positive_prior_rule, name="PositiveDomainPrior", weight=1.0)
    sem_loss = SemanticLoss(axioms=[prior_axiom], task_loss_fn=MSELoss(), loss_method="linear")

    x_train = Tensor(np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 1.5]]))
    y_target = Tensor(np.array([[0.9], [0.95], [0.88]]))

    optimizer = Adam(pred.parameters(), lr=0.05)
    for _ in range(25):
        optimizer.zero_grad()
        pred_out = pred(x_train).tensor
        total_loss = sem_loss(
            task_pred=pred_out,
            task_target=y_target,
            axiom_inputs={"PositiveDomainPrior": x_train},
        )
        total_loss.backward()
        optimizer.step()

    telem = sem_loss.last_telemetry
    assert telem is not None
    assert telem.total_loss < 0.5
    assert telem.mean_semantic_violation < 0.20
    assert "PositiveDomainPrior" in telem.axiom_satisfactions
    assert "T.A.R.K.A." in telem.summary()
