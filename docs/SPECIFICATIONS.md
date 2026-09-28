# miniGrad Frontier Research Paradigm Specifications

This document defines the formal mathematical contracts, computational boundaries, known approximations, and validation standards for the 5 frontier scientific computing paradigms implemented in **miniGrad**.

---

## 1. S.U.T.R.A. (Continuous-Depth Neural ODEs)
*Sanskrit: सूत्र (Thread / Continuous Continuity)*  
**Module:** `minigrad/sutra.py`

### 1.1 Mathematical Formulation
Let $z(t) \in \mathbb{R}^D$ denote a continuous hidden state trajectory governed by a parameterized vector field $f_\theta: \mathbb{R}^D \times \mathbb{R} \to \mathbb{R}^D$:
$$\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(t_0) = z_0$$

The final state $z(t_1)$ is obtained by integrating the initial value problem (IVP):
$$z(t_1) = z(t_0) + \int_{t_0}^{t_1} f_\theta(z(t), t) \, dt$$

Given a scalar loss $\mathcal{L}(z(t_1))$, reverse-mode automatic differentiation computes parameter gradients using the continuous adjoint state $a(t) = \frac{\partial \mathcal{L}}{\partial z(t)}$ via Pontryagin's Maximum Principle:
$$\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial z}$$
$$\frac{d\mathcal{L}}{d\theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta} \, dt$$

### 1.2 Implementation Boundary & Invariants
- **Graph Invariant:** $\mathcal{O}(1)$ autograd computation graph node count with respect to numerical integration step count $N_{\text{steps}}$.
- **Adjoint Integration:** The adjoint state and parameter gradients are computed backwards in time from $t_1$ to $t_0$ as an augmented ODE system, eliminating the $\mathcal{O}(N_{\text{steps}})$ intermediate forward activation node caching inherent in discrete unrolling.

### 1.3 Known Approximations & Error Bounds
- Integration is performed using the adaptive Dormand-Prince 5(4) Runge-Kutta solver (`dopri5`) or fixed-step Euler/RK4 solvers.
- Truncation error is bounded by local error tolerance:
  $$\text{err}_{\text{local}} \le \text{atol} + \text{rtol} \cdot \max(|z_{\text{current}}|, |z_{\text{next}}|)$$

### 1.4 Reference Validation Experiment
- **Showcase:** `examples/12_sutra_neural_ode.py`
- **Validation Standard:** Evaluated on 2D spiral dynamical systems against the closed-form analytical matrix exponential solution $z(t) = \exp(At) z_0$. Observed validation result: convergence within $5.55 \times 10^{-17}$ on the reference experiment.

### 1.5 Limitations & Non-Goals
- Designed for non-stiff ordinary differential equations. Stiff systems requiring implicit differential-algebraic solvers (e.g. Radau IIA) are outside the current architectural scope.

---

## 2. A.V.Y.A.Y.A. (Reversible Invertible Computing)
*Sanskrit: अव्यय (Imperishable / Information-Lossless)*  
**Module:** `minigrad/avyaya.py`

### 2.1 Mathematical Formulation
Input tensors $x \in \mathbb{R}^D$ are partitioned along the feature channel dimension into bipartite halves $x = [x_1, x_2]$ where $x_1, x_2 \in \mathbb{R}^{D/2}$. Transformation proceeds via additive coupling:
$$y_1 = x_1 + f(x_2)$$
$$y_2 = x_2 + g(y_1)$$
where $f$ and $g$ are arbitrary neural sub-networks.

The inverse reconstruction is analytically exact:
$$x_2 = y_2 - g(y_1)$$
$$x_1 = y_1 - f(x_2)$$

### 2.2 Implementation Boundary & Invariants
- **Activation Caching Invariant:** $\mathcal{O}(1)$ forward activation node footprint in the autograd DAG. Forward passes discard intermediate activations across arbitrarily deep stacks (e.g. 50+ to 500+ layers).
- **Dynamic Reconstruction:** Intermediate activation values are dynamically reconstructed in reverse topological order during the backward pass before computing parameter gradients.

### 2.3 Known Approximations & Numerical Accuracy
- **Zero Mathematical Approximations:** Reconstruction error is governed strictly by IEEE-754 floating-point arithmetic.
- **Observed Validation Result:** Empirical reconstruction divergence $\|x_{\text{reconstructed}} - x_{\text{original}}\|_\infty < 3.0 \times 10^{-16}$ in float64 precision on the 50-layer reference showcase.

### 2.4 Reference Validation Experiment
- **Showcase:** `examples/13_avyaya_reversible_computing.py`
- **Validation Standard:** 50-layer reversible neural network tested with zero forward activation caching, verifying identical parameter gradient convergence against non-reversible baseline networks.

### 2.5 Limitations & Non-Goals
- Requires bipartite channel partitioning ($D \pmod 2 = 0$). Non-invertible operations (e.g. spatial pooling, dimension projections) must be placed at the network boundaries.

---

## 3. P.R.A.M.A.N.A. (Distributional Uncertainty Tensors)
*Sanskrit: प्रमाण (Valid Means of Genuine Knowledge)*  
**Module:** `minigrad/pramana.py`

### 3.1 Mathematical Formulation
P.R.A.M.A.N.A. tracks dual-stream computational graphs propagating expectation $\mathbb{E}[X] = \mu_X$ and epistemic variance $\mathrm{Var}[X] = \sigma_X^2$.

For linear combinations of uncorrelated random variables:
$$\mathbb{E}[aX + bY] = a\mu_X + b\mu_Y, \quad \mathrm{Var}[aX + bY] = a^2\sigma_X^2 + b^2\sigma_Y^2$$

For independent bilinear products (Goodman 1960 product algebra):
$$\mathrm{Var}[XY] = \mu_X^2 \sigma_Y^2 + \mu_Y^2 \sigma_X^2 + \sigma_X^2 \sigma_Y^2$$

For non-linear activations $h(X)$ (first-order Taylor analytical approximation):
$$\mathbb{E}[h(X)] \approx h(\mu_X)$$
$$\mathrm{Var}[h(X)] \approx \left( h'(\mu_X) \right)^2 \sigma_X^2$$

### 3.2 Implementation Boundary & Invariants
- Single-pass analytical uncertainty propagation executed concurrently with standard forward passes.
- Maximum likelihood training under heteroscedastic uncertainty via `GaussianNLLLoss`:
  $$\mathcal{L}_{\text{NLL}}(\mu, \sigma^2, y) = \frac{1}{2} \log(\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2} + \text{const}$$

### 3.3 Known Approximations & Observed Discrepancy
- Off-diagonal covariance elements $\mathrm{Cov}[X_i, X_j]$ are assumed zero to preserve $\mathcal{O}(D)$ linear space/time complexity rather than $\mathcal{O}(D^2)$ covariance matrix explosion.
- **Observed Validation Result:** Analytical Taylor expansions demonstrated $< 0.05\%$ discrepancy against empirical Monte Carlo sampling ($N=100,000$) on standard Gaussian test distributions.

### 3.4 Reference Validation Experiment
- **Showcase:** `examples/14_pramana_distributional_uncertainty.py`
- **Validation Standard:** Detection of out-of-distribution (OOD) test inputs via automated epistemic variance spike thresholds.

### 3.5 Limitations & Non-Goals
- Does not model full correlated joint covariance matrices. High-dimensional correlated distributions requiring full Gaussian Processes or MCMC sampling are non-goals.

---

## 4. T.A.R.K.A. (Neuro-Symbolic Differentiable Logic)
*Sanskrit: तर्क (Dialectical Inference & Reductio ad Absurdum)*  
**Module:** `minigrad/tarka.py`

### 4.1 Mathematical Formulation
Continuous relaxations of propositional logic using t-norm fuzzy logics:
- **Product Logic:** $a \land b = ab$, $a \lor b = a + b - ab$, $a \Rightarrow b = \min(1, b/a)$
- **Łukasiewicz Logic:** $a \land b = \max(0, a+b-1)$, $a \lor b = \min(1, a+b)$, $a \Rightarrow b = \min(1, 1 - a + b)$
- **Gödel Logic:** $a \land b = \min(a, b)$, $a \lor b = \max(a, b)$

Universal quantification over a finite grounding set $\{x_i\}_{i=1}^N$ via differentiable softmin:
$$\forall_\tau P(x) = \frac{\sum_{i=1}^N P(x_i) \exp(-P(x_i)/\tau)}{\sum_{j=1}^N \exp(-P(x_j)/\tau)}$$
where $\tau > 0$ is a temperature hyperparameter.

### 4.2 Implementation Boundary & Invariants
- Smooth axiom regularization added directly to objective functions:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \, \mathcal{L}_{\text{semantic}}(\Phi)$$
- Gradient concentration: As $P(x_k) \to 0$ (axiom violation), $\frac{\partial \forall_\tau P}{\partial P(x_k)}$ concentrates $> 99\%$ of backpropagation signal directly on rule-violating instances.

### 4.3 Known Approximations & Error Bounds
- As $\tau \to 0^+$, the softmin quantifier converges uniformly to the non-differentiable hard minimum $\min_i P(x_i)$.
- Finite $\tau \in [0.01, 0.5]$ provides numerically non-vanishing gradients across all training samples.

### 4.4 Reference Validation Experiment
- **Showcase:** `examples/15_tarka_neuro_symbolic_reasoning.py`
- **Validation Standard:** Unsupervised discovery and 100% satisfaction of transitive relations ($R(x, y) \land R(y, z) \Rightarrow R(x, z)$) with zero labeled target data.

### 4.5 Limitations & Non-Goals
- Implements continuous first-order relaxations over finite grounded domains; does not perform discrete theorem proving or infinite-domain combinatorial SAT solving.

---

## 5. S.P.A.N.D.A. (Neuromorphic Event-Driven SNNs)
*Sanskrit: स्पन्द (The Primordial Pulse of Dynamic Consciousness)*  
**Module:** `minigrad/spanda.py`

### 5.1 Mathematical Formulation
Subthreshold biological membrane dynamics modeled via discrete Leaky Integrate-and-Fire (LIF) recurrence over temporal steps $t \in \{1, \dots, T\}$:
$$V[t] = \beta V[t-1] + I[t] - S[t-1] V_{\text{th}}$$
$$S[t] = \Theta(V[t] - V_{\text{th}})$$
where $\beta \in (0, 1)$ is the membrane potential decay factor, $V_{\text{th}}$ is the threshold, and $\Theta$ is the Heaviside step function.

Backpropagation Through Time (BPTT) employs continuous surrogate gradients to bypass the zero-gradient Heaviside barrier ($\frac{d\Theta}{dV} = 0$ almost everywhere):
- **Fast Sigmoid:** $\sigma'(V - V_{\text{th}}) = \frac{1}{(1 + |V - V_{\text{th}}|/\gamma)^2}$
- **ArcTan:** $\sigma'(V - V_{\text{th}}) = \frac{1}{\pi (1 + ((V - V_{\text{th}})/\gamma)^2)}$

### 5.2 Implementation Boundary & Invariants
- Event-driven temporal simulation across discrete integration steps $T$.
- Model-estimated synaptic operations (SynOps) and hardware energy telemetry evaluated under standard 45nm CMOS neuromorphic energy profiles:
  $$E_{\text{MAC}} \approx 4.6\text{ pJ}, \quad E_{\text{AC}} \approx 0.9\text{ pJ}$$

### 5.3 Known Approximations & Error Bounds
- Surrogate gradients provide a smooth pseudo-derivative approximation during the backward pass; forward simulation remains strictly non-linear and discrete binary ($S \in \{0, 1\}$).
- Energy and SynOps metrics are analytical model-estimated telemetry under the specified energy model, not physical oscilloscope measurements on silicon.

### 5.4 Reference Validation Experiment
- **Showcase:** `examples/16_spanda_neuromorphic_snn.py`
- **Validation Standard:** Spatio-temporal event recognition demonstrating $> 95\%$ temporal event sparsity and $> 25\times$ SynOps reduction compared to dense artificial neural networks.

### 5.5 Limitations & Non-Goals
- Executes as a discrete-time software simulator on general-purpose CPUs; does not directly interface with physical asynchronous analog neuromorphic silicon (e.g. Intel Loihi, SynSense Dynap-CNN).
