# Security Notes & Posture

> Security scope, numerical safeguards, and execution boundaries for `pqblocks-population-anneal`.

---

## 1. Non-Cryptographic Classification

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

`pqblocks-population-anneal` is a classical sequential Monte Carlo heuristic optimization solver. It does not implement, expose, or validate cryptographic keys, digital signatures, hash algorithms, or post-quantum cryptographic primitives.

---

## 2. Numerical Safeguards & Stability Safeguards

- **Boltzmann Energy Shift ($\tilde{E}$)**:
  Direct evaluation of $\exp(-\beta E)$ can underflow or overflow IEEE 754 floating-point ranges when energy differences are large. The solver normalizes energies against the minimum population energy:
  $$\tilde{E}_j = E_j - \min_m(E_m)$$
  $$w_j = \exp(-\Delta \beta \cdot \tilde{E}_j)$$
  ensuring all weights $w_j \in (0, 1]$ and preventing numerical underflow during resampling.
- **Population Collapse Detection**:
  If the effective population size $R_{\text{eff}} = 1 / \sum \bar{w}_j^2$ drops drastically, resampling safeguards ensure population diversity is maintained.
- **Input Validation**:
  Matrices and field vectors are strictly verified for finiteness (rejecting NaN, positive and negative infinity), symmetry ($J = J^T$), and zero diagonal elements ($J_{ii} = 0$).
