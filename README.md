# pqblocks-population-anneal

> Quantum-inspired Population Annealing Monte Carlo heuristic solver for Ising and QUBO combinatorial optimization.

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## What It Is

`pqblocks-population-anneal` implements Population Annealing (PA), a sequential Monte Carlo optimization algorithm for Ising spin-glass and Quadratic Unconstrained Binary Optimization (QUBO) problems.

Rather than propagating a single thermal walker as in classical Simulated Annealing, Population Annealing maintains an ensemble of $R$ parallel replica configurations. As temperature decreases, replicas undergo thermal Monte Carlo relaxation combined with population resampling proportional to Boltzmann weights $\exp(-\Delta \beta E)$. Replicas in low-energy states replicate, while those in high-energy states are pruned.

This mechanism allows the population to overcome deep local energy barriers and explore rugged, multi-modal energy landscapes on standard classical CPUs.

---

## Key Features

- **Ensemble Resampling**: Implements residual resampling to replicate low-energy configurations and prune suboptimal states across inverse temperature steps.
- **Dynamic Boltzmann Reweighting**: Re-weights the population using energy differentials relative to current ground states to prevent floating-point underflow.
- **Thermodynamic Free Energy Estimation**: Tracks the dimensionless free energy $-\beta F(\beta)$ as an intrinsic property of the thermal trajectory.
- **Accelerated Numba JIT Kernels**: Optimized inner Metropolis-Hastings sweep kernels with pure-Python fallback.
- **Lossless QUBO Mapping**: Directly compatible with `pqblocks-qubo` for seamless binary-to-spin mapping.

---

## Installation

```bash
pip install pqblocks-population-anneal
```

---

## Quickstart

```python
import numpy as np
from pqblocks_population_anneal import PopulationAnnealingSampler

# Define a 4-spin chain problem
h = np.array([0.5, -0.3, 0.2, -0.1])
J = np.array([
    [0.0, 1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [0.0, -1.0, 1.0, 0.0]
])

# Initialize sampler with 100 replicas and 10 sweeps per temperature step
sampler = PopulationAnnealingSampler(num_replicas=100, num_sweeps=10)
best_state, best_energy, free_energy = sampler.sample(h, J, seed=42)

print(f"Best Spin State: {best_state}")
print(f"Best Energy: {best_energy:.6f}")
print(f"Estimated Free Energy: {free_energy:.6f}")
```

Execution output:

```text
Best Spin State: [-1  1 -1  1]
Best Energy: -6.100000
Estimated Free Energy: -6.102126
```

---

## Functional API: `solve_ising()`

For concise, one-line evaluations:

```python
from pqblocks_population_anneal import solve_ising

best_state, best_energy, free_energy = solve_ising(
    h=h,
    J=J,
    num_replicas=250,
    num_sweeps=10,
    seed=42
)
```

---

## Documentation

- [BENCHMARKS.md](BENCHMARKS.md) — Empirical performance comparisons, TTS evaluations, and winning/losing cases
- [TESTING.md](TESTING.md) — Test suite layout, coverage details, and validation procedures
- [EVIDENCE.md](EVIDENCE.md) — Clean-room execution logs and reproducibility artifacts
- [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) — Formal release verification gates and checks
- [USE_CASES.md](USE_CASES.md) — Multi-modal landscape exploration and QUBO integration patterns
- [SECURITY_NOTES.md](SECURITY_NOTES.md) — Numerical safeguards, non-cryptographic scope, and stability bounds
- [AGENTS.md](AGENTS.md) — Machine-readable interfaces and guidelines for autonomous agents
- [CHANGELOG.md](CHANGELOG.md) — Version history and release notes
