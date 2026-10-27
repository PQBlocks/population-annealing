# Autonomous Agent Integration Guide

> Operational instructions and best practices for AI and autonomous software agents integrating `pqblocks-population-anneal`.

---

## 1. Package Purpose

`pqblocks-population-anneal` provides a Population Annealing Monte Carlo heuristic solver for Ising spin-glass and QUBO optimization problems.

---

## 2. Key APIs

```python
from pqblocks_population_anneal import PopulationAnnealingSampler, solve_ising

# Object-oriented interface
sampler = PopulationAnnealingSampler(
    num_replicas=1000,          # int: target population size (default: 1000)
    num_sweeps=10,              # int: Metropolis sweeps per beta step (default: 10)
    beta_schedule=None          # np.ndarray: increasing inverse temperature sequence
)

best_state, best_energy, free_energy = sampler.sample(
    h=h_vector,                 # array-like: (N,) linear bias terms
    J=J_matrix,                 # array-like: (N, N) symmetric coupling matrix
    seed=42                     # int, optional: PRNG seed for deterministic runs
)

# Functional interface
best_state, best_energy, free_energy = solve_ising(
    h=h_vector,
    J=J_matrix,
    num_replicas=250,
    num_sweeps=10,
    seed=42
)
```

---

## 3. Best Practices for Agents

1. **Population Sizing ($R$)**: For fast exploratory runs on small instances ($N \le 20$), `num_replicas=100` executes in $\sim 13\text{ ms}$. For rugged or highly frustrated instances, scale `num_replicas` to 500–1000.
2. **Determinism**: Pass an explicit integer `seed` to ensure bit-identical reproducibility across runs.
3. **Variable Conventions**: Variables are spins $s_i \in \{-1, +1\}$. For binary $x_i \in \{0, 1\}$, use `pqblocks-qubo` for transformation.
