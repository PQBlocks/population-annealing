# Use Cases & Integration Patterns

> Practical applications and integration scenarios for `pqblocks-population-anneal`.

---

## 1. Navigating Rugged, Multi-Modal Energy Landscapes

In combinatorial problems with heavily fragmented solution spaces (such as frustrated Ising spin systems, Max-Cut on dense graphs, and community detection), classical single-trajectory walkers easily become trapped in unpromising local energy basins.

Population Annealing propagates an entire ensemble of states simultaneously, reallocating computational effort toward promising minima at each temperature reduction:

```python
import numpy as np
from pqblocks_population_anneal import PopulationAnnealingSampler

# Solve frustrated spin system
sampler = PopulationAnnealingSampler(num_replicas=500, num_sweeps=20)
best_state, best_energy, free_energy = sampler.sample(h, J, seed=42)

print(f"Optimal Ground Energy: {best_energy:.4f}")
print(f"Ground Configuration: {best_state}")
```

---

## 2. Estimating Thermodynamic Free Energy

Unlike standard simulated annealing which only outputs final configurations, Population Annealing tracks the partition function ratio $Q_k$ across temperature steps, yielding an estimate of the equilibrium free energy:

$$-\beta_k F(\beta_k) = N \ln(2) + \sum_{m=1}^k \ln Q_m$$

```python
# Free energy estimation is returned directly alongside the ground state
best_state, best_energy, free_energy = sampler.sample(h, J)
print(f"Estimated Free Energy: {free_energy:.6f}")
```

---

## 3. Lossless Integration with QUBO Formulations

Binary quadratic problems ($x \in \{0, 1\}^N$) can be mapped losslessly into Ising spin variables ($s \in \{-1, +1\}^N$) via $x_i = (s_i + 1)/2$ using `pqblocks-qubo`:

```python
from pqblocks_qubo import qubo_to_ising
from pqblocks_population_anneal import PopulationAnnealingSampler

# Convert QUBO to Ising
J, h, offset = qubo_to_ising(Q)

# Run population annealing
sampler = PopulationAnnealingSampler(num_replicas=200, num_sweeps=10)
best_state, ising_energy, _ = sampler.sample(h, J)

# Recover QUBO binary solution and energy
binary_solution = (best_state + 1) // 2
qubo_energy = ising_energy + offset
```
