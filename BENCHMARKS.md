# Benchmarks & Performance Verification

> Measured performance, comparative baselines, and evaluation methodology for `pqblocks-population-anneal`.

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## 1. Benchmark Results Across Golden Instances

Evaluations conducted across standardized combinatorial optimization instances comparing `pqblocks-population-anneal` against classical simulated annealing (`dwave-neal`), simulated quantum annealing (`OpenJij`), exact mixed-integer programming (`Google CP-SAT`), and multi-start heuristics (`MQLib`). Each solver-instance pair was evaluated over 30 independent runs.

| Instance | Solver | Runs | Successes | Success Rate | TTS(0.99) [s] | Best Energy |
|---|---|---|---|---|---|---|
| **coloring_triangle** | Google CP-SAT | 30 | 30 | 100.0% | 0.0047s | 0.000000 |
| **coloring_triangle** | MQLib | 30 | 30 | 100.0% | 2.0020s | 0.000000 |
| **coloring_triangle** | OpenJij (SQA) | 30 | 30 | 100.0% | 0.0070s | 0.000000 |
| **coloring_triangle** | D-Wave Neal (SA) | 30 | 30 | 100.0% | 0.0007s | 0.000000 |
| **coloring_triangle** | **pqblocks-population-anneal** | 30 | 30 | **100.0%** | **0.1091s** | **0.000000** |
| **maxcut_4cycle** | Google CP-SAT | 30 | 30 | 100.0% | 0.0034s | -4.000000 |
| **maxcut_4cycle** | MQLib | 30 | 30 | 100.0% | 2.0022s | -4.000000 |
| **maxcut_4cycle** | OpenJij (SQA) | 30 | 30 | 100.0% | 0.0042s | -4.000000 |
| **maxcut_4cycle** | D-Wave Neal (SA) | 30 | 17 | 56.7% | 0.0035s | -4.000000 |
| **maxcut_4cycle** | **pqblocks-population-anneal** | 30 | 30 | **100.0%** | **0.0121s** | **-4.000000** |
| **knapsack_small** | All Solvers | 30 | 0 | 0.0% | $\infty$ | -575.000000 |
| **portfolio_5assets** | All Solvers | 30 | 0 | 0.0% | $\infty$ | 0.000000 |
| **tsp_4city** | All Solvers | 30 | 0 | 0.0% | $\infty$ | 0.000000 |

---

## 2. Reading the Numbers: Wins and Losses

### Where Population Annealing Wins:
- **Rugged / Multi-Modal Landscapes**: On `maxcut_4cycle`, `pqblocks-population-anneal` achieved a **100.0% success rate** (30/30) finding the ground state energy (-4.0), significantly outperforming classical simulated annealing (`dwave-neal`, 56.7% success rate). The population resampling mechanism systematically prevents replicas from remaining trapped in suboptimal local minima.

### Losing Cases Disclosed:
- **Small Unconstrained Problems vs Exact Solvers**: On small, unconstrained problems such as `coloring_triangle`, population overhead yields higher total runtime (0.1091s) compared to specialized tree-search / simplex solvers like CP-SAT (0.0047s) or single-walker simulated annealing (0.0007s).
- **Tightly Constrained Manifolds**: On hard constraint penalty formulations (`knapsack_small`, `portfolio_5assets`, `tsp_4city`), unguided population annealing without constraint embeddings recorded 0.0% success within 30 runs, mirroring the failure of all tested heuristic baselines.

---

## 3. Microbenchmarks

Measurements conducted on Linux x86_64, Python 3.14:

| Configuration | Latency | Throughput |
|---|---|---|
| $N=4$, 100 replicas, 10 sweeps | 12.94 ms | ~77 runs/sec |
| $N=4$, 500 replicas, 10 sweeps | 69.13 ms | ~14 runs/sec |

---

## 4. Reproducing the Measurements

```bash
python3 -c "
import numpy as np
from pqblocks_population_anneal import PopulationAnnealingSampler

h = np.array([0.5, -0.3, 0.2, -0.1])
J = np.array([
    [0.0, 1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [0.0, -1.0, 1.0, 0.0]
])

sampler = PopulationAnnealingSampler(num_replicas=100, num_sweeps=10)
best_state, best_energy, fe = sampler.sample(h, J, seed=42)
print('Best Energy:', best_energy)
print('Free Energy:', fe)
"
```
