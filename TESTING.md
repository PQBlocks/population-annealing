# Testing & Test Suite Documentation

> Test suite coverage, execution procedures, and verification breakdown for `pqblocks-population-anneal`.

---

## 1. Test Suite Summary

The test suite contains **22 comprehensive tests** covering the Population Annealing sequential Monte Carlo engine, residual resampling mechanics, Numba JIT acceleration, Kolmogorov-Smirnov statistical equivalence against pure-Python reference oracles, and exhaustive boundary conditions.

- **Status**: 22 passed, 0 failed
- **Runtime**: ~14.0 seconds
- **Statement Coverage**: 49% across 301 statements
- **Python Matrix**: Fully verified on Python 3.11, 3.12, 3.13, and 3.14

---

## 2. Test File Breakdown

| Test File | Tests | Focus Area |
|---|---|---|
| `test_detailed_worked_example.py` | 3 | Analytical energy and delta-energy consistency against independent mathematical oracles, worked 4-spin validation |
| `test_internal_functions.py` | 1 | Internal state helpers, Boltzmann weight re-normalization, free energy summation |
| `test_numba_kernel.py` | 5 | Numba JIT kernel functionality, seed reproducibility, two-sample Kolmogorov-Smirnov distribution equivalence, population collapse detection |
| `test_validation.py` | 12 | Input validation: empty models, single-spin models, dimension mismatches, asymmetric coupling, NaN/Inf detection, non-positive sweeps/replicas, strictly increasing temperature schedule checks |
| `test_worked_example.py` | 1 | Standard 4-spin chain ground state convergence |

---

## 3. Running the Tests

To run the complete test suite:

```bash
PYTHONPATH=src pytest -v
```

To run with coverage reporting:

```bash
PYTHONPATH=src pytest --cov=pqblocks_population_anneal --cov-report=term
```
