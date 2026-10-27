# Verification Evidence & Traceability

> Detailed execution evidence, command outputs, and reproducibility artifacts for `pqblocks-population-anneal`.

---

## 1. Clean-Room Test Execution Evidence

Full test execution output in a clean virtual environment running Python 3.14.8:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /tmp/pqblocks-gate-pa/tests
configfile: pyproject.toml
testpaths: tests
plugins: hypothesis-6.168.0, cov-7.1.0, anyio-4.15.1
collected 22 items

tests/test_detailed_worked_example.py ...                                [ 13%]
tests/test_internal_functions.py .                                       [ 18%]
tests/test_numba_kernel.py .....                                         [ 40%]
tests/test_validation.py ............                                    [ 95%]
tests/test_worked_example.py .                                           [100%]

============================= 22 passed in 14.43s ==============================
```

---

## 2. Example Execution Evidence

### Quickstart Execution:

```bash
$ python3 examples/quickstart.py
Best Spin State: [-1  1 -1  1]
Best Energy: -6.100000
Estimated Free Energy: -6.102126
```

### Ensemble Scaling & Free Energy Evaluation:

```bash
$ python3 examples/02_population_annealing_free_energy.py
--- Population Annealing Ensemble Scaling & Free Energy ---
Replicas= 50: Best Energy = -5.348060, Free Energy = -5.344189
Replicas=100: Best Energy = -5.348060, Free Energy = -5.365068
Replicas=250: Best Energy = -5.348060, Free Energy = -5.356215

Population annealing ensemble evaluation completed successfully.
```
