# Changelog

All notable changes to `pqblocks-population-anneal` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-26

### Added
- Population Annealing sequential Monte Carlo heuristic optimization solver for Ising and QUBO models.
- Ensemble propagation with dynamic Boltzmann weight resampling and residual replication.
- Thermodynamic free energy estimation ($-\beta F(\beta)$) computed along annealing trajectories.
- High-performance Numba JIT-compiled inner Metropolis-Hastings sweep kernels with pure-Python fallback.
- Object-oriented (`PopulationAnnealingSampler`) and functional (`solve_ising`) public interfaces.
- Comprehensive test suite with 22 tests passing across Python 3.11, 3.12, 3.13, and 3.14.
- Quickstart and ensemble scaling / free energy evaluation examples.
