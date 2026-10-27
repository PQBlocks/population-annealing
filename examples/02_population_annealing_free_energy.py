"""
Example 02: Population Annealing with Varying Ensemble Sizes & Free Energy.

Demonstrates how Population Annealing tracks the thermal ensemble,
resamples replicas proportional to Boltzmann weights, and estimates free energy.
"""

import numpy as np
from pqblocks_population_anneal import PopulationAnnealingSampler, solve_ising


def main():
    print("--- Population Annealing Ensemble Scaling & Free Energy ---")

    # Define 8-spin frustrated ring model
    N = 8
    rng = np.random.RandomState(42)
    h = rng.uniform(-0.5, 0.5, size=N)
    J = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        J[i, j] = J[j, i] = rng.uniform(-1.0, 1.0)

    for pop_size in [50, 100, 250]:
        sampler = PopulationAnnealingSampler(num_replicas=pop_size, num_sweeps=10)
        best_state, best_energy, free_energy = sampler.sample(h, J, seed=42)
        print(f"Replicas={pop_size:3d}: Best Energy = {best_energy:.6f}, Free Energy = {free_energy:.6f}")

    print("\nPopulation annealing ensemble evaluation completed successfully.")


if __name__ == "__main__":
    main()
