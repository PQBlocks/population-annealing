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
best_state, best_energy, free_energy = sampler.sample(h, J, seed=42)
print(f'Best Spin State: {best_state}')
print(f'Best Energy: {best_energy:.6f}')
print(f'Estimated Free Energy: {free_energy:.6f}')
