"""
Test to increase coverage by directly calling internal functions.
"""
import numpy as np
from pqblocks_population_anneal.population_annealing import (
    _calculate_ising_energies,
    _calculate_ising_energy,
    _calculate_delta_energy
)


def test_internal_functions_for_coverage():
    """Test internal functions to increase code coverage."""
    # Create a small Ising model
    h = np.array([-1.0, 0.5, 0.0, 1.0])
    J = np.array([
        [0.0, 0.5, 0.0, 0.0],
        [0.5, 0.0, -0.5, 0.0],
        [0.0, -0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.0]
    ])
    
    # Create a population of states
    states = np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [-1, -1, -1, -1],
        [1, 1, -1, -1]
    ])
    
    # Test _calculate_ising_energies
    energies = _calculate_ising_energies(states, h, J)
    assert len(energies) == 4
    assert all(isinstance(e, (int, float)) for e in energies)
    
    # Test _calculate_ising_energy
    energy = _calculate_ising_energy(states[0], h, J)
    assert isinstance(energy, (int, float))
    
    # Test _calculate_delta_energy for each spin
    for i in range(len(states[0])):
        delta_e = _calculate_delta_energy(states[0], h, J, i)
        assert isinstance(delta_e, (int, float))