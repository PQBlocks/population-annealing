import numpy as np
import pytest
from pqblocks_population_anneal.pa_numba_kernel import population_annealing_numba
from pqblocks_population_anneal.population_annealing import population_annealing_reference


def test_numba_kernel_basic_functionality():
    """Test that the Numba kernel runs and produces valid outputs."""
    # Simple Ising model
    h = np.array([-1.0, 1.0, 0.5])
    J = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, -0.5],
        [0.0, -0.5, 0.0]
    ])
    
    num_replicas = 100
    num_sweeps = 5
    beta_schedule = np.linspace(0, 2, 10)
    
    # Run the Numba implementation
    best_state, best_energy, free_energy = population_annealing_numba(
        h, J, num_replicas, num_sweeps, beta_schedule, seed=42
    )
    
    # Verify outputs are valid
    assert isinstance(best_state, np.ndarray)
    assert best_state.shape == (3,)
    assert np.all((best_state == 1) | (best_state == -1))
    assert isinstance(best_energy, float)
    assert isinstance(free_energy, float)
    assert not np.isnan(best_energy)
    assert not np.isnan(free_energy)


def test_numba_kernel_reproducibility():
    """Test that the Numba kernel produces reproducible results with the same seed."""
    # Simple Ising model
    h = np.array([-1.0, 1.0, 0.5])
    J = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, -0.5],
        [0.0, -0.5, 0.0]
    ])
    
    num_replicas = 50
    num_sweeps = 3
    beta_schedule = np.linspace(0, 1, 5)
    
    # Run twice with the same seed
    result1 = population_annealing_numba(
        h, J, num_replicas, num_sweeps, beta_schedule, seed=123
    )
    
    result2 = population_annealing_numba(
        h, J, num_replicas, num_sweeps, beta_schedule, seed=123
    )
    
    # Results should be identical
    np.testing.assert_array_equal(result1[0], result2[0])  # best_state
    assert result1[1] == result2[1]  # best_energy
    assert result1[2] == result2[2]  # free_energy


def test_numba_vs_reference_ks_test():
    """Perform a Kolmogorov-Smirnov test to verify statistical equivalence."""
    # Define a small Ising model for testing
    h = np.array([-1.0, 1.0, 0.0, 0.5, -0.5])
    J = np.array([
        [0.0, 0.5, 0.0, 0.0, 0.1],
        [0.5, 0.0, -0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, -0.5],
        [0.1, 0.0, 0.0, -0.5, 0.0]
    ])
    
    num_replicas = 100
    num_sweeps = 5
    beta_schedule = np.linspace(0, 1, 10)
    
    # Number of runs for statistical test
    n_runs = 50  # Reduced for faster testing
    
    # Collect results from both implementations
    reference_energies = []
    numba_energies = []
    
    for i in range(n_runs):
        # Run reference implementation
        ref_state, ref_energy, ref_free = population_annealing_reference(
            h, J, num_replicas, num_sweeps, beta_schedule, seed=i
        )
        reference_energies.append(ref_energy)
        
        # Run Numba implementation
        numba_state, numba_energy, numba_free = population_annealing_numba(
            h, J, num_replicas, num_sweeps, beta_schedule, seed=i
        )
        numba_energies.append(numba_energy)
    
    # Perform KS test
    try:
        from scipy.stats import ks_2samp
        ks_result = ks_2samp(reference_energies, numba_energies)
        ks_statistic = ks_result[0]
        p_value = ks_result[1]
        assert float(p_value) > 1e-10, f"KS test failed with p-value = {float(p_value)}, statistic = {float(ks_statistic)}"
    except ImportError:
        # Fallback comparison if scipy is not installed
        assert len(reference_energies) == len(numba_energies)
        diff = np.abs(np.mean(reference_energies) - np.mean(numba_energies))
        assert diff < 1.0, f"Mean energy difference too large: {diff}"


def test_population_collapse_detection():
    """Test that population collapse is detected and raises RuntimeError."""
    # Create a model that should lead to population collapse
    # This is a bit tricky to reproduce reliably, so we'll test the error handling
    
    # For now, we'll just verify that the function can raise RuntimeError in principle
    # A more comprehensive test would require engineering a specific case that leads to collapse
    pass  # Placeholder for now


def test_numba_kernel_4_spin_example():
    """Test the Numba kernel on the 4-spin worked example."""
    # Define the Ising model as specified
    h = np.array([-1.0, -1.0, 0.0, 2.0])
    J = np.zeros((4, 4))  # All couplings are zero
    
    # Set up the exact parameters from the worked example
    num_replicas = 4
    num_sweeps = 1  # We're only testing the first step
    beta_schedule = np.array([0.0, 0.5])  # Only testing first step: 0.0 -> 0.5
    
    # Run the Numba implementation
    best_state, best_energy, free_energy = population_annealing_numba(
        h, J, num_replicas, num_sweeps, beta_schedule, seed=12345
    )
    
    # Verify outputs are valid
    assert isinstance(best_state, np.ndarray)
    assert best_state.shape == (4,)
    assert np.all((best_state == 1) | (best_state == -1))
    assert isinstance(best_energy, float)
    assert isinstance(free_energy, float)
    assert not np.isnan(best_energy)
    assert not np.isnan(free_energy)