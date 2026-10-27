import pytest
import numpy as np
from pqblocks_population_anneal.population_annealing import PopulationAnnealingSampler, population_annealing_reference


def test_empty_model():
    """Test that empty models raise ValueError."""
    h = np.array([])
    J = np.array([]).reshape(0, 0)
    
    with pytest.raises(ValueError, match="Model cannot be empty"):
        PopulationAnnealingSampler().sample(h, J)


def test_single_spin():
    """Test that single spin models raise ValueError."""
    h = np.array([1.0])
    J = np.array([[0.0]])
    
    with pytest.raises(ValueError, match="Model must have N >= 2 spins"):
        PopulationAnnealingSampler().sample(h, J)


def test_dimension_mismatch():
    """Test that dimension mismatches raise ValueError."""
    h = np.array([1.0, 2.0])
    J = np.array([[0.0, 1.0, 2.0], 
                  [1.0, 0.0, 3.0]])
    
    with pytest.raises(ValueError, match="Dimension mismatch between h and J"):
        PopulationAnnealingSampler().sample(h, J)


def test_non_symmetric_J():
    """Test that non-symmetric J matrices raise ValueError."""
    h = np.array([1.0, 2.0])
    J = np.array([[0.0, 1.0], 
                  [2.0, 0.0]])  # Not symmetric
    
    with pytest.raises(ValueError, match="Matrix J must be symmetric"):
        PopulationAnnealingSampler().sample(h, J)


def test_nan_inputs():
    """Test that NaN inputs raise ValueError."""
    h = np.array([1.0, np.nan])
    J = np.array([[0.0, 1.0], 
                  [1.0, 0.0]])
    
    with pytest.raises(ValueError, match="Inputs h and J must be finite"):
        PopulationAnnealingSampler().sample(h, J)


def test_inf_inputs():
    """Test that Inf inputs raise ValueError."""
    h = np.array([1.0, np.inf])
    J = np.array([[0.0, 1.0], 
                  [1.0, 0.0]])
    
    with pytest.raises(ValueError, match="Inputs h and J must be finite"):
        PopulationAnnealingSampler().sample(h, J)


def test_negative_sweeps():
    """Test that negative num_sweeps raises ValueError."""
    with pytest.raises(ValueError, match="num_sweeps must be positive"):
        PopulationAnnealingSampler(num_sweeps=-1)


def test_zero_sweeps():
    """Test that zero num_sweeps raises ValueError."""
    with pytest.raises(ValueError, match="num_sweeps must be positive"):
        PopulationAnnealingSampler(num_sweeps=0)


def test_negative_replicas():
    """Test that negative num_replicas raises ValueError."""
    with pytest.raises(ValueError, match="num_replicas must be positive"):
        PopulationAnnealingSampler(num_replicas=-1)


def test_zero_replicas():
    """Test that zero num_replicas raises ValueError."""
    with pytest.raises(ValueError, match="num_replicas must be positive"):
        PopulationAnnealingSampler(num_replicas=0)


def test_invalid_beta_schedule():
    """Test that invalid beta_schedule raises ValueError."""
    # Not starting at 0
    with pytest.raises(ValueError, match="beta_schedule must be strictly increasing and non-negative"):
        PopulationAnnealingSampler(beta_schedule=np.array([0.1, 0.5, 1.0]))
    
    # Not strictly increasing
    with pytest.raises(ValueError, match="beta_schedule must be strictly increasing and non-negative"):
        PopulationAnnealingSampler(beta_schedule=np.array([0.0, 0.5, 0.5, 1.0]))
    
    # Negative values
    with pytest.raises(ValueError, match="beta_schedule must be strictly increasing and non-negative"):
        PopulationAnnealingSampler(beta_schedule=np.array([0.0, 0.5, -0.1, 1.0]))
    
    # Too few elements
    with pytest.raises(ValueError, match="beta_schedule must have at least 2 elements"):
        PopulationAnnealingSampler(beta_schedule=np.array([0.0]))


def test_all_zero_model():
    """Test that all-zero models are valid and proceed normally."""
    h = np.array([0.0, 0.0])
    J = np.array([[0.0, 0.0], 
                  [0.0, 0.0]])
    
    # Should not raise any exception
    sampler = PopulationAnnealingSampler()
    best_state, best_energy, free_energy = sampler.sample(h, J, seed=42)
    
    # Verify outputs are valid
    assert isinstance(best_state, np.ndarray)
    assert best_state.shape == (2,)
    assert np.all((best_state == 1) | (best_state == -1))
    assert isinstance(best_energy, float)
    assert isinstance(free_energy, float)