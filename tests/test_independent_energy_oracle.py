import numpy as np


def calculate_ising_energy_independent(state, h, J):
    """
    Independent implementation of Ising energy calculation for testing.
    This is derived directly from the specification and is independent of the solver code.
    
    Parameters
    ----------
    state : array-like
        Spin configuration (N elements, each +1 or -1).
    h : array-like
        Local field terms (N elements).
    J : array-like
        Coupling matrix (N x N matrix).
        
    Returns
    -------
    float
        Energy of the state.
    """
    state = np.asarray(state)
    h = np.asarray(h)
    J = np.asarray(J)
    
    # Linear term: sum h_i * s_i
    linear_energy = np.dot(state, h)
    
    # Quadratic term: sum J_ij * s_i * s_j
    # Apply 0.5 factor to account for symmetric matrix double-counting
    quadratic_energy = 0.5 * np.einsum('ij,i,j', J, state, state)
    
    return linear_energy + quadratic_energy


def calculate_delta_energy_independent(state, h, J, spin_idx):
    """
    Independent implementation of delta energy calculation for testing.
    
    Parameters
    ----------
    state : array-like
        Spin configuration (N elements, each +1 or -1).
    h : array-like
        Local field terms (N elements).
    J : array-like
        Coupling matrix (N x N matrix).
    spin_idx : int
        Index of the spin to flip.
        
    Returns
    -------
    float
        Energy difference of flipping the spin.
    """
    state = np.asarray(state)
    h = np.asarray(h)
    J = np.asarray(J)
    
    # ΔE = E_new - E_old = -2 * s_i * (h_i + sum_j J_ij * s_j)
    current_spin = state[spin_idx]
    field_contribution = h[spin_idx]
    coupling_contribution = np.dot(J[spin_idx], state)
    delta_E = -2 * current_spin * (field_contribution + coupling_contribution)
    return delta_E