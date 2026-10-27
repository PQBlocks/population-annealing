import numpy as np
from typing import Optional, Tuple, List


__all__ = [
    "PopulationAnnealingSampler",
    "population_annealing",
    "solve_ising",
    "population_annealing_reference",
]


class PopulationAnnealingSampler:
    """
    A Population Annealing sampler for Ising models.
    
    This is a quantum-inspired, classically executed, honestly benchmarked algorithm.
    """
    
    def __init__(self, 
                 num_replicas: Optional[int] = None,
                 num_sweeps: Optional[int] = None,
                 beta_schedule: Optional[np.ndarray] = None):
        """
        Initialize the PopulationAnnealingSampler.
        
        Parameters
        ----------
        num_replicas : int, optional
            Number of replicas in the population. Default is 1000.
        num_sweeps : int, optional
            Number of Metropolis sweeps per temperature step. Default is 10.
        beta_schedule : array-like, optional
            Strictly increasing beta schedule. Default is geometric from 0 to 5.0 with 100 steps.
        """
        self.num_replicas = num_replicas if num_replicas is not None else 1000
        self.num_sweeps = num_sweeps if num_sweeps is not None else 10
        
        if beta_schedule is not None:
            self.beta_schedule = np.array(beta_schedule)
        else:
            # Default geometric schedule from 0 to 5.0 with 100 steps
            self.beta_schedule = np.geomspace(0.01, 5.0, 99)
            self.beta_schedule = np.insert(self.beta_schedule, 0, 0.0)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate all hyperparameters according to the edge-case ledger."""
        if self.num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive.")
        
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        
        if len(self.beta_schedule) < 2:
            raise ValueError("beta_schedule must have at least 2 elements.")
        
        if self.beta_schedule[0] != 0.0:
            raise ValueError("beta_schedule must be strictly increasing and non-negative. index 0 must be 0.0")
        
        # Check if beta_schedule is strictly increasing
        if not np.all(np.diff(self.beta_schedule) > 0):
            raise ValueError("beta_schedule must be strictly increasing and non-negative. index 0 must be 0.0")
        
        # Check if all values are non-negative
        if np.any(self.beta_schedule < 0):
            raise ValueError("beta_schedule must be strictly increasing and non-negative. index 0 must be 0.0")
    
    def sample(self, 
               h: np.ndarray, 
               J: np.ndarray, 
               seed: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
        """
        Sample from the Ising model using Population Annealing.
        
        Parameters
        ----------
        h : array-like
            Local field terms of the Ising model (N elements).
        J : array-like
            Coupling matrix of the Ising model (N x N matrix).
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        tuple
            - best_state: The best spin configuration found.
            - best_energy: Energy of the best state.
            - final_free_energy: Estimated free energy at final temperature.
        """
        # Validate inputs
        self._validate_ising_model(h, J)
        
        # Import the numba implementation here to avoid circular imports
        try:
            from .pa_numba_kernel import population_annealing_numba
            # Run the numba implementation
            best_state, best_energy, final_free_energy = population_annealing_numba(
                h, J, self.num_replicas, self.num_sweeps, self.beta_schedule, seed
            )
        except ImportError:
            # Fall back to reference implementation if numba is not available
            best_state, best_energy, final_free_energy = population_annealing_reference(
                h, J, self.num_replicas, self.num_sweeps, self.beta_schedule, seed
            )
        
        return best_state, best_energy, final_free_energy
    
    def _validate_ising_model(self, h: np.ndarray, J: np.ndarray):
        """Validate the Ising model parameters according to the edge-case ledger."""
        h = np.asarray(h)
        J = np.asarray(J)
        
        # Check for empty model
        if h.size == 0 or J.size == 0:
            raise ValueError("Model cannot be empty (N=0).")
        
        # Check for single spin
        if h.size < 2:
            raise ValueError("Model must have N >= 2 spins.")
        
        # Check dimensions
        if h.shape[0] != J.shape[0] or J.shape[0] != J.shape[1]:
            raise ValueError("Dimension mismatch between h and J.")
        
        # Check if J is symmetric
        if not np.allclose(J, J.T):
            raise ValueError("Matrix J must be symmetric.")
        
        # Check for NaN or Inf
        if not np.isfinite(h).all() or not np.isfinite(J).all():
            raise ValueError("Inputs h and J must be finite.")


def population_annealing(h: np.ndarray,
                         J: np.ndarray,
                         num_replicas: Optional[int] = None,
                         num_sweeps: Optional[int] = None,
                         beta_schedule: Optional[np.ndarray] = None,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
    """
    Run Population Annealing to find the ground state spin configuration of an Ising model.

    This is a quantum-inspired, classically executed, honestly benchmarked algorithm.

    Parameters
    ----------
    h : array-like
        Local field terms of the Ising model (N elements).
    J : array-like
        Coupling matrix of the Ising model (N x N symmetric matrix with zero diagonal).
    num_replicas : int, optional
        Number of replicas in the population. Default is 1000.
    num_sweeps : int, optional
        Number of Metropolis sweeps per temperature step. Default is 10.
    beta_schedule : array-like, optional
        Strictly increasing sequence of inverse temperatures starting from 0.
        Default is geometric progression from 0.01 to 5.0 with 100 steps (index 0 forced to 0.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        - best_state (np.ndarray): Best spin configuration found (+/-1).
        - best_energy (float): Energy of best state.
        - final_free_energy (float): Estimated free energy at final temperature.
    """
    sampler = PopulationAnnealingSampler(
        num_replicas=num_replicas,
        num_sweeps=num_sweeps,
        beta_schedule=beta_schedule
    )
    return sampler.sample(h, J, seed=seed)


def solve_ising(h: np.ndarray,
                J: np.ndarray,
                num_replicas: Optional[int] = None,
                num_sweeps: Optional[int] = None,
                beta_schedule: Optional[np.ndarray] = None,
                seed: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
    """
    Solve an Ising model using Population Annealing. Functional alias for population_annealing().

    Parameters
    ----------
    h : array-like
        Local field terms of the Ising model (N elements).
    J : array-like
        Coupling matrix of the Ising model (N x N symmetric matrix with zero diagonal).
    num_replicas : int, optional
        Number of replicas in the population. Default is 1000.
    num_sweeps : int, optional
        Number of Metropolis sweeps per temperature step. Default is 10.
    beta_schedule : array-like, optional
        Strictly increasing beta schedule starting at 0.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        - best_state (np.ndarray): Best spin configuration found (+/-1).
        - best_energy (float): Energy of best state.
        - final_free_energy (float): Estimated free energy at final temperature.
    """
    return population_annealing(h, J, num_replicas=num_replicas, num_sweeps=num_sweeps,
                                beta_schedule=beta_schedule, seed=seed)


def population_annealing_reference(h: np.ndarray,
                                  J: np.ndarray,
                                  num_replicas: int,
                                  num_sweeps: int,
                                  beta_schedule: np.ndarray,
                                  seed: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
    """
    Pure-NumPy reference implementation of Population Annealing.
    
    Parameters
    ----------
    h : array-like
        Local field terms of the Ising model (N elements).
    J : array-like
        Coupling matrix of the Ising model (N x N matrix).
    num_replicas : int
        Number of replicas in the population.
    num_sweeps : int
        Number of Metropolis sweeps per temperature step.
    beta_schedule : array-like
        Strictly increasing beta schedule.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        - best_state: The best spin configuration found.
        - best_energy: Energy of the best state.
        - final_free_energy: Estimated free energy at final temperature.
    """
    # Convert to numpy arrays
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    beta_schedule = np.asarray(beta_schedule, dtype=np.float64)
    
    N = h.shape[0]
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Initialize replicas with random spin states (-1 or +1)
    replicas = rng.choice([-1, 1], size=(num_replicas, N))
    
    # Track the best state found
    best_state = None
    best_energy = np.inf
    
    # Initialize free energy accumulator
    ln_Q_sum = 0.0
    
    # Main annealing loop
    for i in range(len(beta_schedule)):
        beta = beta_schedule[i]
        
        # Calculate energies for all replicas
        energies = _calculate_ising_energies(replicas, h, J)
        
        # Update best state if needed
        min_energy_idx = np.argmin(energies)
        if energies[min_energy_idx] < best_energy:
            best_energy = energies[min_energy_idx]
            best_state = replicas[min_energy_idx].copy()
        
        # If this is the last step, break as we don't need to resample
        if i == len(beta_schedule) - 1:
            break
            
        # Resampling step
        next_beta = beta_schedule[i + 1]
        delta_beta = next_beta - beta
        
        # Calculate relative energies to prevent overflow
        E_min = np.min(energies)
        relative_energies = energies - E_min
        
        # Calculate weights
        weights = np.exp(-delta_beta * relative_energies)
        
        # Calculate normalization factor Q
        Q = np.exp(-delta_beta * E_min) * np.mean(weights)
        ln_Q = -delta_beta * E_min + np.log(np.mean(weights))
        ln_Q_sum += ln_Q
        
        # Calculate expected copies
        expected_copies = num_replicas * weights / np.sum(weights)
        
        # Residual resampling
        uniform_draws = rng.uniform(0, 1, num_replicas)
        allocated_copies = np.floor(expected_copies).astype(int)
        residual_probs = expected_copies - allocated_copies
        residual_draws = (uniform_draws < residual_probs).astype(int)
        copies = allocated_copies + residual_draws
        
        # Create new population
        new_replicas = []
        for j in range(num_replicas):
            for _ in range(copies[j]):
                new_replicas.append(replicas[j].copy())
        
        # If population collapses, raise error
        if len(new_replicas) == 0:
            raise RuntimeError("Population collapse")
        
        # Adjust to exact population size
        if len(new_replicas) > num_replicas:
            # Randomly remove excess replicas
            indices = rng.choice(len(new_replicas), num_replicas, replace=False)
            new_replicas = [new_replicas[idx] for idx in indices]
        elif len(new_replicas) < num_replicas:
            # Randomly duplicate replicas to reach target size
            indices = rng.choice(len(new_replicas), num_replicas - len(new_replicas))
            for idx in indices:
                new_replicas.append(new_replicas[idx].copy())
        
        replicas = np.array(new_replicas)
        
        # Apply Metropolis sweeps
        for _ in range(num_sweeps):
            # Shuffle spin order for each replica
            for replica_idx in range(num_replicas):
                spin_order = rng.permutation(N)
                for spin_idx in spin_order:
                    # Calculate energy difference
                    delta_E = _calculate_delta_energy(replicas[replica_idx], h, J, spin_idx)
                    
                    # Metropolis acceptance
                    if delta_E <= 0 or rng.uniform() < np.exp(-next_beta * delta_E):
                        replicas[replica_idx, spin_idx] *= -1  # Flip spin
    
    # Calculate final free energy
    N_sites = N
    ln_Z_0 = N_sites * np.log(2)  # ln(2^N) for initial uniform distribution
    negative_beta_F = ln_Z_0 + ln_Q_sum
    final_free_energy = -negative_beta_F / beta_schedule[-1]
    
    # Ensure best_state is always a numpy array
    if best_state is None:
        # This should not happen with valid inputs, but for safety we initialize
        best_state = np.ones(N, dtype=np.int8)
        best_energy = _calculate_ising_energy(best_state, h, J)
    
    return best_state, best_energy, final_free_energy


def _calculate_ising_energies(replicas: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Calculate Ising energies for a population of replicas."""
    # Vectorized energy calculation
    # Linear term: sum h_i * s_i for each replica
    linear_energy = np.dot(replicas, h)
    
    # Quadratic term: sum J_ij * s_i * s_j for each replica
    # Using Einstein summation for efficient computation
    # Apply 0.5 factor to account for symmetric matrix double-counting
    quadratic_energy = 0.5 * np.einsum('ij,ki,kj->k', J, replicas, replicas)
    
    return linear_energy + quadratic_energy


def _calculate_ising_energy(state: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """Calculate Ising energy for a single state."""
    # Linear term: sum h_i * s_i
    linear_energy = np.dot(state, h)
    
    # Quadratic term: sum J_ij * s_i * s_j
    # Apply 0.5 factor to account for symmetric matrix double-counting
    quadratic_energy = 0.5 * np.einsum('ij,i,j', J, state, state)
    
    return linear_energy + quadratic_energy


def _calculate_delta_energy(state: np.ndarray, h: np.ndarray, J: np.ndarray, spin_idx: int) -> float:
    """Calculate the energy difference of flipping a single spin."""
    # ΔE = E_new - E_old = -2 * s_i * (h_i + sum_j J_ij * s_j)
    # where s_i is the current spin value
    current_spin = state[spin_idx]
    field_contribution = h[spin_idx]
    coupling_contribution = np.dot(J[spin_idx], state)
    delta_E = -2 * current_spin * (field_contribution + coupling_contribution)
    return delta_E