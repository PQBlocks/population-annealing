import numpy as np
import numba
from numba import uint32
from numba.experimental import jitclass
from typing import Optional, Tuple, List


# PRNG implementation for Numba compatibility
# Using a simple xorshift32 generator to avoid low-order bit periodicity
spec = [
    ('state', uint32),
]

@jitclass(spec)
class Xorshift32:
    def __init__(self, seed):
        self.state = seed if seed != 0 else uint32(1)  # Ensure non-zero state
    
    def next(self) -> uint32:
        # Xorshift32 algorithm
        self.state ^= (self.state << uint32(13)) & uint32(0xFFFFFFFF)
        self.state ^= (self.state >> uint32(17)) & uint32(0xFFFFFFFF)
        self.state ^= (self.state << uint32(5)) & uint32(0xFFFFFFFF)
        return self.state
    
    def random(self) -> float:
        # Generate a float in [0, 1)
        return float((self.next() >> uint32(8)) / uint32(16777216))  # 2^24


# Numba-compatible function to calculate Ising energy for a single state
@numba.jit(nopython=True, fastmath=True)
def _calculate_ising_energy_numba(state: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """Calculate Ising energy for a single state."""
    # Linear term: sum h_i * s_i
    linear_energy = 0.0
    for i in range(state.shape[0]):
        linear_energy += h[i] * state[i]
    
    # Quadratic term: sum J_ij * s_i * s_j
    # Apply 0.5 factor to account for symmetric matrix double-counting
    quadratic_energy = 0.0
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if i != j:  # Only count off-diagonal terms
                quadratic_energy += 0.5 * J[i, j] * state[i] * state[j]
    
    return linear_energy + quadratic_energy


# Numba-compatible function to calculate energy difference of flipping a single spin
@numba.jit(nopython=True, fastmath=True)
def _calculate_delta_energy_numba(state: np.ndarray, h: np.ndarray, J: np.ndarray, spin_idx: int) -> float:
    """Calculate the energy difference of flipping a single spin."""
    # ΔE = E_new - E_old = -2 * s_i * (h_i + sum_j J_ij * s_j)
    # where s_i is the current spin value
    current_spin = state[spin_idx]
    field_contribution = h[spin_idx]
    coupling_contribution = 0.0
    for j in range(state.shape[0]):
        if j != spin_idx:  # Only count off-diagonal terms
            coupling_contribution += J[spin_idx, j] * state[j]
    delta_E = -2 * current_spin * (field_contribution + coupling_contribution)
    return delta_E


# Numba-compatible function to calculate Ising energies for a population of replicas
@numba.jit(nopython=True, fastmath=True)
def _calculate_ising_energies_numba(replicas: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Calculate Ising energies for a population of replicas."""
    num_replicas = replicas.shape[0]
    energies = np.zeros(num_replicas, dtype=np.float64)
    
    # Calculate energy for each replica
    for i in range(num_replicas):
        energies[i] = _calculate_ising_energy_numba(replicas[i], h, J)
    
    return energies


# Main Numba JIT-compiled kernel for Population Annealing
@numba.jit(nopython=True, fastmath=True)
def _pa_kernel(h: np.ndarray,
               J: np.ndarray,
               num_replicas: int,
               num_sweeps: int,
               beta_schedule: np.ndarray,
               seed: int) -> Tuple[np.ndarray, float, float]:
    """
    Numba JIT-compiled kernel for Population Annealing.
    
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
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        - best_state: The best spin configuration found.
        - best_energy: Energy of the best state.
        - final_free_energy: Estimated free energy at final temperature.
    """
    N = h.shape[0]
    
    # Initialize PRNG
    rng = Xorshift32(np.uint32(seed))
    
    # Initialize replicas with random spin states (-1 or +1)
    replicas = np.zeros((num_replicas, N), dtype=np.int8)
    for i in range(num_replicas):
        for j in range(N):
            # Generate random bit and map to -1 or +1
            replicas[i, j] = 1 if rng.random() > 0.5 else -1
    
    # Track the best state found
    best_state = np.zeros(N, dtype=np.int8)
    best_energy = np.inf
    
    # Initialize free energy accumulator
    ln_Q_sum = 0.0
    
    # Main annealing loop
    for i in range(len(beta_schedule)):
        beta = beta_schedule[i]
        
        # Calculate energies for all replicas
        energies = _calculate_ising_energies_numba(replicas, h, J)
        
        # Update best state if needed
        for j in range(num_replicas):
            if energies[j] < best_energy:
                best_energy = energies[j]
                for k in range(N):
                    best_state[k] = replicas[j, k]
        
        # If this is the last step, break as we don't need to resample
        if i == len(beta_schedule) - 1:
            break
            
        # Resampling step
        next_beta = beta_schedule[i + 1]
        delta_beta = next_beta - beta
        
        # Calculate relative energies to prevent overflow
        E_min = energies[0]
        for j in range(1, num_replicas):
            if energies[j] < E_min:
                E_min = energies[j]
        
        # Calculate relative energies
        relative_energies = np.zeros(num_replicas, dtype=np.float64)
        for j in range(num_replicas):
            relative_energies[j] = energies[j] - E_min
        
        # Calculate weights
        weights = np.zeros(num_replicas, dtype=np.float64)
        for j in range(num_replicas):
            # Clamp exponent to prevent overflow
            exponent = -delta_beta * relative_energies[j]
            if exponent < -709:  # Prevent underflow
                weights[j] = 0.0
            else:
                weights[j] = np.exp(exponent)
        
        # Calculate normalization factor Q
        sum_weights = 0.0
        for j in range(num_replicas):
            sum_weights += weights[j]
        mean_weights = sum_weights / num_replicas
        
        ln_Q = -delta_beta * E_min + np.log(mean_weights)
        ln_Q_sum += ln_Q
        
        # Calculate expected copies
        expected_copies = np.zeros(num_replicas, dtype=np.float64)
        for j in range(num_replicas):
            expected_copies[j] = num_replicas * weights[j] / sum_weights
        
        # Residual resampling
        uniform_draws = np.zeros(num_replicas, dtype=np.float64)
        for j in range(num_replicas):
            uniform_draws[j] = rng.random()
        
        allocated_copies = np.zeros(num_replicas, dtype=np.int32)
        residual_probs = np.zeros(num_replicas, dtype=np.float64)
        residual_draws = np.zeros(num_replicas, dtype=np.int32)
        copies = np.zeros(num_replicas, dtype=np.int32)
        
        for j in range(num_replicas):
            allocated_copies[j] = int(expected_copies[j])
            residual_probs[j] = expected_copies[j] - allocated_copies[j]
            residual_draws[j] = 1 if uniform_draws[j] < residual_probs[j] else 0
            copies[j] = allocated_copies[j] + residual_draws[j]
        
        # Create new population
        # Pre-allocate array for new replicas
        total_copies = 0
        for j in range(num_replicas):
            total_copies += copies[j]
        
        # If population collapses, raise error
        if total_copies == 0:
            # We can't raise an exception in numba, so we'll return a special value
            # In the wrapper, we'll check for this condition
            return np.zeros(N, dtype=np.int8), np.inf, np.inf
        
        # Pre-allocate array for new replicas
        new_replicas = np.zeros((total_copies, N), dtype=np.int8)
        new_replicas_idx = 0
        
        for j in range(num_replicas):
            for _ in range(copies[j]):
                if new_replicas_idx < total_copies:
                    for k in range(N):
                        new_replicas[new_replicas_idx, k] = replicas[j, k]
                    new_replicas_idx += 1
        
        # Adjust to exact population size if needed
        if new_replicas_idx > num_replicas:
            # Randomly remove excess replicas
            # For simplicity in numba, we'll just take the first num_replicas
            # In a full implementation, we would do proper random selection
            temp_replicas = np.zeros((num_replicas, N), dtype=np.int8)
            # Use a simple modulo approach to select indices
            for j in range(num_replicas):
                src_idx = j % new_replicas_idx
                for k in range(N):
                    temp_replicas[j, k] = new_replicas[src_idx, k]
            new_replicas = temp_replicas
        elif new_replicas_idx < num_replicas:
            # Randomly duplicate replicas to reach target size
            # For simplicity in numba, we'll duplicate the first few replicas
            # In a full implementation, we would do proper random selection
            # Pre-allocate the additional replicas array
            additional_replicas = np.zeros((num_replicas - new_replicas_idx, N), dtype=np.int8)
            for j in range(num_replicas - new_replicas_idx):
                src_idx = j % max(1, new_replicas_idx)
                for k in range(N):
                    additional_replicas[j, k] = new_replicas[src_idx, k]
                
            # Concatenate the arrays
            # Create a new array with the correct size
            final_replicas = np.zeros((num_replicas, N), dtype=np.int8)
            # Copy existing replicas
            for j in range(new_replicas_idx):
                for k in range(N):
                    final_replicas[j, k] = new_replicas[j, k]
            # Copy additional replicas
            for j in range(num_replicas - new_replicas_idx):
                for k in range(N):
                    final_replicas[new_replicas_idx + j, k] = additional_replicas[j, k]
            new_replicas = final_replicas
        
        replicas = new_replicas
        
        # Apply Metropolis sweeps
        for _ in range(num_sweeps):
            # For each replica
            for replica_idx in range(num_replicas):
                # Shuffle spin order - in numba we'll just iterate in random order
                # by generating a random permutation
                spin_order = np.zeros(N, dtype=np.int32)
                for j in range(N):
                    spin_order[j] = j
                
                # Simple shuffle algorithm
                for j in range(N):
                    k = int(rng.random() * N)
                    if k < 0:
                        k = 0
                    if k >= N:
                        k = N - 1
                    temp = spin_order[j]
                    spin_order[j] = spin_order[k]
                    spin_order[k] = temp
                
                # Perform sweep
                for spin_pos in range(N):
                    spin_idx = spin_order[spin_pos]
                    # Calculate energy difference
                    delta_E = _calculate_delta_energy_numba(replicas[replica_idx], h, J, spin_idx)
                    
                    # Metropolis acceptance
                    if delta_E <= 0 or rng.random() < np.exp(-next_beta * delta_E):
                        replicas[replica_idx, spin_idx] *= -1  # Flip spin
    
    # Calculate final free energy
    N_sites = N
    ln_Z_0 = N_sites * np.log(2)  # ln(2^N) for initial uniform distribution
    negative_beta_F = ln_Z_0 + ln_Q_sum
    final_free_energy = -negative_beta_F / beta_schedule[-1]
    
    return best_state, best_energy, final_free_energy


def population_annealing_numba(h: np.ndarray,
                               J: np.ndarray,
                               num_replicas: int,
                               num_sweeps: int,
                               beta_schedule: np.ndarray,
                               seed: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
    """
    Numba JIT-compiled implementation of Population Annealing.
    
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
    
    # Use a default seed if none provided
    if seed is None:
        seed = 42
    
    # Run the JIT-compiled kernel
    best_state, best_energy, final_free_energy = _pa_kernel(
        h, J, num_replicas, num_sweeps, beta_schedule, seed
    )
    
    # Check for population collapse
    if np.isinf(final_free_energy):
        raise RuntimeError("Population collapse")
    
    return best_state, best_energy, final_free_energy