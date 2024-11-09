# src/utils.py

import numpy as np
from .simulator import apply_single_qubit_gate_tensor

def sample_measurements(state, num_samples=1000):
    """
    Sample measurement outcomes from the statevector or state tensor.

    Args:
        state (np.ndarray): State vector or state tensor.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of sampled outcome indices.
    """
    state_vector = state.flatten()
    probabilities = np.abs(state_vector) ** 2
    probabilities /= probabilities.sum()  # Normalize to avoid numerical errors
    outcomes = np.arange(len(state_vector))
    samples = np.random.choice(outcomes, size=num_samples, p=probabilities)
    return samples

def compute_expectation_value(state, operator, target_qubit):
    """
    Compute the expectation value <Psi|Op|Psi> for a single-qubit operator.

    Args:
        state (np.ndarray): Current state tensor.
        operator (np.ndarray): Single-qubit operator matrix.
        target_qubit (int): Qubit index the operator acts on.

    Returns:
        float: Expectation value.
    """
    # Apply operator to the target qubit
    state_op = apply_single_qubit_gate_tensor(state, operator.reshape(2, 2), target_qubit)
    # Compute the inner product <Psi|Op|Psi>
    state_conj = np.conj(state)
    expectation = np.tensordot(state_conj, state_op, axes=state.ndim)
    return expectation.real
