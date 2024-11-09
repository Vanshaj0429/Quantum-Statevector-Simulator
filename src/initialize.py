# src/initialize.py

import numpy as np

def initialize_state(n_qubits):
    """
    Initialize the quantum state |0>^n.

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        np.ndarray: State vector of size 2^n_qubits.
    """
    state = np.array([1, 0], dtype=complex)
    for _ in range(n_qubits - 1):
        state = np.kron(state, [1, 0])
    return state

def initialize_state_tensor(n_qubits):
    """
    Initialize the quantum state |0>^n as a tensor.

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        np.ndarray: State tensor of shape (2,)*n_qubits.
    """
    state = np.array([1, 0], dtype=complex)
    for _ in range(n_qubits - 1):
        state = np.kron(state, [1, 0])
    state = state.reshape([2]*n_qubits)
    return state
