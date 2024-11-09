# src/simulator.py

import numpy as np
from .initialize import initialize_state, initialize_state_tensor
from .gates import I, X, H, cnot_gate, get_two_qubit_cnot_tensor

def apply_gate(state, gate_matrix):
    """
    Apply a gate to the state vector via matrix multiplication.

    Args:
        state (np.ndarray): Current state vector.
        gate_matrix (np.ndarray): Gate matrix to apply.

    Returns:
        np.ndarray: Updated state vector.
    """
    return gate_matrix @ state

def simulate_circuit_matrix(n_qubits):
    """
    Simulate a quantum circuit using matrix multiplication.

    Example Circuit:
        - Apply H to qubit 0
        - Apply X to qubit 1
        - Apply CNOT with control qubit 0 and target qubit 1

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        np.ndarray: Final state vector.
    """
    state = initialize_state(n_qubits)

    # Build H_full: H on qubit 0, I on others
    H_full = 1
    for qubit in range(n_qubits):
        H_full = np.kron(H_full, H if qubit == 0 else I)

    # Build X_full: X on qubit 1, I on others
    X_full = 1
    for qubit in range(n_qubits):
        X_full = np.kron(X_full, X if qubit == 1 else I)

    # Build CNOT_full
    CNOT_full = cnot_gate(control=0, target=1, n_qubits=n_qubits)

    # Apply gates
    state = apply_gate(state, H_full)
    state = apply_gate(state, X_full)
    state = apply_gate(CNOT_full, state)

    return state

def apply_single_qubit_gate_tensor(state, gate, target_qubit):
    """
    Apply a single-qubit gate to the state tensor.

    Args:
        state (np.ndarray): Current state tensor.
        gate (np.ndarray): Single-qubit gate matrix.
        target_qubit (int): Qubit index to apply the gate.

    Returns:
        np.ndarray: Updated state tensor.
    """
    axes = ([target_qubit], [0])
    state = np.tensordot(state, gate, axes=axes)
    # Move the new axis to the target qubit position
    order = list(range(state.ndim))
    order.insert(target_qubit, order.pop())
    state = state.transpose(order)
    return state

def apply_two_qubit_gate_tensor(state, gate, qubit1, qubit2):
    """
    Apply a two-qubit gate to the state tensor.

    Args:
        state (np.ndarray): Current state tensor.
        gate (np.ndarray): Two-qubit gate tensor.
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.

    Returns:
        np.ndarray: Updated state tensor.
    """
    if qubit1 > qubit2:
        qubit1, qubit2 = qubit2, qubit1
    axes = ([qubit1, qubit2], [0, 1])
    state = np.tensordot(state, gate, axes=axes)
    # Reorder axes to maintain qubit ordering
    order = list(range(state.ndim))
    # New axes from tensordot are at the end
    new_axes = [state.ndim - 2, state.ndim - 1]
    for idx, qubit in zip(new_axes, [qubit1, qubit2]):
        order.insert(qubit, order.pop(idx))
    state = state.transpose(order)
    return state

def simulate_circuit_tensor(n_qubits):
    """
    Simulate a quantum circuit using tensor multiplication.

    Example Circuit:
        - Apply H to qubit 0
        - Apply X to qubit 1
        - Apply CNOT with control qubit 0 and target qubit 1

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        np.ndarray: Final state tensor.
    """
    state = initialize_state_tensor(n_qubits)

    # Define gates
    H_gate = H.reshape(2, 2)
    X_gate = X.reshape(2, 2)
    CNOT_gate = get_two_qubit_cnot_tensor()

    # Apply gates
    state = apply_single_qubit_gate_tensor(state, H_gate, target_qubit=0)
    state = apply_single_qubit_gate_tensor(state, X_gate, target_qubit=1)
    state = apply_two_qubit_gate_tensor(state, CNOT_gate, qubit1=0, qubit2=1)

    return state
