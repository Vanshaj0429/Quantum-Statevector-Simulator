# src/gates.py

import numpy as np

# Single-qubit gates
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]], dtype=complex)

# Two-qubit gate: CNOT
def cnot_gate(control, target, n_qubits):
    """
    Create a CNOT gate matrix for n_qubits.

    Args:
        control (int): Control qubit index.
        target (int): Target qubit index.
        n_qubits (int): Total number of qubits.

    Returns:
        np.ndarray: CNOT gate matrix of size 2^n_qubits x 2^n_qubits.
    """
    size = 2 ** n_qubits
    cnot = np.zeros((size, size), dtype=complex)
    for i in range(size):
        # Convert index to binary representation
        binary = format(i, '0' + str(n_qubits) + 'b')
        binary_list = list(binary)
        if binary_list[control] == '1':
            # Flip the target qubit
            binary_list[target] = '1' if binary_list[target] == '0' else '0'
            j = int(''.join(binary_list), 2)
            cnot[i, j] = 1
        else:
            cnot[i, i] = 1
    return cnot

def get_two_qubit_cnot_tensor():
    """
    Create a tensor representation of the CNOT gate.

    Returns:
        np.ndarray: CNOT gate tensor of shape (2, 2, 2, 2).
    """
    CNOT = np.zeros((2, 2, 2, 2), dtype=complex)
    CNOT[0, 0, 0, 0] = 1
    CNOT[0, 1, 0, 1] = 1
    CNOT[1, 0, 1, 1] = 1
    CNOT[1, 1, 1, 0] = 1
    return CNOT
