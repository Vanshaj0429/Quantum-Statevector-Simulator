import numpy as np
import time
import matplotlib.pyplot as plt

def initialize_state(n_qubits):
    # Start with single |0> state
    state = np.array([1, 0], dtype=complex)
    for _ in range(n_qubits - 1):
        state = np.kron(state, [1, 0])
    return state

# Single-qubit gates
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]], dtype=complex)

# Two-qubit gate
def cnot_gate(control, target, n_qubits):
    # Create a full 2^n x 2^n matrix for the CNOT gate
    size = 2 ** n_qubits
    cnot = np.zeros((size, size), dtype=complex)
    for i in range(size):
        binary = format(i, '0' + str(n_qubits) + 'b')
        if binary[control] == '1':
            flipped = list(binary)
            flipped[target] = '1' if binary[target] == '0' else '0'
            j = int(''.join(flipped), 2)
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


def apply_gate(state, gate_matrix):
    return gate_matrix @ state

def simulate_circuit(n_qubits):
    state = initialize_state(n_qubits)
    
    # Example circuit: Apply H to qubit 0, X to qubit 1, then CNOT with control qubit 0 and target qubit 1
    # Build the full gate matrices
    H_full = 1
    for qubit in range(n_qubits):
        H_full = np.kron(H_full, H if qubit == 0 else I)
    X_full = 1
    for qubit in range(n_qubits):
        X_full = np.kron(X_full, X if qubit == 1 else I)
    CNOT_full = cnot_gate(control=0, target=1, n_qubits=n_qubits)
    
    # Apply gates sequentially
    state = apply_gate(state, H_full)
    state = apply_gate(state, X_full)
    state = apply_gate(state, CNOT_full)
    
    return state

#Runtime Analysis
qubit_counts = range(2, 12)  # Adjust the range as needed
runtimes = []

for n in qubit_counts:
    start_time = time.time()
    simulate_circuit(n)
    end_time = time.time()
    runtimes.append(end_time - start_time)

# Plot the runtime
plt.figure(figsize=(8, 5))
plt.plot(qubit_counts, runtimes, marker='o')
plt.xlabel('Number of Qubits')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs Number of Qubits (Naive Matrix Multiplication)')
plt.grid(True)
plt.show()



