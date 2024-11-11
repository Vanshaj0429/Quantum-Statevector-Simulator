# Initialize the Quantum State Tensor

def initialize_state_tensor(n_qubits):
    # Start with single |0> state tensor
    state = np.array([1, 0], dtype=complex)
    state = state.reshape([2] + [1]*(n_qubits - 1))
    for _ in range(n_qubits - 1):
        state = np.tensordot(state, [1, 0], axes=0)
    state = state.reshape([2]*n_qubits)
    return state

# Apply Gates Using Tensor Multiplication

# Single-Qubit Gate Application
def apply_single_qubit_gate_tensor(state, gate, target_qubit):
    axes = ([target_qubit], [0])
    state = np.tensordot(state, gate, axes=axes)
    # Move the new axis to the target qubit position
    order = list(range(state.ndim))
    order.insert(target_qubit, order.pop())
    state = state.transpose(order)
    return state

# Two-Qubit Gate Application
# For a two-qubit gate like CNOT, we need to reshape and apply it to the correct axes.

def apply_two_qubit_gate_tensor(state, gate, qubit1, qubit2):
    # Ensure qubit1 < qubit2
    if qubit1 > qubit2:
        qubit1, qubit2 = qubit2, qubit1
    axes = ([qubit1, qubit2], [0, 1])
    state = np.tensordot(state, gate, axes=axes)
    # Move the new axes to the qubit positions
    order = list(range(state.ndim))
    new_axes = [state.ndim - 2, state.ndim - 1]
    for idx, qubit in zip(new_axes, [qubit1, qubit2]):
        order.insert(qubit, order.pop(idx))
    state = state.transpose(order)
    return state

# Simulate the Circuit

def simulate_circuit_tensor(n_qubits):
    state = initialize_state_tensor(n_qubits)

    # Define gates
    # Reshape gates for tensor operations
    H_gate = H.reshape(2, 2)
    X_gate = X.reshape(2, 2)
    CNOT_gate = get_two_qubit_cnot_tensor()

    # Apply gates
    state = apply_single_qubit_gate_tensor(state, H_gate, target_qubit=0)
    state = apply_single_qubit_gate_tensor(state, X_gate, target_qubit=1)
    state = apply_two_qubit_gate_tensor(state, CNOT_gate, qubit1=0, qubit2=1)

    return state


# Measure Runtime
import time
qubit_counts = range(2, 14)  # Adjust the range as needed
runtimes = []

for n in qubit_counts:
    start_time = time.time()
    simulate_circuit_tensor(n)
    end_time = time.time()
    runtimes.append(end_time - start_time)

# Plot the runtime
plt.figure(figsize=(8, 5))
plt.plot(qubit_counts, runtimes, marker='o', color='green')
plt.xlabel('Number of Qubits')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs Number of Qubits (Tensor Multiplication)')
plt.grid(True)
plt.show()



