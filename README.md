# Quantum Statevector Simulator

A statevector simulator for quantum circuits implemented from scratch using Python and NumPy. This simulator demonstrates both naive matrix multiplication and advanced tensor multiplication methods for simulating quantum circuits.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

Quantum computing leverages the principles of quantum mechanics to perform computations. Unlike classical bits, which can be either 0 or 1, quantum bits (qubits) can exist in superpositions of states. This simulator allows you to:

- Initialize quantum states.
- Apply quantum gates (X, H, CNOT).
- Simulate quantum circuits using matrix and tensor multiplication.
- Sample measurement outcomes.
- Compute expectation values of observables.

## Features

- **Naive Simulation:** Uses matrix multiplication to apply gates.
- **Advanced Simulation:** Utilizes tensor contraction for efficient gate application.
- **Measurement Sampling:** Simulate quantum measurements based on state probabilities.
- **Expectation Values:** Compute exact expectation values of quantum operators.
- **Performance Analysis:** Compare runtime performance between simulation methods.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/Quantum-Statevector-Simulator.git
   cd Quantum-Statevector-Simulator


2. **Install Dependencies**

    pip install -r requirements.txt


## Usage 

**For best experience use the .pynb file in src**

**Another way reviewing code is given below**

**Simulating a Quantum Circuit with Matrix Multiplication**

    from src.Naive-Matrix import simulate_circuit_matrix

    n_qubits = 3
    final_state = simulate_circuit_matrix(n_qubits)
    print("Final State Vector:")
    print(final_state)

**Simulating a Quantum Circuit with Tensor Multiplication**

    from src.Advanced-Tensor import simulate_circuit_tensor

    n_qubits = 3
    final_state = simulate_circuit_tensor(n_qubits)
    print("Final State Tensor:")
    print(final_state)

**Sampling Measurements**

    from src.Advanced-Tensor import simulate_circuit_tensor
    from src.Bonus import sample_measurements

    n_qubits = 3
    state = simulate_circuit_tensor(n_qubits)
    samples = sample_measurements(state, num_samples=1000)
    bitstrings = [format(sample, '03b') for sample in samples]
    print(f"Sampled bitstrings: {bitstrings[:10]}")

**Computing Expectation Values**

    from src.Advanced-Tensor import simulate_circuit_tensor
    from src.Bonus import compute_expectation_value
    import numpy as np

    n_qubits = 3
    state = simulate_circuit_tensor(n_qubits)
    Z = np.array([[1, 0],
                [0, -1]], dtype=complex)
    expectation = compute_expectation_value(state, Z, target_qubit=0)
    print(f"Expectation value of Z on qubit 0: {expectation}")


## Notebooks

Refer to the notebooks directory for detailed examples and explanations.

## Performance

The simulator includes runtime analysis comparing naive matrix multiplication and tensor multiplication methods. See the generated plots in the plots directory. Also if you go through the notebook then can check the plots there as well.





