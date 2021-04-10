from typing import List, Tuple

import numpy as np
import cirq
import qiskit


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    if len(target_qubits) > 4:
        return NotImplemented, []
    
    # Converting the Unitary to Operations using Qiskit
    qc = qiskit.QuantumCircuit(len(target_qubits))
    qc.unitary(matrix, list(range(len(target_qubits))))
    qc = qiskit.transpile(qc, basis_gates=['cx', 'u3'])
    # Converting Qiskit to Cirq
    from cirq.contrib import qasm_import
    qasm = qc.qasm()
    qx = cirq.Circuit(qasm_import.circuit_from_qasm(qasm))
    # Compiling down to the Sycamore hardware
    convertor = cirq.google.ConvertToSycamoreGates()
    qz = convertor.convert(qx)
    # Running an optimization pass
    qy = cirq.Circuit(qz)
    qy = cirq.google.optimized_for_sycamore(qy)
    qy = list(qy.all_operations())
    
    ans = qz if len(qy) > len(qz) else qy  # Check if optimizations are doing better
    return ans, []
