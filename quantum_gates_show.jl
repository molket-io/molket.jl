module quantum_gates_show
#  © MolKet 2023, MIT License.
# This module is part of MolKet.jl package.
## For more information, please visit out website:
# www.molket.io

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load custom quantum gates module")


#In Qiskit’s convention, higher qubit indices are more significant (little endian convention). 
#In many textbooks, controlled gates are presented with the assumption of more significant 
#qubits as control, which in our case would be q_1. 
# check the qiskit resource for more information
# https://qiskit.org/documentation/stubs/qiskit.circuit.library.CU3Gate.html

using LinearAlgebra


# import conventions
include("conventions.jl")
  using .conventions: big_endian
# import quantum gates
 include("quantum_gates.jl")
using ..quantum_gates: Qgate
# import the sorting function
include("lib_useful/custom_functions.jl")
using ..custom_functions: MK_sortrows

# Define the default error tolerance for checking the norm of the vector
const err_tol = 1e-15

# export the functions
export H, X, CX

function X(qc, qubit, endian=big_endian)
    Xgate = Qgate.X
    if qc.n_qubits == 2
        Xgate = Qgate_T2D(Xgate, qubit, qc.n_qubits, endian)
    end
    
    show(stdout, "text/plain", Xgate)
        
    op(qc,Xgate)
    
    return qc
end # function X

function CX(qc, q_control::Int64, q_target::Int64, endian::Bool=conventions.big_endian)
    CXgate = Qgate.CX(q_control, q_target, endian) 
    if qc.n_qubits == 2
        # Call CX gate with CX_gate(q_control::Int64,q_target::Int64, big_endian::Bool=conventions.big_endian
        #CXgate = Qgate_T2D(CXgate, q_control, q_target, endian)
    end
    
    show(stdout, "text/plain", CXgate)
        
    op(qc,CXgate)
    
    return qc
end # function CX

function H(qc, qubit, endian=big_endian)
    
    Hgate = Qgate.H
    
    if qc.n_qubits == 2
        Hgate = Qgate_T2D(Hgate, qubit, qc.n_qubits, endian)
    end
    
    show(stdout, "text/plain", Hgate)
        
    op(qc,Hgate)
    
    return qc
end # function H


end # module