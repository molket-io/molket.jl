module quantum_circuit
# © MolKet 2023, MIT License.
# This module is part of MolKet.jl package.
## For more information, please visit out website:
# www.molket.io

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load quantum gates constructor")
#println("Using constants from file: ", module_file(constants))


# qiskit order
# https://qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html
#In Qiskit’s convention, higher qubit indices are more significant (little endian convention). 
#In many textbooks, controlled gates are presented with the assumption of more significant 
#qubits as control, which in our case would be q_1. 
# check the qiskit resource for more information
# https://qiskit.org/documentation/stubs/qiskit.circuit.library.CU3Gate.html

# import quantum gates
 include("quantum_gates.jl")
using ..quantum_gates: Qgate

# export functions
export qc_initialize 

# Initialize the order of the qubits in the quantum register
function qubit_order(q_order::String)
    # q_order::String: order of the qubits in the quantum register
    # return: order of the qubits in the quantum register

    if q_order == "big-endian"
        return q_order
    elseif q_order == "little-endian"
        return q_order
    else
        error("The order of the qubits in the quantum register is not defined")
    end
end # end qubit_order


# Initizalize the quantum registers of n qubits
function init_register(n::Int64; q_order::String="big-endian")
    # Initialize the quantum register
    # n::Int64: number of qubits
    # q_order::String: order of the qubits in the quantum register
    # return: quantum register of n qubits

    n_bas = 2 # number of basis states
    n_qubits = n # number of qubits
    n_dim = 2^n # dimensions of the quantum register/Hilbert space
    register = zeros(2^n) # initialize the quantum register
    q_states = zeros(2^n, n) # initialize the quantum states
    q_tab = [0;1] # initialize the quantum table
    # q_order == "big-endian"
    # q_order == "little-endian" 
    n_count = n_bas
    for i = 2:n_qubits
        q_tab = [zeros(n_count,1) q_tab
                ones(n_count,1) q_tab]
        n_count = n_count*2
    end
    q_states = q_tab
    return register, q_states, n_bas, n_dim, q_order 
end # end init_register

Base.@kwdef mutable struct qc_initstruct
    # Initialize the quantum register
    n::Int64
    q_order::String
    n_bas::Int64
    n_dim::Int64
    register::Array{Float64,1}
    q_states::Array{Float64,2}
end # end qc_initialize

function qc_initialize(n::Int64; q_order::String="big-endian")
    # Initialize the quantum register
    # n::Int64: number of qubits
    # q_order::String: order of the qubits in the quantum register
    # return: quantum register of n qubits

    n_bas = 2 # number of basis states
    n_qubits = n # number of qubits
    n_dim = 2^n # dimensions of the quantum register/Hilbert space
    register = zeros(2^n) # initialize the quantum register
    q_states = zeros(2^n, n) # initialize the quantum states
    q_tab = [0;1] # initialize the quantum table
    # q_order == "big-endian"
    # q_order == "little-endian"
    n_count = n_bas
    for i = 2:n_qubits
        q_tab = [zeros(n_count,1) q_tab
                ones(n_count,1) q_tab]
        n_count = n_count*2
    end
    q_states = q_tab
    return qc_initstruct(n, q_order, n_bas, n_dim, register, q_states)
end # end qc_initialize


end # end module
####### end quantum_circuit.jl #######   