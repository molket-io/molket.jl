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
#https://qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html#Basis-vector-ordering-in-Qiskit

#In Qiskit’s convention, higher qubit indices are more significant (little endian convention). 
#In many textbooks, controlled gates are presented with the assumption of more significant 
#qubits as control, which in our case would be q_1. 
# check the qiskit resource for more information
# https://qiskit.org/documentation/stubs/qiskit.circuit.library.CU3Gate.html

using LinearAlgebra


# import quantum gates
 include("quantum_gates.jl")
using ..quantum_gates: Qgate

# export functions
export qc_initialize 

# Define the default error tolerance for checking the norm of the vector
const err_tol = 1e-16



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

# function to print a quantum register and quantum states in latex format 
function print_statevector(
    state_vector::Union{Vector{Float64},Vector{Int64}, Vector{ComplexF64}},
    q_states::Array{Int64,2})
    # print a quantum register and quantum states in latex format 
    # qc::qc_initstruct: quantum register
    # return: print a quantum register and quantum states in latex format 
    
    #### ************* not tested 

    # combine statevector and quantum states in one table 
    # state_vector::Vector{ComplexF64}: statevector of the quantum register
    # q_states::Array{Int64,2}: quantum states of the quantum register
    # return: print a quantum register and quantum states in latex format
    qtab = [q_states convert(Array{Int64}, state_vector)]
    println("The quantum register and quantum states in latex format are: ")
    show(stdout, "text/plain", qtab)
end # end print_latex

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

function qc_initialize(n::Int64, 
    c_sv::Union{Vector{Float64}, Vector{Int64}, Vector{ComplexF64}} = nothing, 
    err_tol::Float64=err_tol,
    q_order::String="big-endian")
    # Initialize the quantum register
    # n::Int64: number of qubits
    # q_order::String: order of the qubits in the quantum register
    # return: quantum register of n qubits
    # q_order == "big-endian"
    # q_order == "little-endian"

    ## notes:
    # the minimum number of qubits is 2 

    # start the function
    n_bas = 2 # number of basis states
    n_qubits = n # number of qubits
    n_dim = 2^n # dimensions of the quantum register/Hilbert space
    state_vector = zeros(2^n) # initialize the state_vector
    # create the default statevector of the quantum register: 
    state_vector[1] = 1 # set the initial state to |000 ...0>
    q_states = zeros(2^n, n) # initialize the quantum states
    q_tab = [0;1] # initialize the basis vectors quantum table
   
    n_count = n_bas
    for i = 2:n_qubits
        q_tab = [zeros(n_count,1) q_tab
                ones(n_count,1) q_tab]
        n_count = n_count*2
    end
    q_states = q_tab # basis vectors 

    # check if the user has provided a custom statevector
    # err_tol = 1e-16 # error tolerance for checking the unitary condition
    if c_sv != nothing
        # check if the custom statevector has the correct dimensions
        if length(c_sv) != n_dim
            error("The custom statevector has the wrong dimensions")
        end
        if isapprox(norm(c_sv), 1, rtol=err_tol) == false
            error("The custom statevector is not normalized")
        end
        state_vector = c_sv
    end

    # one final check for the statevector
    if isapprox(norm(state_vector), 1, rtol=err_tol) == false
        error("The statevector is not normalized")
    end

    return qc_initstruct(n, q_order, n_bas, n_dim, state_vector, q_states)
end # end qc_initialize  

# print the initial state of the quantum register
function print_initstate(qc::qc_initstruct)
    # print the initial state of the quantum register
    # qc::qc_initstruct: quantum register
    # return: print the initial state of the quantum register

    println("The initial state of the quantum register is: ")
    println(qc.register)
end # end print_initstate

end # end module
####### end quantum_circuit.jl #######   