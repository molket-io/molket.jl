module quantum_circuit
# © MolKet 2023, MIT License.
# This module is part of MolKet.jl package.
## For more information, please visit out website:
# www.molket.io

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load quantum_circuit constructor")
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


# import conventions
include("conventions.jl")
  using .conventions: big_endian

# import quantum gates
 include("quantum_gates.jl")
using ..quantum_gates: q

# import the tensor library
include("lib_tensor/QTensor.jl")
using ..QTensor: q_T2D, q_T4D

# import the sorting function
include("lib_useful/custom_functions.jl")
using ..custom_functions: MK_sortrows

# export functions
export qc_initialize, init_register, print_initstate,
show_statevector

# Define the default error tolerance for checking the norm of the vector
const err_tol = 1e-15


# # Initialize the order of the qubits in the quantum register
# function qubit_order(q_order::String)
#     # q_order::String: order of the qubits in the quantum register
#     # return: order of the qubits in the quantum register
#     # not used function 
#     if q_order == "big-endian"
#         return q_order
#     elseif q_order == "little-endian"
#         return q_order
#     else
#         error("The order of the qubits in the quantum register is not defined")
#     end
# end # end qubit_order

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
    n_qubits::Int64
    q_order::String
    n_bas::Int64
    n_dim::Int64
    state_vector
#    state_vector::Array{Float64,1}
    q_states
# table of quantum gates and the operations on the quantum register    
    op_table::Matrix
# Store the quantum circuit matrix representation
    qc_matrix
# big_endian is used to enforce a default convention 
    big_endian::Bool 
# show_matrix is used to print the matrix or the quantum gates by default    
    show_op_mat::Bool
# show the matrix representation of the quantum circuit
    show_qc_mat::Bool
#    q_states::Array{Float64,2}
end # end qc_initialize


function qc_init_old(n::Int64;
    big_endian::Bool=conventions.big_endian,
    c_sv= nothing, 
#     c_sv::Union{Vector{Float64}, Vector{Int64}, Vector{ComplexF64}} = nothing,     
    err_tol::Float64=err_tol,
    show_matrix::Bool=conventions.show_matrix)
    # Initialize the quantum register
    # n::Int64: number of qubits
    # q_order::String: order of the qubits in the quantum register
    # return: quantum register of n qubits
    # q_order == "big-endian"
    # q_order == "little-endian"

    # start the function
    n_bas = 2 # number of basis states
    n_qubits = n # number of qubits
    n_dim = 2^n # dimensions of the quantum register/Hilbert space
    state_vector = zeros(2^n) # initialize the state_vector
    # create the default statevector of the quantum register: 
    state_vector[1] = 1 # set the initial state to |000 ...0>
    q_states = zeros(2^n, n) # initialize the quantum states
    q_tab = [0;1] # initialize the basis vectors quantum table
   
    # store the order of the qubits, big-endian or little-endian
    # for printing purpose only 
    if big_endian
        q_order = "big-endian"
    else
        q_order = "little-endian"
    end

    ## notes:
    # the minimum number of qubits is 2 
    if n_qubits == 1
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
   else
        n_count = n_bas
        if !big_endian
            for i = 2:n_qubits
                q_tab = [ q_tab zeros(n_count,1)
                       q_tab  ones(n_count,1) ]
                n_count = n_count*2
            end
        else
            for i = 2:n_qubits
                q_tab = [zeros(n_count,1) q_tab
                        ones(n_count,1) q_tab]
                n_count = n_count*2
            end
        end
        # for i = 2:n_qubits
        #     q_tab = [zeros(n_count,1) q_tab
        #             ones(n_count,1) q_tab]
        #     n_count = n_count*2
        # end
    q_states = q_tab # basis vectors 
end # end if n_qubits == 1

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

#    return qc_initstruct(n_qubits, q_order, n_bas, n_dim, 1.0, [1.0 1.0])
    return qc_initstruct(n_qubits, q_order, n_bas, n_dim, state_vector, 
                        q_states, big_endian,show_matrix)

#    return 1

end # end qc_initialize  

# prepare the gate in case of being parametrized, i.e. rotational gate 
# or a phase gate
function qU_prep(Qgate;
    theta=nothing,phi=nothing,lambda =nothing)
# some functions from the quantum_gates.jl file are 
# are not or this function is not adapted to. 
# We have to check them case by case 
if theta == nothing && phi == nothing && lambda == nothing
return Qgate

elseif theta != nothing && phi != nothing && lambda != nothing
Qgate = Qgate(theta=theta,phi=phi,lambda=lambda)
return Qgate

elseif theta != nothing && phi != nothing
Qgate = Qgate(theta=theta,phi=phi)
return Qgate

elseif theta != nothing && lambda != nothing
Qgate = Qgate(theta=theta,lambda=lambda)
return Qgate

elseif phi != nothing && lambda != nothing
Qgate = Qgate(phi=phi,lambda=lambda)
return Qgate

elseif theta != nothing
Qgate = Qgate(theta=theta)
return Qgate

elseif phi != nothing
Qgate = Qgate(phi=phi)
return Qgate

elseif lambda != nothing
Qgate = Qgate(lambda=lambda)
return Qgate

else
return error("Error in Qgate_rot_prep")
end # end if

end # end Qgate_prep

# function to choose the corresponding tensor library 
function Op_tensor(Qgate, qtarget::Int64, nqubits::Int64;
    q1control=nothing,q2control=nothing, big_endian::Bool=conventions.big_endian)
# get the size of Qgate matrix 
# get the size of Qgate
n_rows, n_cols = size(Qgate)   
# test if the size of the gate is 2x2
if n_rows != 2 && n_cols != 2
error("The size of the gate is not 2x2")
end 

# once passing the previous check, we can check the size of the gate
# if the size is 2x2, then it is a single qubit gate
# if the size is 4x4, then it is a two qubit gate
# if the size is 8x8, then it is a three qubit gate
# This helps us to determine the tensor library to use   
# in most cases, the one qubit gate is sufficient to determine 
# the tensor library to use with the information about 
# the control qubits and the target qubit 
# single qubit gate
# check if the user has provided a control qubit
if q1control == nothing && q2control == nothing
# no control qubit1 and qubit2, only target qubit
# return the qubit gate in the Hilbert space of a 
# quantum register size 
Qgate = q_T2D(Qgate, qtarget, nqubits,big_endian=big_endian)
return Qgate
elseif q2control == nothing
# only control qubit1 and target qubit
# return the qubit gate in the Hilbert space of a
# quantum register size
Qgate = q_T4D(Qgate, qcontrol=q1control, qtarget=qtarget, 
         nqubits=nqubits, big_endian=big_endian)
return Qgate
end # end if q1control == nothing && q2control == nothing
# now, let's export the gate represented in the Hilbert space of a
# quantum register size
end # end Op_tensor

# initialize a table with first row of [action gate/operator control1_qubit,   control2_qubit, target_qubit theta     phi     lambda; gate_object]
function init_op_tab()
    # initialize the first row of the table q_tab 
    # Op_ind is the index of the operation in the quantum circuit
    # Op is the name of the operation
    # q1_control is the first control qubit
    # q2_control is the second control qubit
    # q_target is the target qubit
    # theta is the angle of rotation theta 
    # phi is the angle of rotation phi
    # lambda is the angle of rotation lambda
    # object is the object of the operation, could be a gate function or a matrix
    # op_mat is the matrix representation of the operation
#    op_tab[2,1:9]=["init" "init" 0 0 0 0 0 0 0]
    op_tab = Matrix(undef, 1,10)
    op_tab[1,1:10]=["Op_ind","Op","q1_control","q2_control","q_target", 
                "theta","phi","lambda","object","op_mat"]
    return op_tab
end # end init_op_tab


function qc_init(n::Int64;
    big_endian::Bool=conventions.big_endian,
    c_sv= nothing, 
#     c_sv::Union{Vector{Float64}, Vector{Int64}, Vector{ComplexF64}} = nothing,     
    err_tol::Float64=conventions.err_tol,
    show_op_mat::Bool=conventions.show_op_mat,
    show_qc_mat::Bool=conventions.show_qc_mat)
    # Initialize the quantum register
    # n::Int64: number of qubits
    # q_order::String: order of the qubits in the quantum register
    # return: quantum register of n qubits
    # q_order == "big-endian"
    # q_order == "little-endian"

    # start the function
    n_bas = 2 # number of basis states
    n_qubits = n # number of qubits
    n_dim = 2^n # dimensions of the quantum register/Hilbert space
    state_vector = zeros(2^n) # initialize the state_vector
    # create the default statevector of the quantum register: 
    state_vector[1] = 1 # set the initial state to |000 ...0>
    q_states = zeros(2^n, n) # initialize the quantum states
    q_tab = [0;1] # initialize the basis vectors quantum table
   
    # store the order of the qubits, big-endian or little-endian
    # for printing purpose only 
    if big_endian
        q_order = "big-endian"
    else
        q_order = "little-endian"
    end

    ## notes:
    # the minimum number of qubits is 2 
    if n_qubits == 1
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
   else
        n_count = n_bas
        if !big_endian
            for i = 2:n_qubits
                q_tab = [ q_tab zeros(n_count,1)
                       q_tab  ones(n_count,1) ]
                n_count = n_count*2
            end
        else
            for i = 2:n_qubits
                q_tab = [zeros(n_count,1) q_tab
                        ones(n_count,1) q_tab]
                n_count = n_count*2
            end
        end
        # for i = 2:n_qubits
        #     q_tab = [zeros(n_count,1) q_tab
        #             ones(n_count,1) q_tab]
        #     n_count = n_count*2
        # end
        q_states = q_tab # basis vectors 
    end # end if n_qubits == 1

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
    # Initialize the matrix representation of the quantum circuit to 
    # the identity matrix of size n_dim
    qc_matrix = Matrix(I, n_dim, n_dim)

    # initiate the table of quantum gates and operations
    op_table = init_op_tab()
#    return qc_initstruct(n_qubits, q_order, n_bas, n_dim, 1.0, [1.0 1.0])
    return qc_initstruct(n_qubits, q_order, n_bas, n_dim, state_vector, 
                        q_states, op_table, qc_matrix, big_endian, 
                        show_op_mat, show_qc_mat)

#    return 1

end # end qc_initialize  



# print the initial state of the quantum register
function show_statevector(qc)
    # print the initial state of the quantum register
    # qc::qc_initstruct: quantum register
    # return: print the initial state of the quantum register
    #println("The initial state of the quantum register is: ")
    #println(qc.state_vector)
    # print the initial state of the quantum register with the quantum 
    # states in the computational basis
    #println("the quantum register is: ")
    q_states = qc.q_states
    state_vector = qc.state_vector
    # for now, the q_order is used to print the basis of the Hilbert space
    # ... in the computational basis with little-endian order 
    if !big_endian
        q_states, q_ind = MK_sortrows(q_states)
        state_vector = state_vector[q_ind[:,1]]
    end # end if
    
    # Note: the basis of the Hilbert space are the quantum states
    # .. they are arranged and sorted in the computational basis 
    # .. according to the q_order=lille-endian.
    # .. accordingly, the statevector is arranged and sorted in the
    # .. computational basis according to the q_order=little-endian 
    # .. as well. 

    # convert the quantum states to integers
    q_table = zeros(Int, qc.n_dim, qc.n_qubits)
    for iq=1:qc.n_qubits
        for i in 1:qc.n_dim
            q_table[i,iq] = trunc(Int, q_states[i,iq])
        end # end for
    end # end for
  
    for i in 1:qc.n_dim
        println( state_vector[i], " * | ", string(q_table[i,:]), ">")
    end # end for
    #show(stdout, "text/plain", [qc.state_vector trunc(Int,qc.q_states)])
end # end print_initstate

# Apply a quantum gate to the quantum register
function op_v1(qc, Qgate)
    # Apply a quantum gate to the quantum register
    # qc::quantum register
    # gate::Qgate: quantum gate
    # return: quantum register with the quantum gate applied
    Nqubits = qc.n_qubits
    Nstates = qc.n_dim
    state_vector = qc.state_vector
    # First check if the dimensions of the quantum gate are the same 
    Qgate_dim = size(Qgate)
    if Qgate_dim[1] != Qgate_dim[2]
        error("The quantum gate is not square")
    end # end if
    # check if the quantum gate is unitary
    #if ishermitian(Qgate) == false
    #    error("The quantum gate is not unitary")
    #end # end if
    # check the unitary condition
    UU = Qgate'*Qgate
    II = Matrix(I, Qgate_dim[1], Qgate_dim[2])
    if isapprox(UU, II,rtol=err_tol) == false
        error("The gate is not unitary")
    end
    # check if the dimensions of the quantum gate 
    # ... and the quantum register do match 
    if Qgate_dim[1] != Nstates
        error("The quantum gate and the quantum register do not match")
    end # end if
    # Apply the quantum gate to the quantum register
    state_vector = Qgate * state_vector
    qc.state_vector = state_vector
    return qc
end # end apply_gate!

# function to update the table of quantum gates and operations
function qctab_update(qctab, Qgate, qtarget::Int64; 
    q1control=nothing,q2control=nothing,
    theta=nothing,phi=nothing,lambda=nothing, 
    op_name=nothing,op_mat=nothing,
    big_endian::Bool=conventions.big_endian)

    # get the size of the rows of the table
    n_rows, n_cols = size(qctab)
    qctab1 = qctab
    op_ind = n_rows 
    # table is 
    #op_tab[1,1:10]=["Op_ind","Op","q1_control","q2_control","q_target", 
    #"theta","phi","lambda","object","op_mat"]

    # initialize the second row of the table: if nothing is provided 
    # then print nothing in the cell of the table
    if op_name == nothing
        op_name = "nothing"
    end
    if q1control == nothing
        q1control = "nothing"
    end
    if q2control == nothing
        q2control = "nothing"
    end
    if theta == nothing
        theta = "nothing"
    end
    if phi == nothing
        phi = "nothing"
    end
    if lambda == nothing
        lambda = "nothing"
    end
    if op_mat == nothing
        op_mat = "nothing"
    end
    # update the table
    qctab2 = Matrix(undef, 1,10)
    qctab2[1,1:10] = [op_ind, op_name, q1control, q2control, qtarget, 
        theta, phi, lambda, Qgate, op_mat]
        return [qctab1; qctab2]
end # end qctab_update
    

# Apply a quantum gate to the quantum register
function op(qc, Qgate, qtarget::Int64; 
    q1_control=nothing, q2_control=nothing, 
    theta=nothing, phi=nothing, lambda=nothing, 
    err_tol::Float64=conventions.err_tol)
    # Apply a quantum gate to the quantum register
    # qc::quantum register
    # gate::Qgate: quantum gate, 2x2 matrix representing a single qubit gate
    # return: quantum register with the quantum gate applied
    # reading the data from the quantum register qc
    Nqubits = qc.n_qubits
    Nstates = qc.n_dim
    state_vector = qc.state_vector
    
    # first record the name of the function
    Qgate_name = string(Qgate)
    # Evaluate the gate and prepare the gate for the tensor product
    # First, check if the function is a rotational/phase gate 
    # and if so Evaluate it over the theta, phi, and lambda parameters
    # if not, then the function is a single qubit gate
    # check if the user has provided a rotational/phase gate 
    # the check is done by checking if the user has provided the
    # theta, phi, and lambda parameters
    # if so the gate is evaluated and the corresponding matrix is returned
    Qgate = qU_prep(Qgate,theta=theta,phi=phi,lambda=lambda)

    # test the gate 
    # First check if the dimensions of the quantum gate are the same 
    Qgate_dim = size(Qgate)
    if Qgate_dim[1] != Qgate_dim[2]
        error("The quantum gate is not square")
    end # end if
    # check if the quantum gate is unitary
    #if ishermitian(Qgate) == false
    #    error("The quantum gate is not unitary")
    #end # end if
    # check the unitary condition
    UU = Qgate'*Qgate
    II = Matrix(I, Qgate_dim[1], Qgate_dim[2])
    if isapprox(UU, II,rtol=err_tol) == false
        error("The gate is not unitary")
    end
    # check if the dimensions of the quantum gate 
    # ... and the quantum register do match 
    #if Qgate_dim[1] != Nstates
    #    error("The quantum gate and the quantum register do not match")
    #end # end if

    gate_tensor = Op_tensor(Qgate, qtarget,Nqubits,q1control=q1_control, 
                            q2control=q2_control,
                            big_endian=big_endian)
    # and the gate is evaluated directly
    # in the Hilbert space
    
    # Apply the quantum gate to the quantum register
    state_vector = gate_tensor * state_vector
    qc.state_vector = state_vector

    # update the matrix representation of the quantum circuit
    qc.qc_matrix = gate_tensor * qc.qc_matrix

    # update the op_table
    qctab = qc.op_table
    qc.op_table = qctab_update(qctab, Qgate, qtarget, 
    q1control=q1_control,q2control=q2_control,
    theta=theta,phi=phi,lambda=lambda, 
    op_name=Qgate_name,op_mat= qc.qc_matrix)
    return qc
end # end apply_gate!


end # end module
####### end quantum_circuit.jl #######   