## QTensor is MolKet's tensor library for quantum computing.
## It is a wrapper around the Julia package TensorOperations.jl.

module QTensor
# Path: QTensor.jl
## QTensor is MolKet's tensor library for quantum computing.
# © MIT LICENSE 2023 by MolKet
## wwww.molket.io 
# ==================================================================
# Author: Taha Selim
# Tests: Taha Selim and Alain Chancé
# Check the online notes for the documentation and math details 
# ... of the tensor library QTensor.jl behind the code.
# ==================================================================
# References 
# Einstein summation convention:
# https://en.wikipedia.org/wiki/Einstein_notation


# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load Tensor module: QTensor.jl")
#println("Using constants from file: ", module_file(constants))
# 
# import LinearAlgebra and SparseArrays
using LinearAlgebra # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
using SparseArrays # https://docs.julialang.org/en/v1/stdlib/SparseArrays/

# import conventions
include("../conventions.jl")
#  using ..conventions: big_endian, qubit_start_1
# import quantum gates
include("../quantum_gates.jl")
using ..quantum_gates: q

# Export the tensor functions
export tensor_mul, tensor_decomp, tensor_constr 

# Export the tensor constructors
export q_tenop, q_T2D,q_CU_T4D, q_CU_T8D

# Define the default error tolerance for checking the unitary condition
const err_tol = 1e-16


### General tensor constructor ####
#################################################
# Note: we use the sparse matrices from the Julia package SparseArrays.jl

# Tensor multiplication: general case
function tensor_mul(A,B)
#function tensor_mul(A::SparseMatrixCSC, B::SparseMatrixCSC)
  # A and B are sparse matrices
  # check if A and B are sparse matrices
  if !issparse(A) 
    A = sparse(A)
  end
  if !issparse(B) 
    B = sparse(B)
  end
# Get the sizes of matrix A
  m_A, n_A = size(A)
# Get the sizes of matrix B
  m_B, n_B = size(B)
# Compute the tensor product of A and B
  C = kron(A, B)
  return C # return the tensor product of A and B as a sparse matrix/array
end # tensor_mul


# Tensor multiplication: construct quantum operator acting on a 
# ... a given qubit
function q_tenop(gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}, 
                 q0_control::Int64, q1_control::Int64,
                 q_target::Int64, nqubits::Int64,
                 big_endian::Bool=conventions.big_endian,
                 err_tol::Float64=err_tol)
  # construct the quantum gate acting on the qubit qubit_index 
  # ... of a quantum register of nqubits qubits with options for
  # ... big_endian or little_endian convention
  # ... and having one or two control qubits. 
  # gate is constructed in the ndimensional qubit (Hilbert) space.
  # check if gate is unitary
  # and if gate is a square matrix
  # Get the size of the matrix 
  r_mat = size(gate)[1]
  c_mat = size(gate)[2]
  if r_mat != c_mat
      error("The gate is not a square matrix")
  end
  
  # check the unitary condition
  UU = gate'*gate
  II = Matrix(I, r_mat, c_mat)
  if isapprox(UU, II,rtol=err_tol) == false
      error("The gate is not unitary")
  end 
# check if qubit_index is an integer
if !isa(qubit_index, Int64)
  error("The qubit index must be an integer")
end

return gate # return the quantum gate acting on the qubit qubit_index


end # Qgate_tenop

# Type 1: tensor product of a 2D quantum gate acting on a qubit
#function q_T2D(gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}},
function q_T2D_old(gate, qtarget::Int64, nqubits::Int64)
  # big_endian::Bool=conventions.big_endian,
  # gate_qn is used to construct the quantum gate acting on the qubit i 
  # ... of a quantum register of nqubits qubits
  # gate is a reduced representation of the quantum gate: minimal representation 
  # ... of the quantum gate in 2x2 matrices, 4x4 matrices, etc.
  # qubit_target is the index of the qubit on which the gate acts
  # nqubits is the number of qubits in the quantum register
  # big_endian is a boolean variable that indicates whether the qubits are
  # ... ordered in big endian or little endian
  # Return statevector after the action of the operator 
  # First construct the array of the qubits 
  # ... in the quantum register
  qubit_target = qtarget
  qubits = collect(0:nqubits-1)

  # check the convention of the index_start where qubit counting can start from 0 or 1
  # ... the default is 0.
  # nqubits is the number of qubits in the quantum register
  #if !qubit_start_1
   # qubit_end = nqubits # number of qubits in the quantum register
  #  qubit_begin = 1
  #else
    qubit_end = nqubits-1 # number of qubits in the quantum register
    qubit_initial= 0
  #end
  # check if qubit_target is an integer, and in range
  if !isa(qubit_target, Int64) || qubit_target <  qubit_initial || qubit_target > nqubits-1
    error("The target qubit must be an integer in range: ", qubit_initial, " ", qubit_end)
  end
    # the tensor product used to construct the quantum gate 
  # ... acting on the qubit qubit_index is independent of the convention. 
  gate_construct = 1
  II = Matrix(I, 2, 2)
  #if big_endian
    for i in  qubit_initial:qubit_end
        if i == qubit_target
          gate_construct = kron(gate_construct,gate)
        else
            gate_construct = kron(II,gate_construct)
        end # if i == qubit_target
    end # for loop
  #else
    #for i in  qubit_end:-1:qubit_initial
#    for i in  qubit_initial:qubit_end
#       if i == qubit_target
#    gate_construct = kron(gate,gate_construct)
#           gate_construct = kron(gate,gate_construct)
#        else
#           gate_construct = kron(gate_construct,II)
#        end # if i == qubit_target
#    end # for loop
#  end # if big_endian
  # note: the little-endian convention is not tested yet
return gate_construct
end # Qgate_T2D



####===========================================####
# Type 1: tensor product of a 2D quantum gate acting on a qubit
#function q_T2D(gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}},
function q_T2D(gate, qtarget::Int64, nqubits::Int64)
  # big_endian::Bool=conventions.big_endian)
  # gate_qn is used to construct the quantum gate acting on the qubit i 
  # ... of a quantum register of nqubits qubits
  # gate is a reduced representation of the quantum gate: minimal representation 
  # ... of the quantum gate in 2x2 matrices, 4x4 matrices, etc.
  # qubit_target is the index of the qubit on which the gate acts
  # nqubits is the number of qubits in the quantum register
  # big_endian is a boolean variable that indicates whether the qubits are
  # ... ordered in big endian or little endian
  # Return statevector after the action of the operator 
  # First construct the array of the qubits 
  # ... in the quantum register
  qubit_target = qtarget
  qubits = collect(0:nqubits-1)
  # check the convention of the index_start where qubit counting can start from 0 or 1
  # ... the default is 0.
  # nqubits is the number of qubits in the quantum register
  #if !qubit_start_1
   # qubit_end = nqubits # number of qubits in the quantum register
  #  qubit_begin = 1
  #else
  qubit_end = nqubits-1 # number of qubits in the quantum register
  qubit_initial= 0
#end
# check if qubit_target is an integer, and in range
if !isa(qubit_target, Int64) || qubit_target <  qubit_initial || qubit_target > nqubits-1
  error("The target qubit must be an integer in range: ", qubit_initial, " ", qubit_end)
end
  # the tensor product used to construct the quantum gate 
# ... acting on the qubit qubit_index is independent of the convention. 
gate_construct = 1
II = Matrix(I, 2, 2)
#if big_endian
  for i in  qubit_initial:qubit_end
      if i == qubit_target
        gate_construct = kron(gate_construct,gate)
      else
          gate_construct = kron(gate_construct,II)
      end # if i == qubit_target
  end # for loop
#else
  #  for i in  qubit_end:-1:qubit_initial
  #  for i in  qubit_initial:qubit_end
  #     if i == qubit_target
  #        gate_construct = kron(gate,gate_construct)
  #      else
  #         gate_construct = kron(II,gate_construct)
  #      end # if i == qubit_target
  #  end # for loop
  #end # if big_endian
  # note: the little-endian convention is not tested yet
  return gate_construct
end # Qgate_T2D

####===========================================####

# Type 2: tensor product of a 4D quantum gate acting on a target qubit 
# ... based on a state of a control qubit.
#function q_T4D(Ugate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}};
function q_T4D(Ugate, qtarget::Int64, nqubits::Int64; qcontrol::Int64=0,
  qubit_start_1::Bool=conventions.qubit_start_1,
  big_endian::Bool=conventions.big_endian,
  err_tol::Float64=err_tol)
  # gate is used to construct the quantum gate acting on the qubit_target 
  # ... of a quantum register of nqubits qubits provided that the qubit_control 
  # ... is in the state |1>.
  # Ugate is a reduced representation of the quantum gate: minimal representation 
  # ... of the quantum gate in 2x2 matrices, 4x4 matrices, etc.
  # By default "Ugate" is given in big endian convention. 
  # Check the convention of the "Ugate" carefully until we have a better testing 
  # ... procedure.
  # shorten the word qubit to q in the keywords of the function 
  # the function q_T4D contains qcontrol = 0 by default.
  
  qubit_control = qcontrol
  qubit_target = qtarget
  
  # Check the convention of the index_start where qubit counting can start from 0 or 1
  # ... the default is 0.
  # nqubits is the number of qubits in the quantum register
  if !qubit_start_1
    qubit_end = nqubits # number of qubits in the quantum register
    qubit_begin = 1
  else
    qubit_end = nqubits-1 # number of qubits in the quantum register
    qubit_begin = 0
  end
  # Initiate matrices 
  II = Matrix(I, 2, 2) # indentity matrix
  # Initiate the qubit states 
  ket_0 = [1; 0]
  ket_1 = [0; 1]

  # Initiate the denisty matrix, will be used depending 
  # ... on the state of the control qubit
  rho_0 = ket_0*ket_0' # |0><0| --> if the state is |0>
  rho_1 = ket_1*ket_1' # |1><1| --> if the state is |1>


  # strategy: the gate is constructed via two sums
  # sum 1
  Gate_matrix_I = 1 # initialize the gate matrix
  if qubit_control == qubit_target
    error("The control and target qubits are the same")
  # Big-endian convention
  elseif (big_endian && (qubit_control < qubit_target)) || 
          (!big_endian && (qubit_control > qubit_target))     
    # construct the quantum gate acting on the qubit qubit_control
      for iq in 0:nqubits-1
          if iq == qubit_control 
            Gate_matrix_I = kron(Gate_matrix_I, rho_0)
          elseif iq == qubit_target
            Gate_matrix_I = kron(Gate_matrix_I, II)
          else
            Gate_matrix_I = kron(Gate_matrix_I, II)
          end
      end # for loop
  # Little-endien convention
  else
      for iq in 0:nqubits-1
          if iq == qubit_control 
              Gate_matrix_I = kron(Gate_matrix_I,rho_0)
          elseif iq == qubit_target
                Gate_matrix_I = kron( Gate_matrix_I,II)
          else
              Gate_matrix_I = kron(Gate_matrix_I,II)
          end
      end # for loop
  end # if big_endian && q_control < q_target ...
  # sum 2
  Gate_matrix_II = 1 # initialize the gate matrix
  if qubit_control == qubit_target
    error("The control and target qubits are the same")
  # Big-endian convention
  elseif (big_endian && (qubit_control < qubit_target)) || 
          (!big_endian && (qubit_control > qubit_target))     
    # construct the quantum gate acting on the qubit qubit_control
      for iq in 0:nqubits-1
          if iq == qubit_control 
            Gate_matrix_II = kron(Gate_matrix_II, rho_1)
          elseif iq == qubit_target
            Gate_matrix_II = kron(Gate_matrix_II, Ugate)
          else
            Gate_matrix_II = kron(Gate_matrix_II, II)
          end
      end # for loop
  # Little-endien convention
  else
      for iq in 0:nqubits-1
          if iq == qubit_control 
              Gate_matrix_II = kron(Gate_matrix_II, rho_1)
          elseif iq == qubit_target
                Gate_matrix_II = kron( Gate_matrix_II,Ugate)
          else
              Gate_matrix_II = kron(Gate_matrix_II,II)

          end
      end # for loop
    end # if big_endian && q_control < q_target ...
    Gate_matrix = Gate_matrix_I + Gate_matrix_II

  return Gate_matrix
end # tensor_gate_apply


# Type 3: tensor product of a 8D quantum gate acting on a target qubit 
# ... based on a state of two control qubits.
function q_CU_T8D(Ugate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}, 
    qubit_control_I::Int64, qubit_control_II::Int64, qubit_target::Int64, nqubits::Int64,
    qubit_start_1::Bool=conventions.qubit_start_1,
    big_endian::Bool=conventions.big_endian,
    err_tol::Float64=err_tol)
    # print that gate is not implemented yet 
    println("The gate is not implemented yet")
    return 0
end # Qgate_CU_T8D

## Qubit tensor product 


### Tensor decomposition: general case ###
# Kroncker decomposition of a tensor

end # module


