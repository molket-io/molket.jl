## QTensor is MolKet's tensor library for quantum computing.
## It is a wrapper around the Julia package TensorOperations.jl.

module QTensor
# Path: QTensor.jl
## QTensor is MolKet's tensor library for quantum computing.
# © MIT LICENSE 2023 by MolKet
## wwww.molket.io 

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
include("conventions.jl")
  using .conventions: big_endian
# import quantum gates
include("quantum_gates.jl")
using ..quantum_gates: Qgate

# Export the functions
export tensor_mul, tensor_decomp, tensor_constr 


# Define the default error tolerance for checking the unitary condition
const err_tol = 1e-16


### General tensor constructor ####
#################################################
# Note: we use the sparse matrices from the Julia package SparseArrays.jl

# Tensor multiplication: general case
function tensor_mul(A::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}, 
                    B::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}})
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
function Qgate_tenop(gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}, 
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
function Qgate_T2D(gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}, 
                 qubit_target::Int64, nqubits::Int64, 
                 qubit_start::Int64=0,
                 big_endian::Bool=conventions.big_endian,
                 err_tol::Float64=err_tol)
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
  qubits = Array{Int64}(undef, nqubits)
  qubits = collect(0:nqubits-1)
  # check the index_start is an integer and starts from 0 or 1
  if !isa(qubit_start, Int64)
    error("The index_start must be an integer")
  end
  if qubit_start != 0 && qubit_start != 1
    error("The index_start must be 0 or 1")
  end
  # we only take the case of qubit_start = 0
  # check if qubit_target is an integer
  if !isa(qubit_target, Int64)
    error("The qubit index must be an integer")
  end
  if qubit_start == 0
    nqubits_convention = nqubits-1
  elseif qubit_start == 1
    nqubits_convention = nqubits
  end
   
  # check that the qubit_target is in the quantum register
  if qubit_target < qubit_start || qubit_target > nqubits-1
    error("The target qubit is not in the quantum register")
  end
    # the tensor product used to construct the quantum gate 
  # ... acting on the qubit qubit_index is independent of the convention. 
  gate_construct = 1
  II = Matrix(I, 2, 2)
  for i in qubit_start:nqubits_convention
    if i == qubit_target
      gate_construct = kron(gate, gate_construct)
    else
      gate_construct = kron(II, gate_construct)
    end
  end

end # tensor_gate_apply


# Type 2: tensor product of a 4D quantum gate acting on a target qubit 
# ... based on a state of a control qubit.


# Type 3: tensor product of a 8D quantum gate acting on a target qubit 
# ... based on a state of two control qubits.


### Tensor decomposition: general case ###
# Kroncker decomposition of a tensor

end # module


