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

println("Load quantum gates constructor")
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


export tensor_mul, tensor_decomp, tensor_constr 

### General tensor constructor ####
#################################################
# Note: we use the sparse matrices from the Julia package SparseArrays.jl

# Tensor multiplication: general case
function tensor_mul(A::Array{ComplexF64,2}, B::Array{ComplexF64,2})
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
function Qgate_tenop(gate::Array{ComplexF64,2}, 
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
function Qgate_T2D(gate::Array{ComplexF64,2}, 
                 qubit_target::Int64, nqubits::Int64, 
                 big_endian::Bool=conventions.big_endian,
                 err_tol::Float64=err_tol)
  # gate_qn is used to construct the quantum gate acting on the qubit i 
  # ... of a quantum register of nqubits qubits
  # gate is a reduced representation of the quantum gate: minimal representation 
  # ... of the quantum gate in 2x2 matrices, 4x4 matrices, etc.
  # qubit_index is the index of the qubit on which the gate acts
  # nqubits is the number of qubits in the quantum register
  # big_endian is a boolean variable that indicates whether the qubits are
  # ... ordered in big endian or little endian
  # Return statevector after the action of the operator 
  
    # the tensor product used to construct the quantum gate 
  # ... acting on the qubit qubit_index is independent of the convention. 
  gate_construct = 1
  II = Matrix(I, 2, 2)
  for i in 1:nqubits
    if i == qubit_target
      gate_construct = tensor_mul(gate, gate_construct)
    else
      gate_construct = tensor_mul(II, gate_construct)
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


