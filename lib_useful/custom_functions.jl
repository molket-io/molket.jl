module custom_functions
# This module contains special or custom functions for general tasks in julia. 

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load quantum_circuit constructor")
#println("Using constants from file: ", module_file(constants))


using LinearAlgebra
using SparseArrays

export MK_sortrows

# function MK_sortrows with an input matrix of union types of Float64 and Int64
function MK_sortrows(A::Matrix{Float64})
    # sort the rows of the matrix A and export the sorting indices
    # A::Array{Float64,2}: matrix to be sorted
    # return: sorted matrix and sorting indices
    # more references over the sorting 
    # https://docs.julialang.org/en/v1/base/sort/
    # sort the rows of the matrix qc_le.q_states and export the sorting indices
    A_sorted=sortslices(A,dims=1)
    # where are the indices of the sorting?
    A_sorted_indices=sortperm(A,dims=1)
    return A_sorted, A_sorted_indices
end # end sortrows



end # module