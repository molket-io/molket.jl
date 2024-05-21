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

export MK_sortrows, str2int, str2bin, eye, def_matrix

# function MK_sortrows with an input matrix of union types of Float64 and Int64
function MK_sortrows(A::Matrix{Float64})
    # sort the rows of the matrix A and export the sorting indices
    # A::Array{Float64,2}: matrix to be sorted
    # return: sorted matrix and sorting indices
    # more references over the sorting 
    # Author: Taha Selim
    # https://docs.julialang.org/en/v1/base/sort/
    # sort the rows of the matrix qc_le.q_states and export the sorting indices
    # A_sorted=sortrows(A,dims=1)
    A_sorted=sortslices(A,dims=1)
    # where are the indices of the sorting?
    A_sorted_indices=sortperm(A,dims=1)
    return A_sorted, A_sorted_indices
end # end sortrows

# convert string to integer
function str2int(s)
    # convert string to integer
    # Author: Taha Selim
    # Date: 2023-12-28
    # note: note tested yet or used
    n = length(s)
    r = 0
    for i in 1:n
        r = r + parse(Int64, s[i])*2^(n-i)
    end
    return r
end

# convert string to binary form 
function str2bin(s)
    # convert string to integer
    # Author: Taha Selim
    # Date: 2023-12-28
    # note: note tested yet or used
    n = length(s)
    r = zeros(Int64, n)
    for i in 1:n
        r[i] = parse(Int64, s[i])
    end
    return r
end

# function eye that creates an identity matrix
function eye(n::Int)
    # create an identity matrix of size n

    # n::Int: size of the identity matrix
    # return: identity matrix of size n
    II = Matrix{Float64}(I,n,n)
    return II
end # end eye


# function to define a matrix 
function def_matrix(nr,nc)
    # define a matrix of size nr x nc
    # nr: number of rows
    # nc: number of columns
    # return: matrix of size nr x nc
    # used as W = def_matrix(2,2), for example: 
    # W[1,1] = 1
    # W[1,2] = 2.2
    # W[2,1] = 3.3
    #W[2,2] = 1.0 + 2.0im
    M_2D =  Matrix{Union{Int64, Float64, ComplexF64}}(undef, nr, nc)
    # fill the matrix with zeros
    M_2D .= 0
    return M_2D
end # end matrix_def
end # module