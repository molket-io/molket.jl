module HamiltonianPrint
# Docs 
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the HamiltonianPrint and the HamiltonianPrint functions
# It prints the Hamiltonian matrix in a readable form as well as the eigenvalues
# TSelim April 2024

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

#println("Load HamiltonianPrint")
#println("Using constants from file: ", module_file(constants))

using LinearAlgebra

include("../lib_constants/const_data.jl")
using ..constants: cm1

export Hprint_fast, Evalues_print

function Hprint_fast(H::Matrix{Float64})
    # Print the Hamiltonian matrix in a readable form
    # Inputs
    # H : Hamiltonian matrix
    # Outputs
    # print the Hamiltonian matrix
    # TSelim April 4th 2024
    println("Hamiltonian matrix")
    show(stdout, "text/plain", H)
end # function Hprint_fast
    
function Evalues_print(H, unit_cm1::Bool=true)
    # Print the eigenvalues of the Hamiltonian matrix in a readable form
    # Inputs
    # H : Hamiltonian matrix
    # Outputs
    # print the eigenvalues of the Hamiltonian matrix
    # TSelim April 4th 2024
    println("Eigenvalues of the Hamiltonian matrix")
    eigvals_H=eigvals(H)
    if unit_cm1
        show(stdout, "text/plain", eigvals_H/cm1)
    else
        show(stdout, "text/plain", eigvals_H)
    end
end # function Eigenprint_fast

#function H_WFs1D

end # module