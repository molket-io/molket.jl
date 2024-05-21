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
    
function Evalues_print(Ev; unit_cm1::Bool=true)
    # Print the eigenvalues of the Hamiltonian matrix in a readable form
    # Inputs
    # H : Hamiltonian matrix
    # Outputs
    # print the eigenvalues of the Hamiltonian matrix
    # TSelim April 4th 2024
    println("Eigenvalues of the Hamiltonian matrix")
    eigvals_H=Ev
    # make sure the eigenvalues are real and no imaginary part
    #if imag(eigvals_H[1]) != 0
    #    println("Eigenvalues have imaginary part")
    #    println("Check the Hamiltonian matrix")
    #end
    # make the eigenvalues real
    eigvals_H = real(eigvals_H)
    if unit_cm1
        show(stdout, "text/plain", eigvals_H/cm1)
    else
        show(stdout, "text/plain", eigvals_H)
    end
end # function Eigenprint_fast

function Evec_print(V,num)
    # Print the eigenvectors of the Hamiltonian matrix in a readable form
    # Inputs
    # H : Hamiltonian matrix
    # num: number of eigenvectors to print starting from the lowest
    # Outputs
    # print the eigenvectors of the Hamiltonian matrix
    # TSelim April 30th 2024
    println("Eigenvectors of the Hamiltonian matrix")
    # print them in a readable form as vectors 
    for i in 1:num
        println("Eigenvector number: ", i)
        show(stdout, "text/plain", V[:,i])
    end
    #show(stdout, "text/plain", V)
end # function Evec_print

end # module HamiltonianPrint


