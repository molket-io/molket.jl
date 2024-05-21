module MK_Heig

# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the eigenvalus and eigenvectors of the Hamiltonian
# used in the quantum molecular dynamics simulations

# Author: TSelim April 2024


# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

using LinearAlgebra

# call special polynomials
# call the lib_SpecialPolynomials.jl
include("../lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: ghermite


# call the lib_useful.jl with the def_matrix function which initializes a matrix
include("../lib_useful/custom_functions.jl")
using ..custom_functions: def_matrix, MK_sortrows, eye

# call the lib_HO_wfs.jl to get the harmonic oscillator wave functions
include("../lib_wavefunctions/HO_wfs.jl")
using ..HO_WFs: HO_wfs1D, HO_wfs1De



export Heig, H_ho1D


function Heig(H_mat)
    # Eigenvalues and eigenvectors of the Hamiltonian
    # Inputs
    # H_mat : Hamiltonian matrix
    # Outputs
    # Ev     : Eigenvalues
    # V     : Eigenvectors
    # Reference:

    # TSelim April 30th 2024
    Ev, V = eigen(H_mat)
    return Ev, V
end # function Heig

mutable struct H_1Dstruct
    qnum::Array{Int64,1} # qnum = [n1;n2;n3,...]
    Ev::Array{Float64,1} # Eigenvalues
    V::Array{Float64,2} # Eigenvectors
    WFs:: Array{Union{Float64,Int64,ComplexF64},2} # wave functions
    Qvec::Array{Float64,1} # Qgrid
end

#function H_ho1D(H_mat,qnum, Qgrid; WFbas_key::String="HO",quad_key::Bool=false)
function H_ho1D(H_mat,qnum, Qgrid; WFbas_key::String="HO")

    # Hamiltonian of the 1D harmonic oscillator diagonalized in harmonic oscillator basis
    # Eigenvalues and eigenvectors of the Hamiltonian and the vector of 
    # the harmonic oscillator wave functions with their quantum numbers
    # Inputs
    # H_mat : Hamiltonian matrix
    # qnum  : quantum numbers, 1D vector of quantum numbers
    #       qnum = [n1;n2;n3,...]
    # Outputs
    # Ev     : Eigenvalues
    # V     : Eigenvectors
    # psi_n  : wave functions, could be harmonic or anharmonic

    # Reference:

    # TSelim April 30th 2024
    # Get the eigenvalues and eigenvectors of the Hamiltonian
    Ev, V = Heig(H_mat)
    # Check if the size of the qnum matches the size of the Hamiltonian matrix
    if size(H_mat,1) != length(qnum)
        println("The size of the Hamiltonian matrix does not match the size of the qnum")
        println("Check the Hamiltonian matrix and the qnum")
    end
    # Check the maximum quantum number
    qnum_max = maximum(qnum)
    # Check the length of the Qgrid
    Qn = length(Qgrid)
    # Check the length of the qnum
    N = length(qnum)
    # initializes the wave functions
    WFs_ho = def_matrix(Qn,N)
    # Evaluate the wave functions based on the basis key
    if WFbas_key == "HO"
        for i in 1:Qn
            # if !quad_key
            WFs_ho[i,:] = HO_wfs1D(qnum_max,Qgrid[i]) 
                                         # rows are the Q values & columns are n
            # else
                # WFs_ho[i,:] = HO_wfs1De(qnum_max,Qgrid[i]) 
                #                          # rows are the Q values & columns are n    
                #                          # this function has the e^(-Q^2/2) factor removed
                #                          # since it is a part of the quadrature weight.
            # end
        end
    end
    # prepare the total wavefunction 
    WFs = def_matrix(Qn,N)
    # multiply the coefficients of the eigenvectors with the ho wavefunctions 
    WFs = WFs_ho * V

    return H_1Dstruct(qnum,Ev,V,WFs,Qgrid)
end # function H_ho1D

# A function to check the probability of the wave functions
function prob_hoWFs(nquad::Int64;WFbas_key::String="HO")
    # a function to check the total probability and normalization of the wave functions
    # of harmonic oscillators
    # Inputs
    # nquad: number of quadrature points
    # Outputs
    # prob : probability of the wave functions
    # By default now the wavefunctions are harmonic oscillator wave functions (Hermite polynomials)
    # Reference:

    # TSelim May 15th 2024
    
    # First getting the quadrature points and weights
    Qgrid, Wgrid = gausslegendre(nquad)
    Qn = nquad
    # evaluate the wave functions at the quadrature points
    WFs = def_matrix(nquad,nquad)
    if WFbas_key == "HO"
        WFs_ho = def_matrix(Qn,N)
        for i in 1:Qn
            WFs_ho[i,:] = HO_wfs1De(qnum_max,Qgrid[i]) 
                                    # rows are the Q values & columns are n    
        end
    end
    # multiply the evaluation of prob of the wave functions with the weights
    prob = def_matrix(nquad,1)
    for i in 1:Qn
        prob[i] = WFs_ho[i,:]'*Wgrid[i]
    end


    return WFs, prob, Qgrid
end # function prob_hoWFs

end # module MK_Heig