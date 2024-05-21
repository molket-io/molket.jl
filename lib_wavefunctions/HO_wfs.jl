module HO_WFs
# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the harmonic oscillator wave functions
# used in the quantum molecular dynamics simulations
# and the quantum algorithms

# Author: TSelim April 2024

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

using LinearAlgebra


# call the lib_SpecialPolynomials.jl
include("../lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: ghermite


# call the lib_useful.jl with the def_matrix function which initializes a matrix
include("../lib_useful/custom_functions.jl")
using ..custom_functions: def_matrix, MK_sortrows, eye

export HO_wfs1D, HO_wfs1De

function HO_wfs1D(n::Int64,x;omega::Float64=1.0,mu::Float64=1.0)
    # Harmonic oscillator wave functions
    # Inputs
    # n    : degree
    # x    : the point
    # omega: frequency
    # mu   : mass
    # Outputs
    # psi_n(x) : wave function 
    # default is omega = mu = 1.0 which means calculating the wavefunctions in terms 
    # of the unitless coordinates Q = sqrt(mu*omega)*x
    # Reference:
    # https://en.wikipedia.org/wiki/Hermite_polynomials
    # https://en.wikipedia.org/wiki/Harmonic_oscillator
    # TSelim April 4th 2024

    # Note: the wavefunction is scaled by default to 1.0 so that the 
    # ... scaling factor alpha = sqrt(mu*omega) is not considered in the 
    # ... default case.
    #psi = (mu*omega/(pi))^(1/4) * 1/sqrt(2^n * factorial(n)) 
    #* exp(-mu*omega*x^2/2) * ghermite(n,mu*omega*x^2)
    coef = 1/sqrt(2^n * factorial(n)) 
    #coef = 1
    const_pi = pi^(-1/4)
    const_muomega = mu*omega # assuming that hbar = 1 becuase of atomic units
    const_exp = exp(-const_muomega*x^2/2)
    Hpoly = ghermite(n,sqrt(const_muomega)*x) # Hermite polynomial
    psi = coef * const_pi * (const_muomega)^(1/4) * const_exp * Hpoly
    return psi
end # function HO_wfs1D


function HO_wfs1De(n::Int64,x;omega::Float64=1.0,mu::Float64=1.0)
    # Harmonic oscillator wave functions without the exponential part 
    # to be used in numerical integration using quadrature methods like 
    # Gauss-Hermite quadrature
    # Inputs
    # n    : degree
    # x    : the point
    # omega: frequency
    # mu   : mass
    # Outputs
    # psi_n(x) : wave function 
    # default is omega = mu = 1.0 which means calculating the wavefunctions in terms 
    # of the unitless coordinates Q = sqrt(mu*omega)*x
    # Reference:
    # https://en.wikipedia.org/wiki/Hermite_polynomials
    # https://en.wikipedia.org/wiki/Harmonic_oscillator
    # TSelim April 4th 2024
    # TSelim May 15th 2024
    # Note: the wavefunction is scaled by default to 1.0 so that the 
    # ... scaling factor alpha = sqrt(mu*omega) is not considered in the 
    # ... default case.
    #psi = (mu*omega/(pi))^(1/4) * 1/sqrt(2^n * factorial(n)) 
    #* exp(-mu*omega*x^2/2) * ghermite(n,mu*omega*x^2)
    #coef = 1/sqrt(2^n * factorial(n)) 
    # ************ Note, this is for testing don't use it for the wave functions
    # Note, this is for testing don't use it for the wave functions
    # Note, this is for testing don't use it for the wave functions
    # Note, this is for testing don't use it for the wave functions
    coef = 1
    const_pi = pi^(-1/4)
    const_muomega = mu*omega # assuming that hbar = 1 becuase of atomic units
    const_exp = exp(-const_muomega*x^2/2)
    Hpoly = ghermite(n,sqrt(const_muomega)*x) # Hermite polynomial
    # U = def_matrix(1,n)
    # U[:,1] = const_muomega^(1/4) * const_pi * const_exp
    # if n > 0 
    #     U[:,2] = sqrt(2) * x * U[:,1]
    # end
    psi = coef * const_pi * (const_muomega)^(1/4) * const_exp * Hpoly
    return psi
end # function HO_wfs1D

end # module