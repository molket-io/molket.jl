module HO_wfs
# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the harmonic oscillator wave functions
# used in the quantum molecular dynamics simulations
# and the quantum algorithms

# Author: TSelim April 2024

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

# call the lib_SpecialPolynomials.jl
include("../lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: laguerre, glaguerre, ghermite, gchebhermite

using LinearAlgebra

export HO_wfs1D

function HO_wfs1D(n::Int64,x::Float64;omega::Float64=1.0,mu::Float64=1.0)
    # Harmonic oscillator wave functions
    # Inputs
    # n    : degree
    # x    : the point
    # omega: frequency
    # mu   : mass
    # Outputs
    # psi_n(x)
    # Reference:
    # https://en.wikipedia.org/wiki/Hermite_polynomials
    # https://en.wikipedia.org/wiki/Harmonic_oscillator
    # TSelim April 4th 2024

    # Note: the wavefunction is scaled by default to 1.0 so that the 
    # ... scaling factor alpha = sqrt(mu*omega) is not considered in the 
    # ... default case.
    psi = 0.0
    #psi = (mu*omega/(pi))^(1/4) * 1/sqrt(2^n * factorial(n)) 
    #* exp(-mu*omega*x^2/2) * ghermite(n,mu*omega*x^2)
    return psi
end # function HO_wfs1D

end # module