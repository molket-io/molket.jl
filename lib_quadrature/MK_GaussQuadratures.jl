module MK_GaussQuadratures
## Module to compute the Gauss quadratures with the associated weights
# Docs
# © MolKet 2023, MIT License
# www.molket.io
# A module defining the Gauss quadratures
# used in the quantum molecular dynamics simulations
# and the quantum algorithms

using LinearAlgebra
using SpecialFunctions

include("../lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: laguerre, glaguerre

export guass_hermite_quadratures


# Gauss-Hermite quadrature
# Function to compute the Gauss-Hermite quadrature with the associated weights
# Inputs
function guass_hermite_quadratures(n::Int64,mu::Float64=1.0,omega::Float64=1.0,re::Float64=0.0)
    # Gauss-Hermite quadrature
    # Function to compute the Gauss-Hermite quadrature with the associated weights
    # Inputs
    # n     : degree
    # mu    : the mean of the Gaussian distribution
    # omega : the harmonic frequency
    # re    : the equilibrium distance/bond length
    # Outputs
    # x     : the points
    # w     : the weights
    # xx    : the points scaled by alpha = sqrt(mu*omega)
    # ww    : the weights scaled by alpha = sqrt(mu*omega)
    # Reference:
    # https://en.wikipedia.org/wiki/Gaussian_quadrature

    # Notes:
    #  Defaults: mu=1, omega=1, re=0
    # Here the points and weights are scaled:
    # α       = sqrt(MU*OMEGA)
    # grid   = re + XT/α
    # weight = WT/α

    #  This is exact for <i|r|j> for harmonic oscillator
    # functions for mass (mu) and frequence (omega)
    # Note: omega = sqrt(k/mu), for potential
    # 0.5 * k * (r-re)²
    # First compute the scaling factor
    alpha = sqrt(mu*omega)
    x     = diagm(1 => sqrt.(0.5*(1:n-1)))
    xt, U = eigen(x+x')
    # calculating the weights
    wt      = sqrt(pi)*(U[1,:]).^2 .* exp.(xt.^2)
    # scaling and shifting
    # sum re and xt
    xt2     = re .+ xt
    wt2     = wt/alpha
    return xt, wt, xt2, wt2
end # function guass_hermite_quadratures

# Gauss-Laguerre quadrature
# Function to compute the Gauss-Laguerre quadrature with the associated weights
function guass_laguerre_quadratures(n,alpha::Float64=0.0,adjusted::Bool=true)
    #G_LAGUERRE Gauss-Laguerre quadrature
    #  [XT, WT] = G_LAGUERRE(N) returns an N-point Gauss-Laguerre
    #  quadrature with "adjusted weights".
    #
    #  Note: requires:
    #
    #  lib_load('lib_eig')
    #
   #  See also: G_HERM for Gauss-Hermite
   #  https://keisan.casio.com/exec/system/1281279441


end # module