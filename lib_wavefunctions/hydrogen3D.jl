module hydrogen3D

using SpecialFunctions
using AssociatedLegendrePolynomials
using LinearAlgebra


include("../lib_sphericalharmonics/MKsphericalharmonics.jl")
using ..MK_SphericalHarmonics: Ylm, Clm, Slm

include("../lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: laguerre, glaguerre

export psi_2Dhydrogen, psi_3Dhydrogen

function psi_2Dhydrogen(n::Int64, m::Int64, r::Float64, theta::Float64)
    # 2D hydrogen atom wave function
    # Reference:
    # Hans Hon Sang Chan et al., 
    # Grid-based methods for chemistry simulations on a quantum computer. 
    #Sci. Adv.9, eabo7484 (2023).
    #DOI:10.1126/sciadv.abo7484,
    # https://www.science.org/doi/10.1126/sciadv.abo7484
    # Inputs
    # n: principal quantum number
    # m: magnetic quantum number
    # r: radial coordinate
    # theta: polar angle
    # Outputs
    # psi_2Dhydrogen: 2D hydrogen atom wave function
    # Eigs: eigenvalues of the 2D hydrogen atom
    # Eigs = -1/(2*(n+0.5)^2)
    # Note: the function required the package AssociatedLegendrePolynomials
    #       to be installed
    q0 = 1/(n+0.5)
    norm = sqrt((q0^3 * factorial(n - abs(m)))/(pi*factorial(n+abs(m))))
    fac_q0 = (2*q0*r)^abs(m)
    fac_L = MK_SpecialPolynomials.glaguerre(n-abs(m),2*abs(m),2*q0*r)
    fac_eq0 = exp(-q0*r)
    fac_th = exp(1im*m*theta)
    psi_2Dhydrogen = norm * fac_q0 * fac_L * fac_eq0 * fac_th

    Eigs = -1/(2*(n+0.5)^2)
    return psi_2Dhydrogen, Eigs
end # function psi_2Dhydrogen

function  psi_3Dhydrogen(n::Int64,l::Int64, m::Int64, r::Float64, 
                         theta::Float64, phi::Float64)
    # 3D hydrogen atom wave function
    # Reference:
    # Hans Hon Sang Chan et al.,
    # Grid-based methods for chemistry simulations on a quantum computer.
    #Sci. Adv.9, eabo7484 (2023).
    #DOI:10.1126/sciadv.abo7484,
    # https://www.science.org/doi/10.1126/sciadv.abo7484
    # Inputs
    # n: principal quantum number
    # m: magnetic quantum number
    # r: radial coordinate
    # theta: polar angle
    # phi: azimuthal angle
    # Outputs
    # psi_3Dhydrogen: 3D hydrogen atom wave function
    # Eigs: eigenvalues of the 3D hydrogen atom
    # Eigs = -1/(2*(n+0.5)^2)
    # Note: the function required the package AssociatedLegendrePolynomials 
    #       to be installed
    z = 1 # the center nuclear charge 
    norm1 = (2*z/n)^3
    norm2 = factorial(n-l-1)/(2*n*factorial(n+l))
    norm = sqrt(norm1*norm2)
    term_z = ((2*z*r)/n)^l
    term_ez = exp(-z*r/n)
    term_L = MK_SpecialPolynomials.glaguerre(n-l-1,2*l+1,2*z*r/n)
    term_Y = Ylm(l,m,theta,phi)
    psi_3Dhydrogen = norm*term_z*term_ez*term_L*term_Y
    Eigs = -1/(2*(n+0.5)^2)
    return psi_3Dhydrogen, Eigs
end # function psi_3Dhydrogen



end # module 3Dhydrogen