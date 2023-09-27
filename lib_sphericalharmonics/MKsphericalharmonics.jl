module MK_SphericalHarmonics

using LinearAlgebra
using SpecialFunctions
using AssociatedLegendrePolynomials

export Ylm, Clm, Slm

# Docs 
# Legendre and associate Legendre polynomials are defined in 
# https://juliapackages.com/p/legendrepolynomials
# call Pl(x,l) --> 
#        to compute the Legendre polynomials for a given argument x and a degree l
# call Plm(x,l,m) -->
#        to compute the associate Legendre polynomials for a given argument x, 
#        a degree l and an order m

# Compute the spherical harmonics in full normalization convention
function Ylm(l, m, theta, phi)
    # Complex spherical harmonics with full normalization convention
    # normalized to unity on the sphere
    # Ylm = -1^m sqrt((2l+1)/(4pi)) * sqrt((l-m)!/(l+m)!) 
    #        * Plm(cos(theta)) * exp(im*m*phi)
    # Reference: 
    #https://www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Spherical_harmonics.html
    # we use the convention that theta is the polar angle and phi is the azimuthal angle
    # theta is in the range [0,pi] and phi is in the range [0,2pi]
    # theta and phi are in radians
    # Inputs 
    # l: degree of the spherical harmonics
    # m: order of the spherical harmonics
    # theta: polar angle in radians
    # phi: azimuthal angle in radians
    # Outputs
    # Y_lm: complex spherical harmonics with full normalization convention
    # Note: the function required the package AssociatedLegendrePolynomials
    #       to be installed

    Y_lm = legendre(LegendreSphereNorm(), l, abs(m), cos(theta))*exp(1im*m*phi)
    return Y_lm
end # function

function Clm(l, m, theta, phi)
    # Complex spherical harmonics with the phase convention 
    # using Racah's normalization convention
    norm = sqrt((2l+1)/(4pi)) 
    C_lm = (1/norm)*Ylm(l,m,theta,phi)
    return C_lm
end # function Clm

function Slm(l, m, theta, phi)
    # Tesseral (real ) harmonics using Racah's normalization convention
    #   S_LM(THETA,PHI) =
    #  [C_LM(THETA,PHI)+(-1)^M C_L(-M)(THETA,PHI)]/sqrt(2)     for M > 0
    #          C_L0(THETA,0)                                   for M = 0
    #  [-i C_LM(THETA,PHI)+i (-1)^M C_L(-M)(THETA,PHI)]/sqrt(2) for M<0
    if m > 0
        S_lm = (Clm(l, m, theta, phi) + (-1)^m*Clm(l, -m, theta, phi))/sqrt(2)
    elseif m == 0
        S_lm = Clm(l, 0, theta, phi)
    elseif m < 0
        S_lm = (-1im*Clm(l, m, theta, phi) + 1im*(-1)^m*Clm(l, -m, theta, phi))/sqrt(2)
    end # if
    return S_lm
end # function Slm

end # module