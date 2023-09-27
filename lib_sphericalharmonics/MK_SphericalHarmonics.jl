module MK_SphericalHarmonics

using LinearAlgebra
using SpecialFunctions
using AssociatedLegendrePolynomials

export Ylm, Clm, Slm, Rlm

# Docs 
# Legendre and associate Legendre polynomials are defined in 
# https://juliapackages.com/p/legendrepolynomials
# call Pl(x,l) --> 
#        to compute the Legendre polynomials for a given argument x and a degree l
# call Plm(x,l,m) -->
#        to compute the associate Legendre polynomials for a given argument x, 
#        a degree l and an order m

## angle convention 
##*****************
## We use the physics convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system
## r = r  * sin(theta) cos(phi),
## y = r  * sin(theta)sin(phi),
## z = r * cos(theta).

## make a combination og bol, and choose harmonics 



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
    # Real (Tesseral) harmonics using Racah normalization convention with the phase convention
    # of Condon and Shortley -1^m
    #   R_LM(THETA,PHI) = -1^m sqrt((l-m)!/(l+m)!) 
    #        * Plm(cos(theta)) * cos(m*phi) for m > 0
    #   R_LM(THETA,PHI) = -1^m  sqrt((l-m)!/(l+m)!)
    #        * Plm(cos(theta)) * sin(m*phi) for m < 0
    #   R_LM(THETA,PHI) = Plm(cos(theta)) for m = 0
    # Reference: https://en.wikipedia.org/wiki/Spherical_harmonics

    #    Norm = sqrt((2l+1)/(4pi)) * sqrt(factorial(l-m)/factorial(l+m))    

    # legendre(LegendreSphereNorm(), l, abs(m), cos(theta)) is already using the full spherical 
    # harmonics normalization convention
    Norm_Racah = 1/sqrt((2l+1)/(4pi)) 
    if m == 0
        S_lm = Norm_Racah *legendre(LegendreSphereNorm(), l, abs(m), cos(theta))
    elseif m > 0
        S_lm = (-1)^m * Norm_Racah *sqrt(2)*legendre(LegendreSphereNorm(), l, abs(m), cos(theta))*cos(m*phi)
    else
        S_lm = (-1)^m * Norm_Racah *sqrt(2)*legendre(LegendreSphereNorm(), l, abs(m), cos(theta))*sin(abs(m)*phi)
    end # if
    return S_lm
end # function Rlm

function Rlm(l, m, theta, phi)
    # Real harmonics using full normalization convention with the phase convention
    # of Condon and Shortley -1^m
    #   R_LM(THETA,PHI) = -1^m sqrt((2l+1)/(4pi)) * sqrt((l-m)!/(l+m)!) 
    #        * Plm(cos(theta)) * cos(m*phi) for m > 0
    #   R_LM(THETA,PHI) = -1^m sqrt((2l+1)/(4pi)) * sqrt((l-m)!/(l+m)!)
    #        * Plm(cos(theta)) * sin(m*phi) for m < 0
    #   R_LM(THETA,PHI) = sqrt((2l+1)/(4pi)) * Plm(cos(theta)) for m = 0
    # Reference: https://en.wikipedia.org/wiki/Spherical_harmonics

#    Norm = sqrt((2l+1)/(4pi)) * sqrt(factorial(l-m)/factorial(l+m))    
    if m == 0
        R_lm = legendre(LegendreSphereNorm(), l, 0, cos(theta))
    elseif m > 0
        R_lm = (-1)^m * sqrt(2)*legendre(LegendreSphereNorm(), l, abs(m), cos(theta))*cos(m*phi)
    else
        R_lm = (-1)^m * sqrt(2)*legendre(LegendreSphereNorm(), l, abs(m), cos(theta))*sin(abs(m)*phi)
    end # if
    return R_lm
end # function Rlm


end # module