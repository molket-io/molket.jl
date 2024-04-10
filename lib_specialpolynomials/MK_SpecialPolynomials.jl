module MK_SpecialPolynomials
# Docs
# © MolKet 2023, MIT License
# www.molket.io
# A module defining the special polynomials 
# used in the quantum molecular dynamics simulations 
# and the quantum algorithms

using SpecialFunctions


export laguerre, glaguerre, ghermite

function glaguerre(n::Int64,alpha::Int64,x::Float64)
    # Generalized Gauss-Laguerre polynomials
    # Function to evaluate associate Laguerre polynomials and 
    # binomial coefficients
    # L_n^(alpha)(x)
    # Inputs
    # n     : degree
    # alpha : arbitrary real
    # x     : the point
    # Outputs
    # L_n^(alpha)(x)
    # Reference:
    # Abramowitz, M. and Stegun, I. A. (Eds.). "Orthogonal Polynomials." Ch. 22
    # in Handbook of Mathematical Functions
    # with Formulas, Graphs, and Mathematical Tables,
    # 9th printing. New York: Dover, pp. 771-802, 1972.
    # Reference:
    # https://en.wikipedia.org/wiki/Laguerre_polynomials
    # Don't use high degree polynomials using this function to keep the numerical 
    # stability. We didn't test the function in this regime.
    LL = 0
    for m in 0:n
      fac1 = (-1)^m
      fac2 = gamma(n+alpha+1) / (gamma(n-m+1) * gamma(n+alpha-n+m+1))
      fac3 = 1 / factorial(m)
      LL += fac1 * fac2 * fac3 * x^m
    end
    return LL
end # function glaguerre  
    
function laguerre(n::Int64,x::Float64)
    # Gauss-Laguerre polynomials
    # TSelim 2023
    L = glaguerre(n,0,x)
    return L
end # function laguerre

# implement Hermite polynomials
function ghermite(n,x)
# Gauss-Hermite polynomials
# Inputs
# n : degree
# x : the point
# Outputs
# H_n(x)
# Reference:
# Abramowitz, M. and Stegun, I. A. (Eds.). "Orthogonal Polynomials." Ch. 22
# in Handbook of Mathematical Functions
# with Formulas, Graphs, and Mathematical Tables,
# 9th printing. New York: Dover, pp. 771-802, 1972.
# TSelim April 4th 2024
    if n == 0
        return 1
    elseif n == 1
        return 2*x
    else
        return 2*x*ghermite(n-1,x) - 2*(n-1)*ghermite(n-2,x)
    end
end # function ghermite

end # module