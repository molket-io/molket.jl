module MK_Hqladder

# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the expressions of normal coordinates in terms 
# of ladder operators

# Author: TSelim April 2024


# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

using LinearAlgebra
using SparseArrays


# call the lib_useful.jl with the def_matrix function which initializes a matrix
include("../lib_useful/custom_functions.jl")
using ..custom_functions: eye

# call the lib_SpecialPolynomials.jl
include("../lib_Operators/MK_bosonicOp.jl")
using ..MK_bosonicOp: a_Op, adag_Op


export Qcan_p1, Qcan_p2, Qcan_p3, Qcan_p4, H_harm
#export P_ladder

function Qcan_p1(N)
    # Normal coordinates (canonical/dimensionless) in terms of ladder operators
    # Inputs
    # N : number of qubits, or the number of modes
    # Outputs
    # Q : normal coordinate
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator
    # and the bosonic operators: https://arxiv.org/abs/1805.09928
    # TSelim April 30th 2024

    # Normal coordinate Q = (a + adag) / sqrt(2)
    # Construct a and adag operators
    a = a_Op(N)
    adag = adag_Op(N)
    Q = (a + adag) / sqrt(2)
    return Q
end # function Qcan_p1

function Qcan_p2(N)
    # Normal coordinates (canonical/dimensionless) in terms of ladder operators
    # Inputs
    # N : number of qubits, or the number of modes
    # Outputs
    # Q : normal coordinate
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator
    # and the bosonic operators: https://arxiv.org/abs/1805.09928
    # TSelim April May 2nd 2024

    # Normal coordinate Q = 0.5*(a_adag*a_dag + a_dag*a + a*a)
    # Construct a and adag operators
    a = a_Op(N)
    adag = adag_Op(N)
    Q = 0.5*(adag*adag + 2*adag*a + a*a)
    return Q
end # function Qcan_p2

# Qcan_p3 is Qcan ^ 3
function Qcan_p3(N)
    # Normal coordinates (canonical/dimensionless) in terms of ladder operators
    # Inputs
    # N : number of qubits, or the number of modes
    # Outputs
    # Q : normal coordinate
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator
    # and the bosonic operators: https://arxiv.org/abs/1805.09928
    # TSelim April May 6 2024
    
    # Construct a and adag operators
    a = a_Op(N)
    adag = adag_Op(N)
    Q = (adag_Op(N)^3 + 3*adag_Op(N)*a_Op(N)^2 
        + 3*adag_Op(N)^2*a_Op(N) + a_Op(N)^3)/2^(3/2)
    
    return Q
end # function Qcan_p3

# Qcan_p4 is Qcan ^ 4
function Qcan_p4(N)
    # Normal coordinates (canonical/dimensionless) in terms of ladder operators
    # Inputs
    # N : number of qubits, or the number of modes
    # Outputs
    # Q : normal coordinate
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator
    # and the bosonic operators: https://arxiv.org/abs/1805.09928
    # TSelim April May 6 2024
    
    # Construct a and adag operators
    a = a_Op(N)
    adag = adag_Op(N)

    Q =  (adag_Op(N)^4 + 4*adag_Op(N)^3*a_Op(N) 
         + 6*adag_Op(N)^2*a_Op(N)^2 + 4*adag_Op(N)*a_Op(N)^3 
         + a_Op(N)^4)/4
    return Q
end # function Qcan_p4

# Harmonic oscillator ladder operators
function H_harm(N;omega=1)
    # Harmonic oscillator ladder operators
    # Inputs
    # N : number of qubits, or the number of modes
    # Outputs
    # Hamiltonian : Harmonic oscillator Hamiltonian matrix in the harmonic oscillator basis
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # TSelim, May 9th 2024

    # Construct a and adag operators
    a = a_Op(N)
    adag = adag_Op(N)
    # Construct the Hamiltonian matrix
    Hamiltonian = omega *(adag*a + 0.5*eye(N))
    return Hamiltonian

end # function H_harm


end  # module MK_Hqladder