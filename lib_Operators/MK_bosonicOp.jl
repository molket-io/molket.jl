module MK_bosonicOp

# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the bosonic operators
# used in the quantum molecular dynamics simulations
# and the quantum algorithms

# Author: TSelim April 2024

using LinearAlgebra

#export a, adag, n, commutator
export a_Op, adag_Op, n_Op

function aOp(Nq::Int)
    # Create annihilation operator
    # Inputs
    # Nq : number of qubits, or the number of modes
    # ...  check the latex notes for more details.
    # Outputs
    # a : annihilation operator
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator 
    # and the bosonic operators: https://arxiv.org/abs/1805.09928 
    # TSelim April 4th 2024
    dim = 2^Nq
    a = zeros(Complex{Float64},dim,dim)
    for i in 1:dim-1
        a[i,i+1] = sqrt(i)
    end
    return a
end # function aOp

function adag_Op(Nq::Int)
    # Create creator operator
    # Inputs
    # Nq : number of qubits, or the number of modes
    # ...  check the latex notes for more details.
    # Outputs
    # a_dag : creator operator
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator
    # and the bosonic operators: https://arxiv.org/abs/1805.09928
    # TSelim April 4th 2024
    dim = 2^Nq
    a_dag = zeros(Complex{Float64},dim,dim)
    for i in 1:dim-1
        a_dag[i+1,i] = sqrt(i)
    end
    return a_dag
end # function adag_Op

function n_Op(Nq::Int)
    # Create number operator
    # Inputs
    # Nq : number of qubits, or the number of modes
    # ...  check the latex notes for more details.
    # Outputs
    # n : number operator
    # Reference:
    # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
    # Also, check the latex notes for more details.
    # This reference is a good source for the quantum harmonic oscillator
    # and the bosonic operators: https://arxiv.org/abs/1805.09928
    # TSelim April 4th 2024
    dim = 2^Nq
    n = zeros(Complex{Float64},dim,dim)
    for i in 1:dim
        n[i,i] = i-1
    end
    return n
end # function n_Op

end # module