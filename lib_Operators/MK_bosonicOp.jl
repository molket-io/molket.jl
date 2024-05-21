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
export a_Op, adag_Op, n_Op, commutator, a_quOp, adag_quOp

function a_Op(Nq::Int)
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
    # TSelim April 30th 2024: I changed the size of the matrix to Nq instead of 2^Nq
    # ... to make it consistent with the number of qubits/modes.
    # Preparing the matrix for the annihilation operator that matches that size of the Hilbert space
    # ... of the qubits/modes is done in function a_quOp
    dim = Nq
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
    # TSelim April 30th 2024: I changed the size of the matrix to Nq instead of 2^Nq
    # ... to make it consistent with the number of qubits/modes.
    # Preparing the matrix for the creation operator that matches that size of the Hilbert space 
    # ... of the qubits/modes is done in function adag_quOp
    dim = Nq
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
    dim = Nq
    n = zeros(Complex{Float64},dim,dim)
    for i in 1:dim
        n[i,i] = i-1
    end
    return n
end # function n_Op

function commutator(A,B)
#    # Commutator of two operators
     # Inputs
        # A : operator A
        # B : operator B
        # Outputs
        # C : commutator of A and B
        # Reference:

        # TSelim April 30th 2024
        C = A*B - B*A
        return C
end # function commutator

# Make the bosonic operators qubit Hilbert space operators
function a_quOp(Nq::Int)
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
    # TSelim April 30th 2024
    dim = 2^Nq
    a = a_Op(dim)
    return a
end # function a_Op

# Make the bosonic operators qubit Hilbert space operators
function adag_quOp(Nq::Int)
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
    # TSelim April 30th 2024
    dim = 2^Nq
    a_dag = adag_Op(dim)
    return a_dag
end # function adag_Op


end # module