module quantum_gates
#module Qgate_create
########## Module quantum_gates ##########
# © MolKet 2023, MIT License.

# This module is part of MolKet.jl package.
## For more information, please visit out website: 
# www.molket.io 
# ----------------------------------------------------

## Documentations
# This module contains the constructor of the quantum gates.
# The quantum gates are constructed using the Julia's matrix structure.
# The quantum gates are stored in a structure called qgate.
# The structure qgate is exported from this module.
# The structure qgate contains the following gates:
# 1- Identity gate
# 2- Hadamard gate
# 3- Pauli-X gate
# 4- Pauli-Y gate
# 5- Pauli-Z gate
# 6- Phase gate
# 7- Pi/8 gate
# 8- Controlled Not gate
# 9- Controlled Y gate
# 10- Controlled Z gate
# 11- Controlled Controlled Not gate
# 12- Controlled Controlled Y gate
# 13- Controlled Controlled Z gate
# 14- Swap gate
# 15- Controlled Swap gate
# 16- U3 gate
# 17- Controlled U3 gate
# 18- Custom gate

# The structure qgate is exported from this module.
# The structure qgate is used in the quantum_circuit.jl module.

## This module is part of MolKet.jl package
# © Taha Selim and Alain Chancé 2023



using LinearAlgebra

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load quantum gates constructor")
#println("Using constants from file: ", module_file(constants))


# import quantum gates
include("conventions.jl")
using ..conventions: conventions

## only export the basic gates structure 
export Qgate

# Define the default error tolerance for checking the unitary condition
const err_tol = 1e-16


# Controlled gates constructor 
function ctrl_gate(gate)
# Controlled gate consutructor  
        r_mat = size(gate)[1]
        c_mat = size(gate)[2]
        # First check the gate is 2x2 or square matrix 
        if r_mat != c_mat
            error("The gate is not a square matrix")
        end
        # Construct Controlled gate
        # Define Identity matrix
        II = Matrix(I, r_mat, c_mat)
        M0 = zeros(r_mat, c_mat)
        return [I M0; M0 gate]
    end


# Create a function to construct a custom gate provided by the user
function user_gate(
    gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}; 
           err_tol::Float64=err_tol)
    # Custom gate constructor: custom_gate(gate)
    # gate is supplied by the user
    # it is assumed to be the basic gate of the minimum dimension
    # gate::Union{Array{Float64}, Array{Int64}, Array{ComplexF64}}: custom gate
    # rtol::Float64: relative tolerance for checking the unitary condition
    # return: custom gate
    # First check the gate is 2x2 or square matrix
    r_mat = size(gate)[1]
    c_mat = size(gate)[2]
    if r_mat != c_mat
        error("The gate is not a square matrix")
    end
    # check the unitary condition
    UU = gate'*gate
    II = Matrix(I, r_mat, c_mat)
    if isapprox(UU, II,rtol=err_tol) == false
        error("The gate is not unitary")
    end
    return gate
end # end custom_gate gate

function U_gate(theta::Float64, phi::Float64, lambda::Float64)
    # U_general_gate(theta, phi, lambda)
    # theta::Float64: angle of rotation around the x-axis
    # phi::Float64: angle of rotation around the y-axis
    # lambda::Float64: angle of rotation around the z-axis
    # return: U_general gate
    # Define the U_general gate
    U = [cos(theta/2) -exp(im*lambda)*sin(theta/2); 
         exp(im*phi)*sin(theta/2) exp(im*(phi+lambda))*cos(theta/2)]
    return U
end # end U_general_gate

function phase_gate(lambda::Float64)
    # phase_gate(lambda)
    # lambda::Float64: angle of rotation around the z-axis
    # return: phase gate
    # Formal definition of the phase gate
    # P(lambda) = u(0,0,lambda)
    # Define the phase gate
    P = U_gate(0.0,0.0,lambda)
    return P
end # end phase_gate

function rotation_gate(theta::Float64,pauli_gate::String)
    # rotation_gate(theta, pauli_axis)
    # theta::Float64: angle of rotation around the pauli_axis
    # pauli_axis::String: pauli axis of rotation
    # return: rotation gate
    # Define the rotation gate
    if pauli_gate == "x"
        Rx = U_gate(theta,-pi/2,pi/2)
        return Rx
    elseif pauli_gate == "y"
        Ry = U_gate(theta,0.0,0.0)
        return Ry
    elseif pauli_gate == "z"
        Rz = U_gate(0.0,0.0,theta)
        return Rz
    else
        error("The pauli axis is not defined")
    end
end # end rotation_gate

function Rx_gate(theta::Float64)
    # Rx_gate(theta)
    # theta::Float64: angle of rotation around the x-axis
    # return: Rx gate
    # Define the Rx gate
    Rx = U_gate(theta,-pi/2,pi/2)
    return Rx
end # end Rx_gate

function Ry_gate(theta::Float64)
    # Ry_gate(theta)
    # theta::Float64: angle of rotation around the y-axis
    # return: Ry gate
    # Define the Ry gate
    Ry = U_gate(theta,0.0,0.0)
    return Ry
end # end Ry_gate

function Rz_gate(theta::Float64)
    # Rz_gate(theta)
    # theta::Float64: angle of rotation around the z-axis
    # return: Rz gate
    # Define the Rz gate
    Rz = U_gate(0.0,0.0,theta)
    return Rz
end # end Rz_gate

function C_U_gate(q_control::Int64,q_target::Int64, 
    theta::Float64, phi::Float64, lambda::Float64, gamma::Float64,
    big_endian::Bool=conventions.big_endian)
    # C_U_gate(U)
    # two-qubit gate applied on the second qubit if the first qubit is in 
    # the state |1>
    # Inputs 
    # U::Function: unitary gate
    # return: C_U gate
    # Define the C_U gate
    # Define the dimension of the unitary gate

    # evaluate the Unitary gate U 
    # References: 
    # https://qiskit.org/documentation/stable/0.26/tutorials/circuits/3_summary_of_quantum_operations.html
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.CUGate.html
    # Define the C_U gate
    # check the convention of the qubits
    if big_endian == true
        # big-endian convention
        # |LSB .... MSB> = |q_control q_target>
        # Define the C_U gate
        if q_control < q_target
           C_U = [1 0 0 0; 
           0 1 0 0; 
           0 0 exp(im*gamma)*cos(theta/2) -exp(im*(gamma+lambda))*sin(theta/2);
           0 0 exp(im*(gamma+phi))*sin(theta/2) exp(im*(gamma+phi+lambda))*cos(theta/2)]
        return C_U
        end # end if statement for the control and target qubits
        if q_control > q_target
           C_U = [1 0 0 0; 
           0 exp(im*gamma)*cos(theta/2) 0 -exp(im*(gamma+lambda))*sin(theta/2);
           0 0 1 0; 
           0 0 exp(im*(gamma+phi))*sin(theta/2) exp(im*(gamma+phi+lambda))*cos(theta/2)]
        return C_U
        end # end if statement for the control and target qubits
    else # little-endian convention
        # |MSB .... LSB> = |q_control q_target>
        # little-endian convention
        # Define the C_U gate
        if q_control < q_target           
           C_U = [1 0 0 0; 
           0 exp(im*gamma)*cos(theta/2) 0 -exp(im*(gamma+lambda))*sin(theta/2);
           0 0 1 0; 
           0 0 exp(im*(gamma+phi))*sin(theta/2) exp(im*(gamma+phi+lambda))*cos(theta/2)]
        return C_U
        end # end if statement for the control and target qubits
        if q_control > q_target
           C_U = [1 0 0 0; 
           0 1 0 0; 
           0 0 exp(im*gamma)*cos(theta/2) -exp(im*(gamma+lambda))*sin(theta/2);
           0 0 exp(im*(gamma+phi))*sin(theta/2) exp(im*(gamma+phi+lambda))*cos(theta/2)]
        return C_U
        end # end if statement for the control and target qubits
    end
    
end # end C_U_gate

function CX_gate(big_endian::Bool=conventions.big_endian)
    # CX_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in 
    # the state |1>
    # Inputs 
    # return: CX gate
    # Define the CX gate
    # check the convention of the qubits
    if big_endian == true
        # big-endian convention
        # |LSB .... MSB> = |q_control q_target>
        # Define the CX gate
        CX =  ctrl_gate([0 1; 
                         1 0])
        return CX
    else # little-endian convention
        # |MSB .... LSB> = |q_control q_target>
        # little-endian convention
        # Define the CX gate
        CX = [1 0 0 0; 
              0 0 0 1; 
              0 0 1 0; 
              0 1 0 0]
        return CX
    end
    
end # end CX_gate

#### Define the structure of the quantum gates #################
## export the functions in a structure 
Base.@kwdef mutable struct qgate
    ### Define single qubit gates ###
    # Identity gate
        II::Array{Float64,2}
    # Define a general single qubit gate: a unitary gate U of dimension 2x2
        U::Function
    # Define general P (phase) gate of dimension 2x2
        P::Function
    ## Define Pauli gates ## 
    # Pauli-X gate
        X::Array{Float64,2}
    # Pauli-Y gate
        Y::Array{Float64,2}
    # Pauli-Z gate
        Z::Array{Float64,2}
    ## Define Clifford gates ##
    # Hadamard gate
        H::Array{Float64,2}
    # Phase gate (Pi/2 gate)
        S::Array{Float64,2}
    # S_dag gate equivalent to sqrt(Z) gate
    # ... which is the conjugate transpose of S gate
    # ... and it is equivalent to P(-Pi/2) gate
        S_dag::Array{Float64,2}
    ### Define C3 gates ###
    # Define T gate or sqrt(S) equivalent to P(Pi/4) gate
        T::Array{Float64,2}
    # Define T_dag gate or sqrt(S_dag) equivalent to P(-Pi/4) gate
        T_dag::Array{Float64,2}
    ## Define standard rotation C2 gates ##
    # Define general rotation gate around the arbitrary Pauli gate
    # ... of dimension 2x2
    # The standard rotation gates are those that define rotations around 
    # the Paulis P = {X, Y, Z} gates
        Rp::Function
     # Define Rx gate
        Rx::Array{Float64,2}
    # Define Ry gate
        Ry::Array{Float64,2}
    # Define Rz gate
        Rz::Array{Float64,2}
    ### Define multi qubit gates ###
    ## two qubit gates ##
    # Define general Control gate of dimension 4x4 applied on two qubits 
    # ... where the first qubit is the control qubit and the second qubit 
    # ... is the target qubit
        C_U::Function
    # Define Controlled Not gate: definition depends on the convention used
    # ... "here we use the convention "big endien" or "little endien".
    # The controlled gate is applied to the second qubit
        CX::Array{Float64,2}
    # Define Controlled Y gate
        CY::Array{Float64,2}
    # Define Controlled Z gate
        CZ::Array{Float64,2}

end # end qgate structure


Qgate = qgate(
## Define single qubit gates
    # Identity gate
    II= Matrix(I,2,2), 
    # Define a general single qubit gate: a unitary gate U of dimension 2x2
    U=U_gate,
    # Define general P (phase) gate of dimension 2x2
    P=phase_gate,
    ## Define Pauli gates ##
    # Pauli-X gate: equivalent to U_gate(Pi,0,Pi)
    X=[0 1; 
       1 0],
    # Pauli-Y gate: equivalent to U_gate(Pi,Pi/2,Pi/2)
    Y=[0 -im; 
       im 0],
    # Pauli-Z gate: equivalent to phase_gate(Pi)
    Z=[1 0; 
       0 -1]
    ## Define Clifford gates ##
    # Hadamard gate: equivalent to U_gate(Pi/2,0,Pi)
    H=[1 1; 
       1 -1],
    # Phase gate (Pi/2 gate)
    S=[1 0; 
       0 im],
    # S_dag gate equivalent to sqrt(Z) gate
    # ... which is the conjugate transpose of S gate
    # ... and it is equivalent to P(-Pi/2) gate
    S_dag=[1 0; 
           0 -im]
    ### Define C3 gates ###
    # Define T gate or sqrt(S) equivalent to P(Pi/4) gate
    T=[1 0; 
       0 exp(im*pi/4)],
    # Define T_dag gate or sqrt(S_dag) equivalent to P(-Pi/4) gate
    T_dag=[1 0; 
           0 exp(-im*pi/4)],
    ## Define standard rotation C2 gates ##
    # Define general rotation gate around the arbitrary Pauli gate
    # ... of dimension 2x2
    # The standard rotation gates are those that define rotations around
    # the Paulis P = {X, Y, Z} gates
    Rp=rotation_gate,
    # Define Rx gate
    Rx=Rx_gate,
    # Define Ry gate
    Ry=Ry_gate,
    # Define Rz gate
    Rz=Rz_gate,
    ### Define multi qubit gates ###
    ## two qubit gates ##
    # Define general Control gate of dimension 4x4 applied on two qubits
    C_U=C_U_gate,
)

end # end module quantum_gates