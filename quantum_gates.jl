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

println("Load quantum gates constructor")

# import quantum gates
include("conventions.jl")
using .conventions: conventions

# Function module file looks up the location of a module from the method table
# Reference: https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

if conventions.verbose1
    println("Using conventions from file: ", module_file(conventions))
end

## only export the basic gates structure 
export qgate, Qgate, Rz_gate1

# Define the default error tolerance for checking the unitary condition
const err_tol = 1e-16


## Define the functions to construct the quantum gates ###
###################################################################################
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
    U = [cos(theta/2) -exp(1im*lambda)*sin(theta/2); 
         exp(1im*phi)*sin(theta/2) exp(1im*(phi+lambda))*cos(theta/2)]
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

function Rz_gate1(phi::Float64)
    # Rphi_gate(phi)
    # phi::Float64: angle of rotation around the z-axis
    # return: Rphi gate
    # a custom gate definition based on the matrix represrenation of 
    # ... the rotation around the z-axis mentioned in the following paper 
    # arXiv:2209.08187v1 [quant-ph] 16 Sep 2022
    Rz = [
        exp(-im*phi/2) 0;
        0              exp(im*phi/2)
    ]
    return Rz
end # end Rphi_gate

function C_U_gate(q_control::Int64,q_target::Int64, 
    theta::Float64, phi::Float64, lambda::Float64, gamma::Float64,
    big_endian::Bool=conventions.big_endian)
    # C_U_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # theta::Float64: angle of rotation around the x-axis
    # phi::Float64: angle of rotation around the y-axis
    # lambda::Float64: angle of rotation around the z-axis
    # gamma::Float64: angle of rotation around the control qubit
    # big_endian::Bool: convention of the qubits, by default is true
    # return: C_U gate

    # evaluate the Unitary gate U 
    # References: 
    # https://qiskit.org/documentation/stable/0.26/tutorials/circuits/3_summary_of_quantum_operations.html
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.CUGate.html
    # Define the C_U gate
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
   # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the C_U gate
        C_U = [1 0 0 0; 
        0 1 0 0; 
        0 0 exp(im*gamma)*cos(theta/2) -exp(im*(gamma+lambda))*sin(theta/2);
        0 0 exp(im*(gamma+phi))*sin(theta/2) exp(im*(gamma+phi+lambda))*cos(theta/2)]
        return C_U
    else
        C_U = [1 0 0 0; 
        0 exp(im*gamma)*cos(theta/2) 0 -exp(im*(gamma+lambda))*sin(theta/2);
        0 0 1 0; 
        0 0 exp(im*(gamma+phi))*sin(theta/2) exp(im*(gamma+phi+lambda))*cos(theta/2)]
        return C_U
    end # end if statement for the control and target qubits

end # end C_U_gate


function CX_gate(q_control::Int64,q_target::Int64, 
    big_endian::Bool=conventions.big_endian)
    # CX_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # big_endian::Bool: convention of the qubits, by default is true
    # return: CX gate
    # check whether the control and target qubits are different
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
        # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the CX gate
        CX =  ctrl_gate([0 1;
                         1 0])
        return CX
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CX gate
        CX = [1 0 0 0;
              0 0 0 1;
              0 0 1 0;
              0 1 0 0]
        return CX
    end # end if statement for the control and target qubits
    end # end CX_gate

function CY_gate(q_control::Int64,q_target::Int64, 
    big_endian::Bool=conventions.big_endian)
    # CY_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # big_endian::Bool: convention of the qubits, by default is true
    # return: CY gate
    # check whether the control and target qubits are different
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
        # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the CY gate
        CY =  ctrl_gate([0 -1im;
                         1im 0])
        return CY
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CY gate
        CY = [1 0 0   0;
              0 0 0 -1im;
              0 0 1   0;
              0 1im 0 0]
        return CY
    end # end if statement for the control and target qubits
    end # end CY_gate


function CZ_gate()
    # CZ_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in 
    # the state |1>
    # Inputs 
    # return: CZ gate
    # Important note: the CZ gate is symmetric and it is the same in both 
    # ... conventions and it is the same regardless of the control qubit 
    # ... position, being the LSB or the MSB.
    CZ = ctrl_gate([1 0; 
                     0 -1])
    return CZ
end # end CZ_gate


function CH_gate(big_endian::Bool=conventions.big_endian)
    # C_H(): controlled Hadamard gate
    # two-qubit gate applied on the second qubit if the first qubit is in 
    # the state |1>
    # Inputs 
    # return: CH gate
    #check whether the control and target qubits are different
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target) 
        # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the CH gate
        CH = [1 0 0 0; 
              0 1 0 0; 
              0 0 1/sqrt(2) 1/sqrt(2);
              0 0 1/sqrt(2) -1/sqrt(2)]
        return CH
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CH gate
        CH = [1 0 0 0; 
              0 1/sqrt(2) 0 1/sqrt(2); 
              0 0 1 0;
              0 1/sqrt(2) 0 -1/sqrt(2)]
        return CH
    end # end if statement for the control and target qubits
end # end CH_gate

## Controlled Rotation gates ##
function CRz_gate(q_control::Int64,q_target::Int64,lambda::Float64,
    big_endian::Bool=conventions.big_endian)
    # CRz_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # lambda::Float64: angle of rotation around the z-axis
    # big_endian::Bool: convention of the qubits, by default is true
    # return: CRz gate
    # check whether the control and target qubits are different
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
        # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the CRz gate
        CRz = [1 0 0 0; 
               0 1 0 0; 
               0 0 exp(im*lambda/2) 0;
               0 0 0 exp(im*lambda/2)]
        return CRz
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CRz gate
        CRz = [1 0 0 0; 
               0 exp(-im*lambda/2) 0 0; 
               0 0 1 0;
               0 0 0 exp(im*lambda/2)]
        return CRz
    end # end if statement for the control and target qubits
    end # end CRz_gate

function CRp_gate(q_control::Int64,q_target::Int64,lambda::Float64,
    big_endian::Bool=conventions.big_endian)
    # CRp_gate()
    # Controlled phase rotation gate 
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # lambda::Float64: angle of rotation of the phase if both qubits are in 
    # ... the state |11>
    # big_endian::Bool: convention of the qubits, by default is true
    # return: CRp gate
    # The matrix representation of the CRp gate looks the same regardless of 
    # ... the convention used and the position of the control qubit.
    CRp = [1 0 0 0; 
           0 1 0 0; 
           0 0 1 0;
           0 0 0 exp(im*lambda)]
    return CRp
end # end CRp_gate

function CU_gate(q_control::Int64,q_target::Int64,
    theta::Float64, phi::Float64, lambda::Float64,
    big_endian::Bool=conventions.big_endian)
    # CU_gate()
    # Controlled U rotation gate
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # theta::Float64: angle of rotation around the x-axis
    # phi::Float64: angle of rotation around the y-axis
    # lambda::Float64: angle of rotation around the z-axis
    # big_endian::Bool: convention of the qubits, by default is true
    # return: CU gate
    # --> definitions of the angles need to be checked.
    # check whether the control and target qubits are different
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
        # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the CU gate
        CU = [1 0 0 0; 
              0 1 0 0; 
              0 0 exp(-im*(phi+lambda)/2)*cos(theta/2) -exp(-im*(phi-lambda)/2)*sin(theta/2);
              0 0 exp(-im*(phi-lambda)/2)*sin(theta/2) exp(im*(phi+lambda)/2)*cos(theta/2) ]
        return CU
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CU gate
        CU = [1 0 0 0; 
              0 exp(-im*(phi+lambda)/2)*cos(theta/2) 0 -exp(-im*(phi-lambda)/2)*sin(theta/2);
              0 0 1 0; 
              0 exp(-im*(phi-lambda)/2)*sin(theta/2) 0 exp(im*(phi+lambda)/2)*cos(theta/2) ]
        return CU
    end # end if statement for the control and target qubits
    end # end CU_gate


function Swap_gate(q_a::Int64,q_b::Int64,
    big_endian::Bool=conventions.big_endian)
    # CSwap_gate()
    # two-qubit type gate 
    # The SWAP gate exhanges two qubits. It transforms the basis vectors as 
    # ... |q_a q_b> --> |q_b q_a>.
    # Examples: |01> --> |10>, |10> --> |01>, |00> --> |00>, |11> --> |11>.
    # Inputs
    # q_a::Int64: first qubit
    # q_b::Int64: second qubit
    # big_endian::Bool: convention of the qubits, by default is true
    # return: Swaped basis vectors.
    
    # check whether the two qubits are different
    if q_a == q_b
        error("The two qubits are the same, the SWAP gate is not applied")
    else
        # Define the Swap gate
        Swap = [1 0 0 0; 
                0 0 1 0; 
                0 1 0 0;
                0 0 0 1]
        return Swap
    end # end if statement for the control and target qubits
    end # end Swap_gate

function XX_gate(theta::Float64,big_endian::Bool=conventions.big_endian)
    # XX_gate()
    # two-qubit type gate 
    # The XX gate is a two-qubit gate that is equivalent to the product of 
    # ... two Pauli-X gates.
    # Inputs
    # theta::Float64: angle of rotation around the x-axis
    # return: XX gate
    # Define the XX gate
    if big_endian
        XX = [cos(pi*theta/2) 0 0 -im*sin(pi*theta/2); 
              0 cos(pi*theta/2) -im*sin(pi*theta/2) 0; 
              0 -im*sin(pi*theta/2) cos(pi*theta/2) 0;
              -im*sin(pi*theta/2) 0 0 cos(pi*theta/2)]
        return XX
    else
        error("The little-endian convention is not implemented yet")
    end # end if statement
end # end XX_gate

function YY_gate(theta::Float64,big_endian::Bool=conventions.big_endian)
    # YY_gate()
    # two-qubit type gate 
    # The YY gate is a two-qubit gate that is equivalent to the product of 
    # ... two Pauli-Y gates.
    # Inputs
    # theta::Float64: angle of rotation around the y-axis
    # return: YY gate
    # Define the YY gate
    if big_endian
        YY = [cos(pi*theta/2) 0 0 im*sin(pi*theta/2); 
              0 cos(pi*theta/2) -im*sin(pi*theta/2) 0; 
              0 -im*sin(pi*theta/2) cos(pi*theta/2) 0;
              im*sin(pi*theta/2) 0 0 cos(pi*theta/2)]
        return YY
    else
        error("The little-endian convention is not implemented yet")
    end # end if statement
end # end YY_gate

function ZZ_gate(theta::Float64,big_endian::Bool=conventions.big_endian)
    # ZZ_gate()
    # two-qubit type gate 
    # The ZZ gate is a two-qubit gate that is equivalent to the product of 
    # ... two Pauli-Z gates.
    # Inputs
    # theta::Float64: angle of rotation around the z-axis
    # return: ZZ gate
    # Define the ZZ gate
    if big_endian
        ZZ = [exp(-im*pi*theta/2) 0 0 0; 
              0 exp(im*pi*theta/2)  0 0; 
              0 0 exp(im*pi*theta/2)  0;
              0 0 0 exp(-im*pi*theta/2)]
        return ZZ
    else
        error("The little-endian convention is not implemented yet")
    end # end if statement
end # end ZZ_gate



## three qubit gates ##
########################################

function CSwap_gate()
    # CSwap_gate()
    # Controlled Swap gate: called Fredkin gate.
    # three-qubit type gate 
    # The CSWAP gate exhanges the second and the third qubits if the first
    # ... qubit is in the state |1>.
    # ... |q_a q_b q_c> --> |q_a q_c q_b>.
    # Examples: |010> --> |001>, |100> --> |100>, |000> --> |000>, |111> --> |111>.
    # Inputs
    # return: Swaped basis vectors.
    # Notes: the CSWAP gate should be the same in both conventions.

    # Define the CSwap gate
    CSwap = [1 0 0 0 0 0 0 0; 
             0 1 0 0 0 0 0 0; 
             0 0 1 0 0 0 0 0;
             0 0 0 1 0 0 0 0;
             0 0 0 0 1 0 0 0;
             0 0 0 0 0 0 1 0;
             0 0 0 0 0 1 0 0;
             0 0 0 0 0 0 0 1]
    return CSwap
end # end CSwap_gate

function CCX_gate(q0_control::Int64,q1_control::Int64,q_target::Int64,
    big_endian::Bool=conventions.big_endian)
    # CCX_gate()
    # Controlled Controlled X gate: called Toffoli gate.
    # three-qubit type gate 
    # The CCX gate exhanges the second and the third qubits if the first
    # ... qubit is in the state |1>.
    # ... |q0_control q1_control q_target> --> |q0_control q1_control q_target xor q0_control q1_control>.
    # Examples: |010> --> |001>, |100> --> |100>, |000> --> |000>, |111> --> |111>.
    # Inputs
    # return: Swaped basis vectors.
    # Notes: the CCX gate should be the same in both conventions.
    # check whether the control and target qubits are different
    if q0_control == q_target || q1_control == q_target || q0_control == q1_control
        error("The control and target qubits are the same")
    elseif (big_endian && q0_control < q1_control && 
        q0_control < q_target && q1_control < q_target) || 
        (!big_endian && q0_control > q1_control && 
        q0_control > q_target && q1_control > q_target)
    # Define the CCX gate
        CCX = [1 0 0 0 0 0 0 0; 
               0 1 0 0 0 0 0 0; 
               0 0 1 0 0 0 0 0;
               0 0 0 1 0 0 0 0;
               0 0 0 0 1 0 0 0;
               0 0 0 0 0 1 0 0;
               0 0 0 0 0 0 0 1;
               0 0 0 0 0 0 1 0]
        return CCX
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CCX gate
        CCX = [1 0 0 0 0 0 0 0; 
               0 1 0 0 0 0 0 0; 
               0 0 1 0 0 0 0 0;
               0 0 0 0 0 0 0 1;
               0 0 0 0 1 0 0 0;
               0 0 0 0 0 1 0 0;
               0 0 0 0 0 0 1 0;
               0 0 0 1 0 0 0 0]
        return CCX
    end # end if statement for the control and target qubits
end # end CCX_gate


################################################################
#### Define the structure of the quantum gates #################
################################################################

## export the functions in a structure 
Base.@kwdef struct qgate
    ### Define single qubit gates ###
    # Identity gate
        II::Array{Float64,2} = Matrix(I,2,2)
    # Define a general single qubit gate: a unitary gate U of dimension 2x2
        U::Function = U_gate
    # Define general P (phase) gate of dimension 2x2
        P::Function = phase_gate
    ## Define Pauli gates ## 
    # Pauli-X gate
        X::Array{Float64,2} = [0 1; 1 0]
    # Pauli-Y gate
        Y::Array{ComplexF64,2} = [0 -1im; 1im 0]
    # Pauli-Z gate
        Z::Array{Float64,2} = [1 0; 0 -1]
    ## Define Clifford gates ##
    # Hadamard gate
        H::Array{Float64,2} =(1/sqrt(2))* [1 1; 1 -1]
    # Phase gate (Pi/2 gate)
        S::Array{ComplexF64,2} = [1 0; 0 1im]
    # S_dag gate equivalent to sqrt(Z) gate
    # ... which is the conjugate transpose of S gate
    # ... and it is equivalent to P(-Pi/2) gate
        S_dag::Array{ComplexF64,2} = [1 0; 0 -1im]
    ### Define C3 gates ###
    # Define T gate or sqrt(S) equivalent to P(Pi/4) gate
        T::Array{ComplexF64,2} = [1 0; 0 exp(im*pi/4)]
    # Define T_dag gate or sqrt(S_dag) equivalent to P(-Pi/4) gate
        T_dag::Array{ComplexF64,2} = [1 0; 0 exp(-im*pi/4)]
    ## Define standard rotation C2 gates ##
    # Define general rotation gate around the arbitrary Pauli gate
    # ... of dimension 2x2
    # The standard rotation gates are those that define rotations around 
    # the Paulis P = {X, Y, Z} gates
        Rp::Function = rotation_gate
     # Define Rx gate
        Rx::Function = Rx_gate
    # Define Ry gate
        Ry::Function = Ry_gate
    # Define Rz gate
        Rz::Function = Rz_gate
    ### Define multi qubit gates ###
    ## two qubit gates ##
    # Define general Control gate of dimension 4x4 applied on two qubits 
    # ... where the first qubit is the control qubit and the second qubit 
    # ... is the target qubit
        C_U::Function = C_U_gate
    # Define Controlled Not gate: definition depends on the convention used
    # ... "here we use the convention "big endien" or "little endien".
    # The controlled gate is applied to the second qubit
        CX::Function = CX_gate
    # Define Controlled Y gate
        CY::Function = CY_gate
    # Define Controlled Z gate
        CZ::Function = CZ_gate
    # Define Controlled Hadamard gate
        CH::Function = CH_gate
    ## Controlled Rotation gates ##
    # Define Controlled Rz gate
        CRz::Function = CRz_gate
    # Define Controlled phase rotation gate
        CRp::Function = CRp_gate
    # Define Controlled U gate
        CU::Function = CU_gate
    # Define Swap gate
        Swap::Function = Swap_gate
    ## Define Ising gates ##
        XX::Function = XX_gate
        YY::Function = YY_gate
        ZZ::Function = ZZ_gate
    ## three qubit gates ##
    # Define Controlled Controlled X gate
        CCX::Function = CCX_gate
    # Define Controlled Swap gate
        CSwap::Function = CSwap_gate
   
end # end qgate structure
    
Qgate = qgate()

end # end module quantum_gates