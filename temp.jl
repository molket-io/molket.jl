
function CX_gate(q_control::Int64,q_target::Int64, 
    big_endian::Bool=conventions.big_endian)
# CX_gate()
# two-qubit gate applied on the second qubit if the first qubit is in 
# the state |1>
# Inputs 
# return: CX gate
# check the convention of the qubits
if big_endian == true
# big-endian convention
# |LSB .... MSB> = |q_control q_target>
# Define the CX gate
if q_control < q_target
CX =  ctrl_gate([0 1; 
            1 0])
return CX
end # end if statement for the control and target qubits
if q_control > q_target
CX = [1 0 0 0; 
     0 0 0 1; 
     0 0 1 0; 
     0 1 0 0]
return CX
end # end if statement for the control and target qubits
else # little-endian convention
# |MSB .... LSB> = |q_control q_target>
# little-endian convention
# Define the CX gate
if q_control < q_target
CX = [1 0 0 0; 
     0 0 0 1; 
     0 0 1 0; 
     0 1 0 0]
return CX
end # end if statement for the control and target qubits
if q_control > q_target
CX =  ctrl_gate([0 1; 
                1 0])
return CX
end # end if statement for the control and target qubits
end
end # end CX_gate




function CY_gate(big_endian::Bool=conventions.big_endian)
    # CY_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in 
    # the state |1>
    # Inputs 
    # return: CY gate
    # check the convention of the qubits
    if big_endian == true
        # big-endian convention
        # |LSB .... MSB> = |q_control q_target>
        # Define the CY gate
        # if the control qubit is the LSB
        if q_control < q_target
            CY =  ctrl_gate([0 -1im; 
                             1im  0])
            return CY
        end # end if statement for the control and target qubits
        # if the control qubit is the MSB
        if q_control > q_target
            CY = [1 0   0  0; 
                  0 0   0 -1im; 
                  0 0   1  0; 
                  0 1im 0  0]
            return CY
        end # end if statement for the control and target qubits
    else # little-endian convention
        # |MSB .... LSB> = |q_control q_target>
        # little-endian convention
        # Define the CY gate
        # if the control qubit is the MSB
        if q_control > q_target
            CY =  ctrl_gate([0 1im; 
                             -1im 0])
            return CY
        end # end if statement for the control and target qubits
        # if the control qubit is the LSB
        if q_control < q_target
            CY = [1 0   0   0; 
                  0 0   0 -1im; 
                  0 0   1   0; 
                  0 1im 0   0]
            return CY
        end # end if statement for the control and target qubits
    end
end # end CY_gate



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


function CRy_gate(q_control::Int64,q_target::Int64,lambda::Float64,
    big_endian::Bool=conventions.big_endian)
    # CRy_gate()
    # two-qubit gate applied on the second qubit if the first qubit is in
    # the state |1>
    # Inputs
    # q_control::Int64: control qubit
    # q_target::Int64: target qubit
    # lambda::Float64: angle of rotation around the z-axis
    # big_endian::Bool: convention of the qubits, by default is true
    # return: CRy gate
    # check whether the control and target qubits are different
    if q_control == q_target
        error("The control and target qubits are the same")
    elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
        # Test for big-endian convention and the control qubit is the LSB 
        # |LSB .... MSB> = |q_control q_target>
        # OR 
        # Test for little-endian convention and the control qubit is the MSB
        # |MSB .... LSB> = |q_control q_target>
        # Define the CRy gate
        CRy = [1 0 0 0; 
               0 1 0 0; 
               0 0 cos(lambda/2) -sin(lambda/2);
               0 0 sin(lambda/2) cos(lambda/2)]
        return CRy
    else
        # Case: little-endian convention and the control qubit is the LSB
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_control q_target>
        # Define the CRy gate
        CRy = [1 0 0 0; 
               0 cos(lambda/2) 0 -sin(lambda/2); 
               0 0 1 0;
               0 sin(lambda/2) 0 cos(lambda/2)]
        return CRy
    end # end if statement for the control and target qubits
    end # end CRy_gate

    function CSwap_gate(q_control::Int64,q_target::Int64,q_swap::Int64,
        big_endian::Bool=conventions.big_endian)
        # CSwap_gate()
        # three-qubit gate applied on the third qubit if the first two qubits are 
        # ... in the state |11>
        # Inputs
        # q_control::Int64: control qubit
        # q_target::Int64: target qubit
        # q_swap::Int64: swap qubit
        # big_endian::Bool: convention of the qubits, by default is true
        # return: CSwap gate
        # check whether the control and target qubits are different
        if q_control == q_target
            error("The control and target qubits are the same")
        elseif (big_endian && q_control < q_target) || (!big_endian && q_control > q_target)
            # Test for big-endian convention and the control qubit is the LSB 
            # |LSB .... MSB> = |q_control q_target>
            # OR 
            # Test for little-endian convention and the control qubit is the MSB
            # |MSB .... LSB> = |q_control q_target>
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
        else
            # Case: little-endian convention and the control qubit is the LSB
            # This is qiskit convention. 
            # |MSB .... LSB> = |q_control q_target>
            # Define the CSwap gate
            CSwap = [1 0 0 0 0 0 0 0; 
                     0 1 0 0 0 0 0 0; 
                     0 0 1 0 0 0 0 0;
                     0 0 0 1 0 0 0 0;
                     0 0 0 0 1 0 0 0;
                     0 0 0 0 0 0 0 1;
                     0 0 0 0 0 0 1 0;
                     0 0 0 0 0 1 0 0]
            return CSwap
        end # end if statement for the control and target qubits
        end # end CSwap_gate
    


function Swap_gate(q_a::Int64,q_b::Int64,
    big_endian::Bool=conventions.big_endian)
    # CSwap_gate()
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
    elseif (big_endian && q_a < q_b) || (!big_endian && q_a > q_b)
        # Big-endien convention
        # |LSB .... MSB> = |q_a q_b>
        # OR 
        # Test for little-endian convention
        # |MSB .... LSB> = |q_a q_b>
        # Define the Swap gate
        Swap = [1 0 0 0; 
                0 0 1 0; 
                0 1 0 0;
                0 0 0 1]
        return Swap
    else
        # Case: little-endian convention
        # This is qiskit convention. 
        # |MSB .... LSB> = |q_a q_b>
        # Define the Swap gate
        Swap = [1 0 0 0; 
                0 0 0 1; 
                0 0 1 0;
                0 1 0 0]
        return Swap
    end # end if statement for the control and target qubits
    end # end Swap_gate


function CCX_gate()
    # CCX_gate()
    # Controlled Controlled X gate 
    # three-qubit type gate 
    # The CCX gate flips the third qubit if the first and the second
    # ... qubits are in the state |1>.
    # ... |q_a q_b q_c> --> |q_a q_c q_b>.
    # Examples: |010> --> |001>, |100> --> |100>, |000> --> |000>, |111> --> |111>.
    # Inputs
    # return: Swaped basis vectors.
    # Notes: the CCX gate should be the same in both conventions.
    # Only the sequence of applying the tensor operations is different.
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
end # end CCX_gate



# Compute the spherical harmonics in full normalization convention
function Ylm(l::Int64, m::Int64, theta::Float64, phi::Float64)
    norm = sqrt((2*l+1)/(4*pi)) * sqrt(factorial(l+m)/factorial(l-m))
    if m == 0
        return norm * Pl(cos(theta),l)
    elseif m > 0
        return norm * Plm(cos(theta),l,m) * cos(m*phi)
    else
        return norm * Plm(cos(theta),l,-m) * sin(-m*phi)
    end
end # function


# print the initial state of the quantum register
function print_initstate(qc)
    # print the initial state of the quantum register
    # qc::qc_initstruct: quantum register
    # return: print the initial state of the quantum register
    println("The initial state of the quantum register is: ")
    println(qc.state_vector)
    # print the initial state of the quantum register with the quantum 
    # states in the computational basis
    println("The initial state of the quantum register with the 
    quantum states in the computational basis is: ")
    q_table = zeros(Int, qc.n_dim, qc.n_qubits)
    for iq=1:qc.n_qubits
        for i in 1:qc.n_dim
            q_table[i,iq] = trunc(Int, qc.q_states[i,iq])
        end # end for
    end # end for


    for i in 1:qc.n_dim
        println( qc.q_states[i], " | ", string(q_table[i,:]), ">")
    end # end for
    #show(stdout, "text/plain", [qc.state_vector trunc(Int,qc.q_states)])
end # end print_initstate