module quantum_gates
#module Qgate_create
########## Module quantum_gates ##########
# © Taha Selim and Alain Chancé 2023, MIT License.
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
# The structure qgate is exported from this module.
# The structure qgate is used in the quantum_circuit.jl module.

## This module is part of MolKet.jl package
# © Taha Selim and Alain Chancé 2023

using LinearAlgebra

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

println("Load quantum gates constructor")
#println("Using constants from file: ", module_file(constants))


## only export the basic gates structure 
export Qgate

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

function hadamard()
    # Hadamard gate constructor
    Norm ::Float64 = 1/sqrt(2)
    return Norm*[1 1; 1 -1]
end

## export the functions in a structure 

Base.@kwdef mutable struct qgate
# Identity gate
    II::Array{Float64,2}
# Hadamard gate
    H ::Array{Float64,2}  
# Pauli-X gate
    X ::Array{Float64,2}
# Pauli-Y gate
    Y::Array{Complex{Float64},2}
# Pauli-Z gate
    Z::Array{Float64,2}
# Phase gate
    S::Array{Complex{Float64},2}
# Pi/8 gate
    T::Array{Complex{Float64},2}
# Controlled Not gate
    CX::Array{Float64,2}
# Controlled Y gate
    CY::Array{Complex{Float64},2}
# Controlled Z gate
    CZ::Array{Float64,2}
# Controlled Controlled Not gate
    CCX::Array{Float64,2}
# Controlled Controlled Y gate
    CCY::Array{Complex{Float64},2}
# Controlled Controlled Z gate
    CCZ::Array{Float64,2}
# Swap gate
    SWAP::Array{Float64,2}
# Controlled Swap gate
    CSWAP::Array{Float64,2}

    end # end qgate structure

# normalization constant used in constructing the quantum gates 
# .. involving the superposition
Norm ::Float64 = 1/sqrt(2)


Qgate = qgate(
II= Matrix(I,2,2), 
H = Norm*[1 1; 
         1 -1],
X =      [0 1;
          1 0],
Y =      [0 -im;
          im 0],
Z =       [1 0;
          0 -1],
S =       [1 0;
           0 im],
T =        [1 0;
            0 exp(im*pi/4)], 
CX =       ctrl_gate([0 1;
                     1 0]),
CY =       ctrl_gate([0 -im;
                     im 0]),
CZ =       ctrl_gate([1 0;  
                     0 -1]),
CCX =      ctrl_gate(ctrl_gate([0 1;
                                1 0])),
CCY =      ctrl_gate(ctrl_gate([0 -im;
                                im 0])),
CCZ =      ctrl_gate(ctrl_gate([1 0;
                                0 -1])),
SWAP =     [1 0 0 0;
            0 0 1 0;
            0 1 0 0;
            0 0 0 1],
CSWAP =    ctrl_gate([1 0 0 0;
                      0 0 0 1;
                      0 0 1 0;
                      0 1 0 0]
                      ) 
) # end qgate structure input

end # end module 

########## End of module Qgate_create ##########

