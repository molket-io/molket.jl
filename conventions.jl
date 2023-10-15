module conventions
# A module defining the conventions of the quantum circuit
# ... and the quantum gates
# ... and the classical registers
# ... and the quantum registers
# ... and the quantum states
# ... and the quantum operators
# ... and the quantum circuits
# ... and the quantum algorithms

# © MolKet 2023, MIT License
# www.molket.io

# Define the convention of the sequence of the qubits and the classical registers 
# ... in the quantum circuit
# choose between "big-endian" or "little-endian"
big_endian::Bool = true
qubit_begin::Bool = false
qubit_start_1::Bool = false

# Define the level of the printed information 
verbose::Int64 = 0 # 0: no print, 1: print, 2: print and plot
verbose0::Bool = verbose == 0
verbose1::Bool = verbose == 1
verbose2::Bool = verbose == 2


export big_endian, verbose, verbose0, verbose1, verbose2

end # module conventions