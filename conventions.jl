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
little_endian::Bool = false
qubit_begin::Bool = false
qubit_start_1::Bool = false
show_op_mat::Bool= false # show the matrix of the operators
show_qc_mat::Bool= false # show the matrix of the quantum circuit

# Julia Machine epsilon
# https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Machine-epsilon
err_tol = 100.0*eps(Float64) #err_tol=2.22044e-14

# Seed for the Random module
# https://docs.julialang.org/en/v1/stdlib/Random/
# Random.seed! (seed)
seed = 1234

# Define the level of the printed information 
verbose::Int64 = 0 # 0: no print, 1: print, 2: print and plot
verbose0::Bool = verbose == 0
verbose1::Bool = verbose == 1
verbose2::Bool = verbose == 2


export big_endian, verbose, verbose0, verbose1, verbose2, little_endian,
show_matrix, err_tol, seed, show_qc_mat, show_op_mat

end # module conventions