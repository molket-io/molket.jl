
## Import external packages/libraries
#########################################
using Plots # or StatsPlots
using LinearAlgebra
using SpecialFunctions
using AssociatedLegendrePolynomials
using LaTeXStrings
using Quantikz
using SparseArrays # https://docs.julialang.org/en/v1/stdlib/SparseArrays/


# import conventions
include("conventions.jl")
  using .conventions: little_endian, big_endian, qubit_begin, err_tol, show_op_mat, show_qc_mat

# import constants and unit conversions
include("lib_constants/const_data.jl")
using ..constants: cm1,NA,h_const

# import quantum gates
include("quantum_gates.jl")
using ..quantum_gates: q, Rz_gate1

include("lib_tensor/QTensor.jl")
using ..QTensor: q_T2D, q_T4D, tensor_mul

include("lib_useful/custom_functions.jl")
using ..custom_functions: MK_sortrows, eye

include("quantum_circuit.jl")
using ..quantum_circuit: qc_init, init_register, show_statevector, op, statevector
#using ..quantum_circuit: qc_init, init_register, print_initstate

include("lib_measurements/measurements.jl")
using ..measurements: z_measure, x_measure, peek_states, measure_state,
                      plot_bas4shots

# Load the special polynomials
include("lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: laguerre, glaguerre, ghermite, gchebhermite

# Load the bosonic operators
include("lib_Operators/MK_bosonicOp.jl")
using ..MK_bosonicOp: a_Op, adag_Op, n_Op

# Load the HamiltonianPrint
include("lib_Hamiltonian/MK_HamiltonianPrint.jl") 
using ..HamiltonianPrint: Hprint_fast, Evalues_print
