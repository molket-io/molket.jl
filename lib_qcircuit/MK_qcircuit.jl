
module MK_qcircuit
# construct the quantum circuit for the quantum algorithm 

# import LinearAlgebra and SparseArrays
using LinearAlgebra # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
using SparseArrays # https://docs.julialang.org/en/v1/stdlib/SparseArrays/

# import conventions
include("../conventions.jl")
  using .conventions: big_endian
# import quantum gates
include("../quantum_gates.jl")
using .quantum_gates: Qgate

function qc_initialize0(nqubit::Int64, big_endian::Bool=conventions.big_endian)
# initialize the quantum register to |0>^nqubit
# input: 
#   nqubit: number of qubits
#   big_endian: true if the qubit register is in big endian convention
# output:
#   qc: quantum circuit initialized to |0>^nqubit
#   qubit_start: starting qubit index



    return 1
end # function qc_initialize


end # module qcircuit