module measurements

using LinearAlgebra


# import conventions
include("../conventions.jl")
  using .conventions: big_endian

# import quantum gates
 include("../quantum_gates.jl")
using ..quantum_gates: q

# import the tensor library
include("../lib_tensor/QTensor.jl")
using ..QTensor: q_T2D, q_T4D

# import the quantum circuit library
include("../quantum_circuit.jl")
using .quantum_circuit: statevector

# import the sorting function
#include("../lib_useful/custom_functions.jl")
#using ..custom_functions: MK_sortrows


# Function module file looks up the location of a module from the method table
# Reference: https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

if conventions.verbose1
    println("Using conventions from file: ", module_file(measurements))
end

# export functions
export z_measure, x_measure, peek_states

function peek_states(qc)
    # Having a peek at the states of a quantum register in the computational basis
    # Inputs
    # ... qc: quantum circuit
    # Outputs
    # ... q_table: table of the quantum register
    # ... q_table[:,1]: basis
    # ... q_table[:,2]: statevector
    # ... q_table[:,3]: probability
    # It computes the probability of each state in the quantum register 
    # ... and returns the states sorted by the probability.
    # ==================================================================
    # Author: Taha Selim, Nov 20-21, 2025
    # ==================================================================
    # get the statevector of the quantum register
    q_states, state_vector, q_bits = statevector(qc)
    # get the proability of each state in the quantum register
    prob = abs.(state_vector).^2
    # get the index of the state with the highest probability
    ind_max = findmax(prob)[2]
    # get the state with the highest probability
    state_max = q_states[ind_max,:]
    # get the statevector, the basis, and the prob in a q_table
    q_table = Matrix(undef,qc.n_dim, 3)
    q_table[:,1] = q_bits # the basis
    q_table[:,2] = state_vector # the statevector
    q_table[:,3] = prob # the probability
    # sort the q_table by the probability
    q_ind = sortperm(prob)
    q_table = q_table[q_ind[:,1],:]
    # return the table of the quantum register
    return q_table
end # end measure_state

function z_measure(qc, qubit::Int; big_endian::Bool=true, show::Bool=true)
  # Measurements in the computational (z-)basis.
  # Depending on the convention, the number of qubits is counted
  # ... from the left or from the right
  # Inputs 
  # ... qc: quantum circuit
  # ... qubit: qubit to be measured
  # ... big_endian: convention to count the qubits
  # ... show: show the results
  # Outputs
  # ... qc: quantum circuit
  # ... sv_0: statevector after measuring the |0> state
  # ... sv_1: statevector after measuring the |1> state
  # ... p0: probability of measuring the |0> state
  # ... p1: probability of measuring the |1> state
  # So far, qc is not updated with one of the two possible outcomes
  # ... of the measurement
  # This is done in the sampler.
  # ==================================================================
  # Author: Taha Selim, Nov 15, 2025
  # check the arxiv notes made by Taha for the math behind the measurements
  # ================================================================== 
  nqubits = qc.n_qubits
  q_states = qc.q_states
  sv = qc.state_vector
  II = Matrix(I, 2, 2)
  # iInitiate the measurements operator
  P0 = 1
  P1 = 1
  ket0 = [1;0]
  ket1 = [0;1]
  # density operator of the |0> state
  rho0 = ket0*ket0'
  # density operator of the |1> state
  rho1 = ket1*ket1'
  if big_endian
      for i in 0:nqubits-1
          if i == qubit
              P0 = kron(P0,rho0)
              P1 = kron(P1,rho1) 
          else
              P0 = kron(P0,II)
              P1 = kron(P1,II)
          end
      end
  else
      for i in nqubits-1:-1:0
          if i == qubit
              P0 = kron(P0,rho0)
              P1 = kron(P1,rho1) 
          else
              P0 = kron(P0,II)
              P1 = kron(P1,II)
          end
      end
  end
  # New statevectors after the measurements
  sv_0 = P0 * sv
  sv_1 = P1 * sv
  # Calculate the probabilities
  p0 = norm(sv_0)^2
  p1 = norm(sv_1)^2
  # Normalize the new statevectors
  sv_0 = sv_0 / norm(sv_0)
  sv_1 = sv_1 / norm(sv_1)
  # Print the results
  if show
      println("The probability of measuring the |0> state is $p0")
      println("The probability of measuring the |1> state is $p1")
  end
  # Return the statevectors after the measurements
  return qc,sv_0,sv_1,p0,p1
end # function z_measure

function x_measure(qc, qubit::Int; big_endian::Bool=true, show::Bool=true)
    # Measurements in the x-basis.
    # Depending on the convention, the number of qubits is counted
    # ... from the left or from the right
    # Inputs 
    # ... qc: quantum circuit
    # ... qubit: qubit to be measured
    # ... big_endian: convention to count the qubits
    # ... show: show the results
    # Outputs
    # ... qc: quantum circuit
    # ... sv_plus: statevector after measuring the |+> state
    # ... sv_minus: statevector after measuring the |-> state
    # ... p_plus: probability of measuring the |+> state
    # ... p_minus: probability of measuring the |-> state
    # So far, qc is not updated with one of the two possible outcomes
    # ... of the measurement
    # This is done in the sampler. 
    # ==================================================================
  # Author: Taha Selim, Nov 15, 2025
  # check the arxiv notes made by Taha for the math behind the measurements
  # ================================================================== 
    nqubits = qc.n_qubits
    q_states = qc.q_states
    sv = qc.state_vector
    II = Matrix(I, 2, 2)
    # Initiate the measurements operator
    P_plus = 1
    P_minus = 1
    ket0 = [1;0]
    ket1 = [0;1]
    # density operator of the |+><+| state
    rho_plus = (1/2)*(ket0+ket1)*(ket0'+ket1')
    # density operator of the |-> state
    rho_minus = (1/2)*(ket0-ket1)*(ket0'-ket1')
    if big_endian
        for i in 0:nqubits-1
            if i == qubit
                P_plus = kron(P_plus,rho_plus)
                P_minus = kron(P_minus,rho_minus) 
            else
                P_plus = kron(P_plus,II)
                P_minus = kron(P_minus,II)
            end
        end
    else
        for i in nqubits-1:-1:0
            if i == qubit
                P_plus = kron(P_plus,rho_plus)
                P_minus = kron(P_minus,rho_minus) 
            else
                P_plus = kron(P_plus,II)
                P_minus = kron(P_minus,II)
            end
        end
    end
    # New statevectors after the measurements
    sv_plus = P_plus * sv
    sv_minus = P_minus * sv
    # Calculate the probabilities
    p_plus = norm(sv_plus)^2
    p_minus = norm(sv_minus)^2
    # Normalize the new statevectors
    sv_plus = sv_plus / norm(sv_plus)
    sv_minus = sv_minus / norm(sv_minus)
    # Print the results
    if show
        println("The probability of measuring the |+> state is $p_plus")
        println("The probability of measuring the |-> state is $p_minus")
    end
    # Return the statevectors after the measurements
    return qc,sv_plus,sv_minus,p_plus,p_minus
end # function x_measure


end # module