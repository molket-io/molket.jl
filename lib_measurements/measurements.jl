module measurements

using LinearAlgebra
using LaTeXStrings
using Plots # or StatsPlots


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
export z_measure, x_measure, peek_states, measure_state, plot_bas4shots, 
measure_qubits, find_states

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
end # end peek_states

function find_states(qc, qubits)
    # Inputs:
    # qc: quantum circuit
    # qubits: qubits to measure
    # Outputs:
    # states: states of the qubits in the computational basis
    # get the number of qubits to measure
    # Author: Taha Selim, June 10th, 2025
    n_qs = length(qubits)
    # get the number of qubits in the quantum circuit
    nqubits = qc.n_qubits
    # get the state vector
    sv = qc.state_vector
    # get the number of states
    nstates = length(sv)
    # get the number of states of the measured qubits
    nstates_q = 2^n_qs
    # get the number of states of the remaining qubits
    nstates_r = nstates/nstates_q
    # get the states of the measured qubits
    states = zeros(Int,nstates_q)
    for i in 1:nstates_q
        states[i] = i-1
    end
    return states
end # end of the function find_states

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

function measure_state(qc,shots;fig_format="png",save_fig=false)
    ## Execute the simulator 
    # Author: Taha Selim 
    # Date: 2023-12-27
    # get the qc structure, the statevectors, the basis,
    # ... and the number of qubits
    # the statevector is a complex vector of 2^n_qubits: state_vector
    # the basis is a vector of 2^n_qubits: q_states
    # the number of qubits is n_qubits
    q_states, state_vector, q_bits = statevector(qc)
    n_qubits = qc.n_qubits
    n_dim = qc.n_dim
    # probability density: prob, the square of the absolute value of the statevector
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
    # distribute the probabilities over a line interval from 0 to 1
    # The interval from 0 to 1 is divided into n_dim intervals
    # The length of each interval is the probability of the state
    # The interval is assigned to the state
    # write a for loop that distributes the probabilities over the interval from 0 to 1
    # the loop variable is i
    # the loop starts at 1 and ends at n_dim
    # Store the start and end of the interval in the prob_table
    # the prob_table has 5 columns
    prob_table = Matrix(undef,qc.n_dim, 5)
    prob_table[:,1] = q_table[:,1] # the basis
    prob_table[:,2] = q_table[:,2] # the statevector
    prob_table[:,3] = q_table[:,3] # the probability
    # the loop variable i is the index of the state
    start_interval = 0
    # Distribute the probabilities over the interval from 0 to 1
    for i in 1:n_dim
        # the length of the interval is the probability of the state
        length_interval = prob[i]
        # the end of the interval is the start of the interval plus the length of the interval
        end_interval = start_interval + length_interval
        # write the start and end of the interval into the q_table
        prob_table[i,4] = start_interval
        prob_table[i,5] = end_interval
        # the start of the next interval is the end of the current interval
        start_interval = end_interval  
    end
   # generate a vector of shots random numbers
   r = rand(shots)
   # locate each random number in the interval and store them in specific vectors 
   # Check whether a given r random number is contained in an interval of the prob_table and assign true or false
   # and store them in an array of booleans
   # the array of booleans is called r_in_interval
   state_count = zeros(Int64, n_dim)
   for i_interval in 1:n_dim
       interval = prob_table[i_interval,4:5]
       event_count = filter(t -> interval[1] < t < interval[2], r)
       state_count[i_interval] = length(event_count) 
   end
   # amend the q_table with the number of shots per state
   event_table = Matrix(undef,qc.n_dim, 4)
   event_table[:,1:3] = q_table[:,1:3]
   event_table[:,4] = state_count
   # sort the event_table by the number of shots
   event_ind = sortperm(state_count,rev=true)
   event_table = event_table[event_ind[:,1],:]
   plot_bas4shots(event_table,qc;
   fig_format=fig_format,save_fig=save_fig)
end # function measure_state


function plot_bas4shots(event_table,qc;
    fig_format="png",save_fig=true)
    # plot the bar chart of the number of shots per basis/state.
    # Author: Taha Selim
    # Date: 2023-12-28
    # the event_table has 4 columns
    # the first column is the basis
    # the second column is the statevector
    # the third column is the probability
    # the fourth column is the number of shots
    # the number of shots is the number of times the state was measured
    # read the dimensions of the Hilbert space
    #n_dim = qc.n_dim
    # add "|" and ">" to the basis of the event_table column 1
    n_dim = qc.n_dim
    # add "|" and ">" to the basis of the event_table column 1
    event_table[:,1] = "|" .* event_table[:,1] .* ">"
    # the number of plots is the number of states divided by 8
    n_states_default = 8 
    n_plots = Int64(ceil(n_dim/8))
    # the number of states in the last plot is the remainder of the division
    n_states_last_plot = mod(n_dim,8)
    
    for iplot in 1:n_plots
    # the number of states in the plot is by default 8
        n_states_plot = 8
        if iplot == n_plots && n_states_last_plot != 0
            n_states_plot = n_states_last_plot
        end
        # the start of the states in the plot is the start of the states in the event_table
        start_state = (iplot-1)*n_states_plot+1
        # the end of the states in the plot is the start of the states in the event_table plus 8
        end_state = start_state + n_states_plot - 1
        # the states in the plot are the states in the event_table from start_state to end_state
        states_plot = event_table[start_state:end_state,1]
        # the shots in the plot are the shots in the event_table from start_state to end_state
        shots_plot = event_table[start_state:end_state,4]
        # plot the bar chart of the number of shots per basis/state
        display(bar(states_plot,shots_plot, label="shots", 
        xlabel="basis", ylabel="shots", 
        title= "shots per basis: " * string(start_state) * " to " * string(end_state)))
         # save the plot as a png file
         #save_fig = true
         #fig_format = "png" # default
         if save_fig
         figname = "shots_per_basis_state_" * string(iplot) * "." * fig_format
          savefig(figname)
         end
    end
end # function plot_bas4shots

# partial measurement function 
function measure_qubits(qc, qubits; big_endian::Bool=true, bas= "Zbas")
    # Inputs:
    # qc: quantum circuit
    # qubits: qubits to measure
    # big_endian: boolean to determine the endianess of the qubits
    # bas: basis to measure the qubits, default is Z basis
    # shots: number of shots to simulate the measurement, it is not defined at the moment.
    # Outputs:
    # qc: quantum circuit with measured qubits
    # get the number of qubits to measure
    # Author: Taha Selim, June 10th, 2025
    n_qs = length(qubits)
    # get the number of qubits in the quantum circuit
    nqubits = qc.n_qubits
    for iq in 1:n_qs
        if bas == "Zbas"
            qc,sv_0,sv_1,p0,p1 = z_measure(qc, qubits[iq],big_endian=big_endian,
            show=false)
        elseif bas == "Xbas"
            qc,sv_0,sv_1,p0,p1 = x_measure(qc, qubits[iq],big_endian=big_endian,
            show=false)
        end
        # update the state vector randomly based on the measurement outcomes
        if rand() < p0
            qc.state_vector = sv_0
        else
            qc.state_vector = sv_1
        end
    end
    return qc
end # end of the function measure_qubits        



end # module