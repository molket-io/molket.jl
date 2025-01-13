module MK_neural_surrogates
#------------------------------------------------------------------------------------------------------------------------------
# Docs
# © MolKet 2024, MIT License
# www.molket.io
#
# Author: Alain Chancé June 2024
#
# Build neural surrogate function that predicts the wavefunction for a given omega and save neural model state
#
# Flux: The Julia Machine Learning Library, https://fluxml.ai/Flux.jl/stable/
# Flux, GPU Support: https://fluxml.ai/Flux.jl/stable/gpu/
#
# Surrogates.jl: Surrogate models and optimization for scientific machine learning, https://docs.sciml.ai/Surrogates/stable/
# Neural network tutorial, https://docs.sciml.ai/Surrogates/stable/neural/
#------------------------------------------------------------------------------------------------------------------------------
export Param, run_surrogate

using Flux, Surrogates, JLD2, Plots, Printf

#--------------------------------
# Define parameter structure
#--------------------------------
Base.@kwdef mutable struct Param
    f::Function = x -> x
    #
    # f parameters follow
    #
    verbose1::Bool = false
    qnum = [0, 1, 2, 3]
    npoints::Int64 = 175
    Qgrid = range(-5,stop=5,length=npoints)
    n::Int64 = 0
    omega::Union{Vector{Float64},Int64,Float64} = 1200.0
    g1_constant::Float64 = -0.0030609
    t111_constant::Float64 = 0.30667
    u1111_constant::Float64 = 0.31927
    shift::Float64 = 0.7
    zerofy::Bool = false
    vmin::Float64 = 1e-3
    #
    # Neural surrogate parameters follow
    #
    maxiters::Int64 = 10
    num_new_samples::Int64 = 40000
    n_echos::Int64 = 50000
    n_iters::Int64 = 10
    # delta = euclidean(model(x), y[1])
    #delta_target::Float64 = 0.01
    delta_target::Float64 = 1e-5
    grad::Float64 = 0.1
    show::Bool = true
    save_png::Bool = true
    model_file_name::String = "model.jld2"
    model = Chain(
      Dense(1, 20, σ),
      Dense(20, 80, σ),
      Dense(80, npoints, σ)
    )
end

#----------------------------------------------------------------------------------------
# Ref. Distances.jl, A Julia package for evaluating distances (metrics) between vectors.
# https://github.com/JuliaStats/Distances.jl
#----------------------------------------------------------------------------------------
using Distances: euclidean

import Zygote
#----------------------------------------------------------------
# Define function train1
# https://github.com/FluxML/Flux.jl/blob/master/src/train.jl
#----------------------------------------------------------------
function train1!(loss, model, data, opt; cb = nothing)
  isnothing(cb) || error("""train! does not support callback functions.
                            For more control use a loop with `gradient` and `update!`.""")
  for (i,d) in enumerate(data)
    d_splat = d isa Tuple ? d : (d,)

    l, gs = Zygote.withgradient(m -> loss(m, d_splat...), model)

    if !isfinite(l)
      throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end

    opt, model = Optimisers.update!(opt, model, gs[1])

    Base.haslength(data) ? i/length(data) : nothing
  end
end

#----------------------------------------------------------------
# Define mutable structure NeuralSurrogate
# Source: Surrogates.jl/lib/SurrogatesFlux/src/SurrogatesFlux.jl 
# abstract type AbstractSurrogate <: Function end
#----------------------------------------------------------------
mutable struct NeuralSurrogate{X, Y, M, L, O, P, N, A, U} <: AbstractSurrogate
    x::X
    y::Y
    model::M
    loss::L
    opt::O
    ps::P
    n_echos::N
    lb::A
    ub::U
end

#--------------------------------------------------------------------------------------------------------------------
# Define build_surrogate() which builds a neural model and a neural surrogate function for function f()
#
# Building a surrogate, https://docs.sciml.ai/Surrogates/stable/neural/#Building-a-surrogate
# Module SurrogatesFlux, https://github.com/SciML/Surrogates.jl/blob/master/lib/SurrogatesFlux/src/SurrogatesFlux.jl
#
# Flux Saving and Loading Models, https://fluxml.ai/Flux.jl/stable/saving/#Saving-and-Loading-Models
# JLD2, https://github.com/JuliaIO/JLD2.jl
#
# Flux.Losses.mse returns the loss corresponding to mean square error
# https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse
#---------------------------------------------------------------------------------------------------------------------
function build_surrogate(x, y; param::Param=param)

    # Retrieve parameters from param data structure
    verbose1 = param.verbose1
    omega = param.omega
    npoints = param.npoints
    n_echos = param.n_echos
    n_iters = param.n_iters
    grad = param.grad
    delta_target = param.delta_target
    model_file_name = param.model_file_name
    model = param.model

    x_data = vec.(collect.(x))
    y_data = vec.(collect.(y))
    
    data = zip(x_data, y_data)

    if verbose1
        println("\nBuilding a neural surrogate function that predicts the function f()") 
        
        println("\nbuild_surrogate - x: ")
        println(x)
        
        println("\nbuild_surrogate - y: ")
        println(y)
    end
    
    delta = 1E+03
    loss = 1E-03
    ok = true
    
    for i in 1:n_iters
        try
            opt = Descent(grad)
            ps = Flux.trainable(model)
            opt_state = Flux.setup(opt, model)
            loss_fn = (model, x, y) -> Flux.mse(model(x), y)

            # Training loop
            for epoch in 1:n_echos
                # Using explicit-style `train!(loss, model, data, opt_state)
                #Flux.train!(loss, model, data, opt_state)
                train1!(loss_fn, model, data, opt_state)

                loss_ = Flux.mse(model(x), y[1])

                if abs(loss_ - loss)/loss < 1E-07
                    break
                else
                    loss = loss_
                end

                # Print loss occasionally
                if verbose1 && epoch % 1000 == 0
                    s = @sprintf("%.3e", loss)
                    println("Epoch $epoch: Loss = $s")
                end
            end
            
            delta_ = euclidean(model(x), y[1])

            if abs(delta_ - delta)/delta < 1E-07
                break
            end

            if verbose1 && i % 10 == 0
                s = @sprintf("%.3e", delta_)
                println("Iteration: $i, delta = euclidean(model(x), y): $s")
            end

           if delta_ < delta
                delta = delta_
            
                # Update model state
                model_state = Flux.state(model)
                jldsave(model_file_name; model_state)
            end

            if delta_ < delta_target
                if verbose1
                    s = @sprintf("%.3e", delta_)
                    println("\nbuild surrogate function optimized delta: $s is less than target: $delta_target")
                end
                break
            end
        
        catch e
            n_echos -= 1
            if n_echos == 0
                ok = false
                break
            end
        end
    end

    if !ok
        return ok
    end
    
    #-----------------------
    # Load best model state 
    #-----------------------
    if isfile(model_file_name)
        model_state = JLD2.load(model_file_name, "model_state")
        Flux.loadmodel!(model, model_state)
    else
        println("\nFile not found: $model_file_name")
    end

    #------------------------------------
    # Compute delta for best model state
    #------------------------------------
    y_pred = model(x)
    delta = euclidean(y_pred, y[1])
    
    if verbose1
        s = @sprintf("%.3e", delta)
        println("\ndelta = euclidean(model(x), y): $s")
        println("\ndelta target: $delta_target")
        println("\ny_pred: $y_pred")
    end
    
    return ok
end

#--------------------------------------------------------------------------------------------------------------
# Define print_surrogate() that prints the wavefunctions computed by the neural surrogate and for the training 
#--------------------------------------------------------------------------------------------------------------
function print_surrogate(model, x, y; param::Param=param)

    # Retrieve parameters from param data structure
    verbose1 = param.verbose1
    Qgrid = param.Qgrid
    omega = param.omega
    npoints = param.npoints
    qnum = param.qnum
    n = param.n
    model_file_name = param.model_file_name
    show = param.show
    save_png = param.save_png

    neural_x = model(x)
    delta = euclidean(neural_x, y[1])

    s = @sprintf("%.3e", delta)
    println("\ndelta = euclidean(model(x), y): $s")

# Plot wavefunction and surrogate
# https://docs.juliaplots.org/latest/tutorial/#Basic-Plotting:-Line-Plots

    p = plot(Qgrid, [y[1][:,1] neural_x[:,1]], title="omega: $omega, n: $n", label=["Wavefunction" "Neural surrogate"])
    
    if show
        println("")
        display(p)
    end

    if save_png
        plot_png = replace(model_file_name, "jld2" => "png")
        savefig(p, plot_png)
    end
        
    return neural_x, y[1]
end

#---------------------------------------------------------------------------------
# Define run_surrogate() that trains a model and plots wavefunction and surrogate
#---------------------------------------------------------------------------------
function run_surrogate(; param::Param=param, npoints=175, qnum=[0, 1, 2, 3], n=0, omega=1200.0, shift=0.7, grad=0.1, show::Bool=true, save_png::Bool=true, build::Bool=true, verbose1::Bool=false)

    f = param.f
    
    # Set up parameter data structure
    param.verbose1 = verbose1
    param.npoints = npoints
    param.Qgrid = range(-5,stop=5,length=npoints)
    param.qnum = qnum
    param.n = n
    param.omega = omega
    param.shift = shift
    param.grad = grad
    param.show = show
    param.save_png = save_png

    model = param.model

    ok = true

    x = [n]
    y = []

    for omega1 in omega
        push!(y, f(omega1))
    end

    if build
        #-------------------------------------------------
        # Train model and save best model state in a file
        #-------------------------------------------------
        if length(omega) == 1
            param.model_file_name = string("model_", omega, "_", param.n, ".jld2")
        else
            param.model_file_name = string("model_", Int(omega[1]), "_to_", Int(omega[end]), "_", param.n, ".jld2")
        end
        
        ok = build_surrogate(x, y; param=param)
    else
        #------------------------------------------------------
        # Load trained model state model_400_to_1800 from file
        #------------------------------------------------------
        model_file_name = string("model_400_to_1800_", param.n, ".jld2")
        
        if isfile(model_file_name)
            model_state = JLD2.load(model_file_name, "model_state")
            Flux.loadmodel!(model, model_state)

            # Set-up model file name for test run
            if length(omega) == 1
                param.model_file_name = string("model_", omega, "_", param.n, ".jld2")
            else
                param.model_file_name = string("model_", Int(omega[1]), "_to_", Int(omega[end]), "_", param.n, ".jld2")
            end
        else
            ok = false
            println("\nFile not found: $model_file_name")
            return ok, ""
        end
    end

    if ok
        A, Y = print_surrogate(model, x, y; param=param)
    end
    
    return ok, param.model_file_name
end

end # module MK_neural_surrogates