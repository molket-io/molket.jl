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

using Flux, Surrogates, JLD2, Plots

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
    omega::Float64 = 1200
    g1_constant::Float64 = -0.0030609
    t111_constant::Float64 = 0.30667
    u1111_constant::Float64 = 0.31927
    shift::Float64 = 0.7
    zerofy::Bool = false
    vmin::Float64 = 1e-5
    #
    # Neural surrogate parameters follow
    #
    lower_bound::Union{Int64,Float64} = 800.0
    upper_bound::Union{Int64,Float64} = 1600.0
    maxiters::Int64 = 10
    num_new_samples::Int64 = 40000
    n_echos::Int64 = 40000
    n_iters::Int64 = 1
    # delta = euclidean(model(x), y[1])
    delta_target::Float64 = 0.01
    grad::Float64 = 0.1
    model_file_name::String = "model.jld2"
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
    npoints = param.npoints
    n_echos = param.n_echos
    n_iters = param.n_iters
    lower_bound = param.lower_bound
    upper_bound = param.upper_bound
    grad = param.grad
    
    if verbose1
        println("\nBuilding a neural surrogate function that predicts the wavefunctions ") 
    end

    X = vec.(collect.(x))
    data = zip(X, y)

    MyModel() = Chain(
      Dense(1, 20, σ),
      Dense(20, 80, σ),
      Dense(80, npoints, σ)
    )
    
    model = MyModel()
    
    for i in 1:n_iters
        try
            opt = Descent(grad)
            ps = Flux.trainable(model)
            opt_state = Flux.setup(opt, model)
            loss = (model, x, y) -> Flux.mse(model(x), y)

            for epoch in 1:n_echos
                # Using explicit-style `train!(loss, model, data, opt_state)
                #Flux.train!(loss, model, data, opt_state)
                train1!(loss, model, data, opt_state)
            end
            
            neural = NeuralSurrogate(x, y, model, loss, opt, ps, n_echos, lower_bound, upper_bound)

            if size(model(x))[1] != size(y[1])[1]
                println("build_surrogate - Inconsistent dimensions")
                println("\nsize(model(x)): $size(model(x))") 
                println("\nsize(y[1]) $size(y[1])")
            else
                delta = euclidean(model(x), y[1])
                println("build_surrogate - delta = euclidean(model(x), y[1]): $delta")
            end
            
            return true, MyModel(), model, neural
        
        catch e
            n_echos -= 1
            if n_echos == 0
                break
            end
        end
    end
    
    return false, MyModel(), model, nothing
end

#---------------------------------------------------------------------------------------------------------
# Define function neural_optimize() which optimizes the neural surrogate function 
# Optimization techniques, https://docs.sciml.ai/Surrogates/stable/optimizations/#Optimization-techniques
#---------------------------------------------------------------------------------------------------------
function neural_optimize(x, y, MyModel, model, model_file_name, neural; param::Param=param)
        
    # Retrieve parameters from param data structure
    f = param.f
    verbose1 = param.verbose1
    n_echos = param.n_echos
    n_iters = param.n_iters
    lower_bound = param.lower_bound
    upper_bound = param.upper_bound
    maxiters = param.maxiters
    grad = param.grad
    num_new_samples = param.num_new_samples
    delta_target = param.delta_target
    
    #----------------------------------------------------------------------------------------------------
    # Save current model state into file model_file_name.jld2
    # Flux Saving and Loading Models, https://fluxml.ai/Flux.jl/stable/saving/#Saving-and-Loading-Models
    #-----------------------------------------------------------------------------------------------------
    if verbose1
        println("\nSaving neural model state in file: $model_file_name")
        println(" ")
    end
    
    model_state = Flux.state(model)
    jldsave(model_file_name; model_state)
    
    x_a = x
    y_a = y

    neural_x_a = model(x_a)
    
    delta = euclidean(neural_x_a, y_a[1])

    if delta < delta_target
        if verbose1
            println("No optimization needed since delta: $delta is less than target: $delta_target")
        end
        return neural, model
    end
    
    delta_ = delta

    for i in 1:n_iters
        if verbose1
            println("\niteration: $i  delta: $delta_")
        end
    
        try
            surrogate_optimize(f, SRBF(), lower_bound, upper_bound, neural, RandomSample(), maxiters=maxiters, num_new_samples=num_new_samples)
        catch e
            maxiters -= 1
            if maxiters == 0
                break
            end
        end
        
        neural_x_a = model(x_a)
        delta_ = euclidean(neural_x_a, y_a[1])
        
        if delta_ < delta
            delta = delta_
            
            # Update model state
            model_state = Flux.state(model)
            jldsave(model_file_name; model_state)
            
            if delta < delta_target
                if verbose1
                    println("Neural surrogate function optimized delta: $delta_ is less than target: $delta_target")
                end
                break
            else
                if verbose1
                    println("Neural surrogate function optimized delta: $delta_,  delta target: $delta_target")
                end
            end
            
        else
            if verbose1
                println("Exiting optimization delta: $delta_, delta target: $delta_target")
            end
            break
        end
    end

    #-----------------------
    # Load best model state 
    #-----------------------
    model_state = JLD2.load(model_file_name, "model_state")
    Flux.loadmodel!(model, model_state)

    #-----------------------------------------------
    # Update neural surrogate with best model state
    #-----------------------------------------------
    neural = NeuralSurrogate(x, y, model, (model, x, y) -> Flux.mse(model(x), y), Descent(grad), Flux.trainable(model), n_echos, lower_bound, upper_bound)

    if verbose1
        println("\nmodel(x_a): ", model(x_a), " f(x_a): ", y_a, " delta: ", delta, " delta target: ", delta_target)
        println(trace_f, "\nmodel(x_a): ", model(x_a), " f(x_a): ", y_a, " delta: ", delta, " delta target: ", delta_target)
    end

    return neural, model
end

#------------------------------------------------------------------------------------------------------------------------
# Define neural_surrogate() that builds a neural surrogate function, optimizes it and saves the neural model into a file
#------------------------------------------------------------------------------------------------------------------------
function neural_surrogate(x, y; param::Param=param)

    # Retrieve parameters from param data structure
    f = param.f
    model_file_name = param.model_file_name
        
    ok, MyModel(), model, neural = build_surrogate(x, y; param=param)

    if ok
        neural, model = neural_optimize(x, y, MyModel, model, model_file_name, neural; param=param)
        return ok, model
    else
        return ok, nothing
    end
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

    neural_x = model(x)
    delta = euclidean(neural_x, y[1])

    println("\nneural_surrogate - delta = euclidean(model(x), y[1]): ", delta)

    if verbose1
        println("\nneural(x): ")
        show(stdout, "text/plain", neural_x)
        
        println("")
        println("\ny: ")
        show(stdout, "text/plain", y[1])
    end

    println("")
    println("\nPlot with omega = ", omega, ", n = ", n, ", y[1]")
    display(plot(Qgrid,y[1][:,1]))
    
    println("\nPlot with omega = ", omega, ", n = ", n, ", neural surrogate")
    display(plot(Qgrid,neural_x[:,1]))
        
    return neural_x, y[1]
end

#---------------------------------------------------------------------------------
# Define run_surrogate() that trains a model and plots wavefunction and surrogate
#---------------------------------------------------------------------------------
function run_surrogate(; param::Param=param, npoints=175, qnum=[0, 1, 2, 3], n=0, omega=1200, shift=0.7, grad=0.1, verbose1=false)

    f = param.f
    
    # Set up parameter data structure
    param.verbose1 = verbose1
    param.npoints = npoints
    param.qnum = qnum
    param.n = n
    param.omega = omega
    param.shift = shift
    param.lower_bound = min(param.lower_bound, omega)
    param.upper_bound = max(param.upper_bound, omega)
    param.grad = grad

    x = [omega]
    y = []
    for omega1 in x
        push!(y, f(omega1))
    end

    if n == 0
        shift = sum(y[1])/sizeof(y[1])[1]
    end
    
    param.model_file_name = string("model_", omega, "_", param.n, ".jld2")
    ok, model = neural_surrogate(x, y; param=param)
    if ok
        A, Y = print_surrogate(model, x, y; param=param)
    end
    
    return model, x, y
end

end # module MK_neural_surrogates