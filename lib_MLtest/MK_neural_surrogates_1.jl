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
    theta0::Float64 = 90.0
    R::Vector{Float64} = collect(4.0:0.1:6.0)
    Q::Float64 = 0.0
    theta::Vector{Float64}= collect(0.0:10.0:180.0)
    phi::Float64 = 0.0
    n_samples::Int64 = 50
    npoints::Int64 = n_samples
    #
    # Neural surrogate parameters follow
    #
    theta_rad::Vector{Float64} = theta*pi/180
    phi_rad::Float64 = phi*pi/180
    lower_bound::Float64 = min(-pi, minimum(theta_rad))
    upper_bound::Float64 = max(pi, maximum(theta_rad))
    maxiters::Int64 = 100
    num_new_samples::Int64 = length(R)
    n_echos::Int64 = length(R)
    n_iters::Int64 = 400
    # delta = euclidean(model(x), y[1])
    delta_target::Float64 = 0.01
    grad::Float64 = 0.1
    model_file_name::String = "model1.jld2"
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
    R::Vector{Float64} = param.R
    theta::Vector{Float64} = param.theta
    npoints = param.npoints
    n_echos = param.n_echos
    n_iters = param.n_iters
    lower_bound = param.lower_bound
    upper_bound = param.upper_bound
    grad = param.grad
    delta_target = param.delta_target
    model_file_name = param.model_file_name
    
    X = vec.(collect.(x))
    data = zip(X, y)

    if verbose1
        println("\nBuilding a neural surrogate function that predicts the function f()") 
        
        println("\nbuild_surrogate - X: ")
        println(X)
        
        print("\nbuild_surrogate - length(y) :", length(y))
        println("\nbuild_surrogate - y: ")
        println(y)
    end

    k = length(y)

    MyModel() = Chain(
      Dense(1, 10, σ),
      Dense(10, 5, σ),
      Dense(5, k, σ)
    )
    
    model = MyModel()
    delta = 1E+03
    ok = true
    
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
            delta_ = euclidean(model(x), y)

            if verbose1
                println("build_surrogate - delta = euclidean(model(x), y): $delta_")
            end

            if delta_ < delta
                delta = delta_
            
                # Update model state
                model_state = Flux.state(model)
                jldsave(model_file_name; model_state)
            end

            if delta_ < delta_target
                if verbose1
                    println("Neural surrogate function optimized delta: $delta_ is less than target: $delta_target")
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
        return ok, MyModel(), model, nothing
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
        neural_x = model(x)
        println("\nmodel(x): $neural_x")
        println("\nf(x): $y")
        println("\ndelta: $delta")
        println("\ndelta target: $delta_target")
    end
    
    return ok, MyModel(), model, nothing
end

#--------------------------------------------------------------------------------------------------------------
# Define print_surrogate() that prints the wavefunctions computed by the neural surrogate and for the training 
#--------------------------------------------------------------------------------------------------------------
function print_surrogate(model, x, y; param::Param=param)

    # Retrieve parameters from param data structure
    f = param.f
    verbose1 = param.verbose1
    theta0 = param.theta0
    R = param.R
    Q = param.Q
    theta = param.theta
    phi = param.phi
    n_samples = param.n_samples
    npoints = param.npoints
    
    delta = euclidean(model(x), y)

    println("\nneural_surrogate - delta = euclidean(model(x), y[1]): ", delta)

    xlims = (4.0, 6.0)
    
    Qgrid = range(4,stop=6,length=length(R))
    
    println("")
    println("\nPlot Vpot*2.0E-05")
    display(plot(Qgrid,y))
    
    println("\nPlot model(R)*2.0E-05")
    display(plot(Qgrid,model(x)[:]))
    
end

#---------------------------------------------------------------------------------
# Define run_surrogate() that trains a model and plots wavefunction and surrogate
#---------------------------------------------------------------------------------
function run_surrogate(; param::Param=param, n_samples=50, theta0=90.0, R=collect(4.0:0.1:6.0), Q=0.0, theta=collect(0.0:10.0:180.0), phi=0.0, grad=0.1, verbose1=false)

    f = param.f
    
    # Set up parameter data structure
    param.verbose1 = verbose1
    param.theta0 = theta0
    param.R = R
    param.Q = Q
    param.theta = theta
    param.phi = phi
    param.n_samples = n_samples
    param.grad = grad
    
    x = [theta0]
    y = f(theta0; param=param)
        
    param.model_file_name = string("model1.jld2")
    
    ok, MyModel(), model, neural = build_surrogate(x, y; param=param)
    if ok
        print_surrogate(model, x, y; param=param)
    end
    
    return model, x, y
end

end # module MK_neural_surrogates