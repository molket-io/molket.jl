module MK_neural_surrogates
#------------------------------------------------------------------------------------------------------------------------------
# Docs
# © MolKet 2024, MIT License
# www.molket.io
#
# Author: Alain Chancé December 2024
#
# Create and train a Radial Basis Function (RBF) neural network surrogate function for the 1D plot of a potential.
#
# Flux: The Julia Machine Learning Library, https://fluxml.ai/Flux.jl/stable/
# Flux, GPU Support: https://fluxml.ai/Flux.jl/stable/gpu/
#
# Surrogates.jl: Surrogate models and optimization for scientific machine learning, https://docs.sciml.ai/Surrogates/stable/
# Neural network tutorial, https://docs.sciml.ai/Surrogates/stable/neural/
#
# Radial Surrogates
# https://docs.sciml.ai/Surrogates/stable/radials/
#
#------------------------------------------------------------------------------------------------------------------------------
export Param, run_surrogate

using Flux, Surrogates, JLD2, Plots, Random

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
    Qgrid = R
    Q::Float64 = 0.0
    theta::Vector{Float64}= collect(0.0:10.0:180.0)
    phi::Float64 = 0.0
    n_samples::Int64 = 50
    npoints::Int64 = length(R)
    #
    # Neural surrogate parameters follow
    #
    theta_rad::Vector{Float64} = theta*pi/180
    phi_rad::Float64 = phi*pi/180
    lower_bound::Float64 = minimum(R)
    upper_bound::Float64 = maximum(R)
    maxiters::Int64 = 100
    num_new_samples::Int64 = 1000
    n_echos::Int64 = 50
    n_iters::Int64 = 30
    # delta = euclidean(model(x), y[1])
    delta_target::Float64 = 0.01
    grad::Float64 = 0.1
    model_file_name::String = "model1.jld2"
end

# Define the RBF Layer
struct RBFLayer
    centers::Array{Float32, 2}  # Centers of RBF units
    σ::Float32                  # Spread (standard deviation)
end

# Define the complete RBF network
struct RBFNet
    rbf_layer::RBFLayer
    weights::Dense
end

# Define Radial Basis Function (RBF)
function rbf(x, c, σ)
    return exp.(-sum((x .- c).^2) / (2σ^2))
end

function (l::RBFLayer)(x)
    h = [rbf(x, l.centers[:, i], l.σ) for i in 1:size(l.centers, 2)]
    return h
end

function (net::RBFNet)(x)
    # Apply the RBF layer and then the Dense layer
    rbf_output = net.rbf_layer(x)
    return net.weights(rbf_output)
end

# Loss function
function loss_fn(model, x, y)
    y_pred = model(x)
    return Flux.mse(y_pred, y)
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

#------------------------------------------------------------------------------------------------------------------------
# Define build_surrogate() which Trains a Radial Basis Function (RBF) neural network surrogate function for function f()
#
# Building a surrogate, https://docs.sciml.ai/Surrogates/stable/neural/#Building-a-surrogate
# Module SurrogatesFlux, https://github.com/SciML/Surrogates.jl/blob/master/lib/SurrogatesFlux/src/SurrogatesFlux.jl
#
# Flux Saving and Loading Models, https://fluxml.ai/Flux.jl/stable/saving/#Saving-and-Loading-Models
# JLD2, https://github.com/JuliaIO/JLD2.jl
#
# Flux.Losses.mse returns the loss corresponding to mean square error
# https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse
#------------------------------------------------------------------------------------------------------------------------
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

    ok = true

    if verbose1
        println("\nBuilding a neural surrogate function that predicts the function f()") 
        
        println("\nbuild_surrogate - x: ")
        println(x)
        
        print("\nbuild_surrogate - length(y) :", length(y))
        println("\nbuild_surrogate - y: ")
        println(y)
    end

    # Reshape x for compatibility with struct RBFLayer
    x_data::Matrix{Float32} = reshape(x, 1, :)
    y_data::Matrix{Float32} = reshape(y, 1, :)

    data = zip(x_data, y_data)

    # Initialize RBF Layer
    n_centers = length(y)  # Same as the number of data points
    centers = x_data  # Place centers at the data points
    σ = 0.5           # Initial spread (can be tuned)
    rbf_layer = RBFLayer(centers, σ)

    # Initialize Dense Layer (weights)
    weights = Dense(n_centers, 1)

    # Combine into RBFNet
    model = RBFNet(rbf_layer, weights)

    y_pred::Vector{Float32} = zeros(Float32, length(y))
    delta = 1000.0

    for i in 1:n_iters
        opt = Descent(grad)  # Gradient descent optimizer
        ps = Flux.trainable(model)
        opt_state = Flux.setup(opt, model)

        # Training loop
        for epoch in 1:n_echos
            # Compute gradients
            train1!(loss_fn, model, data, opt_state)
    
            # Print loss occasionally
            if epoch % 100 == 0
                println("Epoch $epoch: Loss = $(loss_fn(model, x_data, y_data))")
            end
        end
        
        i = 1
        for xi in x_data
            #println("xi: $xi, model(xi)[1]): ", model(xi)[1])
            y_pred[i] = model(xi)[1]
            i += 1
        end
    
        delta_ = euclidean(y_pred, y)

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
    end

    #-----------------------
    # Load best model state 
    #-----------------------
    model_state = JLD2.load(model_file_name, "model_state")
    Flux.loadmodel!(model, model_state)

    #------------------------------------
    # Compute delta for best model state
    #------------------------------------
    i = 1
    for xi in x_data
        y_pred[i] = model(xi)[1]
        i += 1
    end
    
    delta = euclidean(y_pred, y)
    
    if verbose1
        println("\nx_data: ")
        println(x_data)

        println("\ny_pred: ")
        println(y_pred)
        
        println("build_surrogate Radial Basis Function (RBF) neural network - delta = euclidean(y_pred, y): $delta")
    end
    
    return ok, model
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
    Qgrid = param.Qgrid
    Q = param.Q
    theta = param.theta
    phi = param.phi
    n_samples = param.n_samples
    npoints = param.npoints

    y_pred::Vector{Float32} = zeros(Float32, length(y))
    i = 1
    for xi in x
        y_pred[i] = model(xi)[1]
        i += 1
    end
        
    delta = euclidean(y_pred, y)
    
    println("build_surrogate - delta = euclidean(y_pred, y): $delta")
    
    println("")
    println("\nPlot Vpot*2.0E-05")
    display(plot(Qgrid, y))
    
    println("\nPlot RBF model(R)*2.0E-05")
    display(plot(Qgrid, y_pred))
    
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
    
    x = R
    y = f(theta0; param=param)
        
    param.model_file_name = string("model1.jld2")
    
    ok, model = build_surrogate(x, y; param=param)
    if ok
        print_surrogate(model, x, y; param=param)
    end
    
    return model, x, y
end

end # module MK_neural_surrogates