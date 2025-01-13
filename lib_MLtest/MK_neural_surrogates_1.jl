module MK_neural_surrogates
#----------------------------------------------------------------------------------------------------------------------------
# Docs
# © MolKet 2024, MIT License
# www.molket.io
#
# Author: Alain Chancé 
# December 2024
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
#-----------------------------------------------------------------------------------------------------------------------------
export Param, run_surrogate

using Flux, Surrogates, JLD2, Plots, Random, Printf

#--------------------------------
# Define parameter structure
#--------------------------------
Base.@kwdef mutable struct Param
    potential::Function = x -> x
    #
    # f parameters follow
    #
    verbose1::Bool = false
    R::Vector{Float64} = collect(4.0:0.1:6.0)
    Q::Float64 = 0.0
    theta::Vector{Float64}= collect(0.0:10.0:180.0)
    phi::Vector{Float64}= collect(0.0:10.0:180.0)
    n_samples::Int64 = 50
    scale = 2.0E-05
    #
    # Neural surrogate parameters follow
    #
    theta_rad::Vector{Float64} = theta*pi/180
    theta_rad0::Float64 = minimum(theta_rad)
    phi_rad::Vector{Float64} = phi*pi/180
    phi_rad0::Float64 = minimum(phi_rad)
    
    lower_bound::Union{Vector{Float64},Float64} = minimum(R)
    upper_bound::Union{Vector{Float64},Float64} = maximum(R)
    
    maxiters::Int64 = 100
    num_new_samples::Int64 = 1000
    n_echos::Int64 = 20000
    n_iters::Int64 = 5
    # delta = euclidean(model(x), y[1])
    delta_target::Float64 = 1E-5
    grad::Float64 = 0.1
    show::Bool = true
    save_png::Bool = true
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
# Define function train1 adapted from train in Flux.jl
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

#------------------------------------------------------------------------------------------------------------------------
# Define build_RBF_1D() which creates and trains a Radial Basis Function (RBF) neural network surrogate function 
# for the 1D plot of a potential.
#
# The Radial Basis Function is defined as follows:
# $$\operatorname{RBF}(x, c, \sigma)=\exp \left(-\frac{\|x-c\|^2}{2 \sigma^2}\right)$$
# where $c$ is the center and $\sigma$ is the spread.
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
function build_RBF_1D(x, y; param::Param=param)

    # Retrieve parameters from param data structure
    verbose1 = param.verbose1
    R::Vector{Float64} = param.R
    theta::Vector{Float64} = param.theta
    phi::Vector{Float64} = param.phi
    n_echos = param.n_echos
    n_iters = param.n_iters
    lower_bound = param.lower_bound
    upper_bound = param.upper_bound
    grad = param.grad
    delta_target = param.delta_target
    model_file_name = param.model_file_name
    show = param.show
    scale = param.scale
    save_png = param.save_png

    l_R = length(R)
    l_theta = length(theta)
    l_phi = length(phi)

    ok = true

    x_data = reshape(x, 1, :)
    y_data = reshape(y, 1, :)

    if verbose1
        println("\nBuilding a Radial Basis Function (RBF) neural network surrogate function for the 1D plot of a potential") 
        
        println("size(x): ", size(x))
        println("\nbuild_surrogate - x: ")
        println(x)
        
        println("size(y): ", size(y))
        println("\nbuild_surrogate - y: ")
        println(y)
        println(" ")
    end

    data = zip(x_data, y_data)

    # Initialize RBF Layer
    n_centers = length(y)  # Same as the number of data points
    centers = x_data  # Place centers at the data points
    σ = 8E-02           # Initial spread (can be tuned)
    rbf_layer = RBFLayer(centers, σ)

    # Initialize Dense Layer (weights)
    weights = Dense(n_centers, 1)

    # Combine into RBFNet
    model = RBFNet(rbf_layer, weights)

    y_pred::Vector{Float32} = zeros(Float32, length(y))
    delta = 1000.0
    loss = 1E-03

    for i in 1:n_iters
        opt = Descent(grad)  # Gradient descent optimizer
        ps = Flux.trainable(model)
        opt_state = Flux.setup(opt, model)
    
        # Training loop
        for epoch in 1:n_echos
            # Compute gradients
            train1!(loss_fn, model, data, opt_state)

            i = 1
            for xi in x_data
                y_pred[i] = model(xi)[1]
                i += 1
            end
                
            loss_ = Flux.mse(y_pred, y)

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
    
        delta_ = euclidean(y_pred, y)

        if abs(delta_ - delta)/delta < 1E-07
            break
        end

        if verbose1 && i % 5 == 0
            s = @sprintf("%.3e", delta_)
            println("\nIteration: $i, delta = euclidean(model(x), y): $s")
        end

        if delta_ < delta
            delta = delta_
            
            # Update model state
            model_state = Flux.state(model)
            jldsave(model_file_name; model_state)
        end

        if delta_ < delta_target
            if verbose1
                s = @sprintf("%.3e", delta)
                println("\nNeural surrogate function optimized delta: $s is less than target: $delta_target")
            end
            break
        end
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
    i = 1
    for xi in x_data
        y_pred[i] = model(xi)[1]
        i += 1
    end
    
    delta = euclidean(y_pred, y)
    s = @sprintf("%.3e", delta)
    println("\nbuild_RBF_1D - delta = euclidean(y_pred, y): $s")
    
    if verbose1
        println("\nx_data: ")
        println(x_data)

        println("\ny_pred: ")
        println(y_pred)
    end

    #-------------------------------------------------------------------------
    # Plot Potential and RBF surrogate functions
    # https://docs.juliaplots.org/latest/tutorial/#Basic-Plotting:-Line-Plots
    #-------------------------------------------------------------------------
    p = plot(R, [y y_pred], title="Plot Vpot, scale = $scale", label=["Potential" "RBF surrogate"])

    if show
        println("")
        display(p)
    end
    
    if save_png
        s = model_file_name
        plot_png = replace(s, "jld2" => "png")
        savefig(p, plot_png)
    end
    
    return ok, model
end

#------------------------------------------------------------------------------------------------------------------------
# Define build_RBF_2D() which creates and trains a Radial Basis Function (RBF) neural network surrogate function 
# for the 2D plot of a potential.
#
# Building a surrogate, https://docs.sciml.ai/Surrogates/stable/neural/#Building-a-surrogate
# Module SurrogatesFlux, https://github.com/SciML/Surrogates.jl/blob/master/lib/SurrogatesFlux/src/SurrogatesFlux.jl
#
# Flux Saving and Loading Models, https://fluxml.ai/Flux.jl/stable/saving/#Saving-and-Loading-Models
# JLD2, https://github.com/JuliaIO/JLD2.jl
#
# Flux.Losses.mse returns the loss corresponding to mean square error
# https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse
#
# Radial Surrogates
# https://docs.sciml.ai/Surrogates/stable/radials/
# https://github.com/SciML/Surrogates.jl/blob/master/src/Radials.jl
#
# linearRadial() = RadialFunction(0, z -> norm(z))
# cubicRadial() = RadialFunction(1, z -> norm(z)^3)
# multiquadricRadial(c = 1.0) = RadialFunction(1, z -> sqrt((c * norm(z))^2 + 1))
#------------------------------------------------------------------------------------------------------------------------
function build_RBF_2D(x, y; param::Param=param)
    verbose1 = param.verbose1
    potential = param.potential
    R = param.R
    Q = param.Q
    
    theta_rad = param.theta_rad
    theta_rad0 = param.theta_rad0
    
    phi_rad = param.phi_rad
    phi_rad0 = param.phi_rad0
    
    n_samples = param.n_samples
    model_file_name = param.model_file_name
    scale = 2.8E-05
    show = param.show
    save_png = param.save_png

    l_R = length(R)
    l_theta = length(theta_rad)
    l_phi = length(phi_rad)

    xs::Vector{Float64} = zeros(Float64,n_samples) 
    ys::Vector{Float64} = zeros(Float64,n_samples)

    ok = true
    model = nothing
    title_p1 = ""
    title_p2 = ""
    title_p3 = ""

    if l_theta == 1 && l_phi == 1
        println("Either theta, or phi but not both must be of length 1")
        return ok, nothing

    elseif l_phi == 1
        func_Rtheta = R_theta -> potential(R_theta[1], Q, R_theta[2], phi_rad)*scale

        lower_bound = [minimum(R), 0.0]
        upper_bound = [maximum(R), pi/2.0]

        xys = sample(n_samples, lower_bound, upper_bound, SobolSample())

        xs = [xy[1] for xy in xys]
        ys = [xy[2] for xy in xys]
        zs = func_Rtheta.(xys)

        xlims, ylims = minimum(R):maximum(R), minimum(theta_rad):maximum(theta_rad)

        model = RadialBasis(xys, zs, lower_bound, upper_bound)
        zpred = model.(xys)

        if verbose1
            println("\nBuilding a Radial Basis Function (RBF) neural network surrogate function for the 2D plot of a potential")
        end
    
        delta = euclidean(zpred, zs)
        s = @sprintf("%.3e", delta)
        println("\nbuild_RBF_2D - delta = euclidean(zpred, zs): $s")

        title_p1 = string("\nSurface plot Potential(R, theta_rad), scale = ", scale)
        title_p2 = "Contour plot Potential(R, theta_rad)"
        title_p3 = string("Surface plot RBF(R, theta_rad), scale = ", scale)

        if show || save_png
            s1 = surface(xlims, ylims, (x1,x2) -> func_Rtheta((x1,x2)))
            scatter!(xs, ys, zs)
            p1 = plot(s1, title=title_p1)

            c2 = contour(xlims, ylims, (x1,x2) -> func_Rtheta((x1,x2)))
            scatter!(xs, ys)
            p2 = plot(c2, title=title_p2)

            s3 = surface(xlims, ylims, (x1,x2) -> func_Rtheta((x1,x2)))
            scatter!(xs, ys, zs)
            p3 = plot(s3, title=title_p3)
        end
    
    elseif l_theta == 1
        func_Rphi = R_phi -> potential(R_phi[1], Q, theta_rad, R_phi[2])*scale

        lower_bound = [minimum(R), 0.0]
        upper_bound = [maximum(R), pi]

        xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
        
        xs = [xy[1] for xy in xys]
        ys = [xy[2] for xy in xys]
        zs = func_Rphi.(xys)

        xlims, ylims = minimum(R):maximum(R),  minimum(phi_rad):maximum(phi_rad)

        model = RadialBasis(xys, zs, lower_bound, upper_bound)
        zpred = model.(xys)

        if verbose1
            println("\nBuilding a Radial Basis Function (RBF) neural network surrogate function for the 2D plot of a potential")
        end
    
        delta = euclidean(zpred, zs)
        s = @sprintf("%.3e", delta )
        println("\nbuild_RBF_2D - delta = euclidean(zpred, zs): $s")

        title_p1 = string("\nSurface plot Potential(R, phi_rad), scale = ", scale)
        title_p2 = "Contour plot Potential(R, phi_rad)"
        title_p3 = string("Surface plot RBF(R, phi_rad), scale = ", scale)

        if show || save_png
            s1 = surface(xlims, ylims, (x1,x2) -> func_Rphi((x1,x2)))
            scatter!(xs, ys, zs)
            p1 = plot(s1, title=title_p1)

            c2 = contour(xlims, ylims, (x1,x2) -> func_Rphi((x1,x2)))
            scatter!(xs, ys)
            p2 = plot(c2, title=title_p2)

            s3 = surface(xlims, ylims, (x1,x2) -> func_Rphi((x1,x2)))
            scatter!(xs, ys, zs)
            p3 = plot(s3, title=title_p3)
        end

    else
        println("Either theta, or phi or both must be of length 1")
        return false, nothing
    end

    if show
        display(p1)
        display(p2)
        display(p3)
    end
        
    if save_png
        p1_png = replace(model_file_name, ".jld2" => "_p1.png")
        savefig(p1, p1_png)

        p2_png = replace(model_file_name, ".jld2" => "_p2.png")
        savefig(p2, p2_png)

        p3_png = replace(model_file_name, ".jld2" => "_p3.png")
        savefig(p3, p3_png)
    end

    # Update model state
    model_state = Flux.state(model)
    jldsave(model_file_name; model_state)

    return ok, model
end

#-----------------------------------------------------------------------------------------------------------------------------
# Define build_surrogate() which Trains a Radial Basis Function (RBF) neural network surrogate function for function potential
#
# Building a surrogate, https://docs.sciml.ai/Surrogates/stable/neural/#Building-a-surrogate
# Module SurrogatesFlux, https://github.com/SciML/Surrogates.jl/blob/master/lib/SurrogatesFlux/src/SurrogatesFlux.jl
#
# Flux Saving and Loading Models, https://fluxml.ai/Flux.jl/stable/saving/#Saving-and-Loading-Models
# JLD2, https://github.com/JuliaIO/JLD2.jl
#
# Flux.Losses.mse returns the loss corresponding to mean square error
# https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse
#-----------------------------------------------------------------------------------------------------------------------------
function build_surrogate(; param::Param=param)

    # Retrieve potential function from param structure
    potential = param.potential

    # Retrieve parameters from param data structure
    verbose1 = param.verbose1
    
    R::Vector{Float64} = param.R
    Q::Float64 = param.Q

    theta::Vector{Float64} = param.theta
    theta_rad = (pi/180.0)*theta
    param.theta_rad = theta_rad
    param.theta_rad0 = minimum(param.theta_rad)
    theta_rad0 = param.theta_rad0
    
    if theta_rad0 < 0
        theta_rad0 = -theta_rad0
    end
    if theta_rad0 > pi/2.0
        theta_rad0 = theta_rad0 - pi/2
    end

    phi::Vector{Float64} = param.phi
    phi_rad = (pi/180.0)*param.phi
    param.phi_rad = phi_rad
    param.phi_rad0 = minimum(param.phi_rad)
    phi_rad0 = param.phi_rad0 

    l_R = length(R)
    l_theta = length(param.theta)
    l_phi = length(param.phi)

    ok = true
    model = nothing 

    if l_theta == 1 && l_phi == 1
        x = R
        param.lower_bound = minimum(R)
        param.upper_bound = maximum(R)
        
        y_R::Vector{Float64} = zeros(Float64, length(R))
        for i in 1:l_R
            y_R[i] = potential(R[i], Q, theta_rad0, phi_rad0)
        end

        scale = 2.5E-05
        if theta_rad0 < pi/4.0
            scale = 2.5E-05
        elseif abs(theta_rad0 - pi/4.0) < 0.25
            scale = 1.0E-04
        elseif abs(theta_rad0 - pi/3.0) < 0.25
            scale = 2.0E-04
        elseif abs(theta_rad0 - pi/2.0) < 0.25
            scale = 1.0E-03
        else
            scale = 2.5E-05
        end

        param.scale = scale
        y = y_R*param.scale
        
        param.model_file_name = string("model_theta_", theta[1], "_phi_", phi[1], ".jld2")
        
        ok, model = build_RBF_1D(x, y; param=param)

    elseif l_phi == 1
        x = [[r,th] for r in R, th in theta_rad]
        
        func_Rtheta = r_th -> potential(r_th[1], Q, r_th[2], phi_rad0)

        param.scale = 1.0E-05
        y = func_Rtheta.(x)*param.scale

        param.model_file_name = string("model_theta_", Int(theta[1]), "_to_", Int(theta[end]), "_phi_", phi[1], ".jld2")
        
        ok, model = build_RBF_2D(x, y; param=param)

    elseif l_theta == 1
        x = [[r,ph] for r in R, ph in phi_rad]
        
        func_Rphi = r_ph -> potential(r_ph[1], Q, theta_rad0, r_ph[2])

        param.scale = 1.0E-05
        y = func_Rphi.(x)*param.scale

        param.model_file_name = string("model_phi_", Int(phi[1]), "_to_", Int(phi[end]), "_theta_", theta[1], ".jld2")

        ok, model = build_RBF_2D(x, y; param=param)
        
    else
        println("Either theta, or phi or both must be of length 1")
    end
    
    return ok, model
end

#---------------------------------------------------------------------------------
# Define run_surrogate() that trains a model and plots wavefunction and surrogate
#---------------------------------------------------------------------------------
function run_surrogate(; param::Param=param, n_samples::Int64=50, R=collect(4.0:0.1:6.0), Q=0.0, theta=[0.0], phi=[0.0], 
        grad=0.1, show::Bool=true, build::Bool=true, verbose1::Bool=false)
    
    # Set up parameter data structure
    param.verbose1 = verbose1
    param.n_samples = n_samples
    param.R = R
    param.Q = Q
    param.theta = theta
    param.phi = phi
    param.grad = grad
    param.show = show

    if build
        ok, model = build_surrogate(; param=param)
    end
    
    return ok, param.model_file_name
end

end # module MK_neural_surrogates