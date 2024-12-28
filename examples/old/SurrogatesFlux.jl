#-------------------------------------------------------------------------------------
# This file is downloaded from Surrogates.jl/lib/SurrogatesFlux/src/SurrogatesFlux.jl 
# Neural network tutorial, https://docs.sciml.ai/Surrogates/dev/neural/
#-------------------------------------------------------------------------------------

module SurrogatesFlux

import Surrogates: add_point!, AbstractSurrogate, _check_dimension
export NeuralSurrogate

using Flux

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

"""
NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)

  - model: Flux layers
  - loss: loss function
  - opt: optimization function
"""
function NeuralSurrogate(x, y, lb, ub; model = Chain(Dense(length(x[1]), 1), first),
        loss = (x, y) -> Flux.mse(model(x), y), opt = Descent(0.01),
        n_echos::Int = 1)
    X = vec.(collect.(x))
    data = zip(X, y)
    ps = Flux.params(model)
    for epoch in 1:n_echos
        Flux.train!(loss, ps, data, opt)
    end
    return NeuralSurrogate(x, y, model, loss, opt, ps, n_echos, lb, ub)
end

function (my_neural::NeuralSurrogate)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(my_neural, val)
    v = [val...]
    out = my_neural.model(v)
    if length(out) == 1
        return out[1]
    else
        return out
    end
end

function add_point!(my_n::NeuralSurrogate, x_new, y_new)
    if eltype(x_new) == eltype(my_n.x)
        append!(my_n.x, x_new)
        append!(my_n.y, y_new)
    else
        push!(my_n.x, x_new)
        push!(my_n.y, y_new)
    end
    X = vec.(collect.(my_n.x))
    data = zip(X, my_n.y)
    for epoch in 1:(my_n.n_echos)
        Flux.train!(my_n.loss, my_n.ps, data, my_n.opt)
    end
    nothing
end

end # module
