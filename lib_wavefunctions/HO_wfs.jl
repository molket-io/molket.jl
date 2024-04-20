module HO_wfs
# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the harmonic oscillator wave functions
# used in the quantum molecular dynamics simulations
# and the quantum algorithms

# Author: TSelim April 2024

# Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)

# call the lib_SpecialPolynomials.jl
include("../lib_SpecialPolynomials/MK_SpecialPolynomials.jl")
using ..MK_SpecialPolynomials: laguerre, glaguerre, ghermite, gchebhermite


using LinearAlgebra

end # module