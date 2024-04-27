module constants
# Docs 
# Constants and units used in the code and the whole package are defined here.

export au2cm1, NA, h_const, cm1

cm1::Float64 = 1/219474.63137098 # atomic units to cm^-1 conversion factor
                            # calcualted as 1/cm1 gives the value as 1/cm1 
au2cm1::Float64 = 1/cm1 # atomic units to cm^-1 conversion factor
NA::Float64 = 6.02214199e23 # Avogadro's number, copied from NIST
h_const::Float64 = 6.6260693e-34 # Planck's constant, J.s
end # module