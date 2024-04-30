module MK_Hcoordinates

# Docs
# © MolKet 2024, MIT License
# www.molket.io
# A module defining the used coordinates in defining the Hamiltonian

# Author: TSelim April 2024

using LinearAlgebra

export Q_canonical, Q_cartesian

# Convert the Q_cartesian to Q_canonical
function Q_canonical(Q_cartesian, omega, mu)
    # Canonical coordinates
    # Inputs
    # Q_cartesian : Cartesian coordinates
    # omega       : frequency
    # mu          : reduced mass
    # Outputs
    # Q_canonical : Canonical coordinates
    # 
    # Author: TSelim April 2024
    Q_canonical = sqrt(mu*omega) * Q_cartesian

    return Q_canonical
end # function Q_canonical


# Convert the Q_canonical to Q_cartesian
function Q_cartesian(Q_canonical, omega, mu)
    # Cartesian coordinates
    # Inputs
    # Q_canonical : Canonical coordinates
    # omega       : frequency
    # mu          : reduced mass
    # Outputs
    # Q_cartesian : Cartesian coordinates
    #

    Q_cartesian = Q_canonical / sqrt(mu*omega)
    return Q_cartesian

end # function Q_cartesian





end # module