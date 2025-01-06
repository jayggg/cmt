"""
Solve dY/dt = F(t, Y) using the RKC class.
"""

from cmt import RKC
import ngsolve as ng


# Define a function F(t, Y) = t^3 in python:
def cubed(t, Y):
    for i in range(len(Y)):
        Y[i] = t**3


# Set initial vector Y = Y0
Y = ng.Vector([1, 2 + 1j, 1, 1j, 1] * 2)
Y0 = ng.Vector(Y)

# Set up an RKC object using the python function
nstage = 4
rk = RKC(cubed, nstage, len(Y))

# Solve
t0 = 0
dt = 0.01
nsteps = 100
rk.SolveIVP(Y, t0, dt, nsteps)

# Print final solution
print('Computed RK solution at end:\n', Y)

# Error: difference between computed & exact solution t^4/4:
t = nsteps * dt
err = ng.Vector(Y0.NumPy() + t**4 / 4 - Y.NumPy())
print('Error wrt exact solution:\n', err.Norm())
