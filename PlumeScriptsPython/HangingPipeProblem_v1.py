import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

"""
Beni Bienz 9/2018

Original notes (Mike Wolovick 7/18/2018):

This script solves the problem of the steady-state configuration of a
flexible pipe being hung in a horizontal current with a weight at the
end.
"""

# Parameters:
# Physical Parameters:
rho_w = 1030
dragcoeff = 2
pipediameter = 0.2
uppervelocity = 0.45
lowervelocity = 0.1
endmass = 200
stratdepth = np.arange(25, 175, 25)
stratdepth = np.array([150])
pipelength = 200
g = 9.8
centralvalue = 100
xsize = 1000
maxiterations = 500
tolerance = 1e-06
damping = 0.85

# Aesthetic Settings:
figname = 'HangingPipeProblem_stratdepth_v1.png'

# Results
results = []
# ------------------------------------------------------

# Work:
# Loop through test variable:
numsamples = len(stratdepth)
centralind = np.abs(stratdepth - centralvalue).argmin()

for d1 in range(numsamples):

    # Make along-pipe grid:
    X_lr = np.linspace(0, pipelength, xsize + 1)
    X_c = 0.5 * (X_lr[:-1] + X_lr[1:])
    dx = X_lr[1] - X_lr[0]

    # Pre-allocate variables:
    Drag_c = np.zeros(xsize)
    Tension_lr = np.zeros(xsize + 1)

    # Make initial guess (semicricle):
    Theta_c = (np.pi / 2) * np.linspace(0.5 / xsize, 1 - 0.5 / xsize, xsize)
    SinTheta_c = np.sin(Theta_c)
    CosTheta_c = np.cos(Theta_c)
    Distance_lr = np.cumsum(np.hstack(([0], dx * CosTheta_c)))
    Depth_lr = np.cumsum(np.hstack(([0], dx * SinTheta_c)))


    # Optional figure
    # plt.plot(Distance_lr,-Depth_lr,'k')
    # plt.xlim([0,pipelength])
    # plt.ylim([-pipelength,0])
    # plt.xlabel('Distance (m)')
    # plt.ylabel('Depth (m)')
    # plt.title('Iteration=0')
    # plt.show()

    # Iterate for convergence:
    done = 0
    iteration = 1
    Theta_c_last = np.copy(Theta_c)

    while done == 0:

        # Identify grid cells fully above the stratification:
        AboveStrat_lr = Depth_lr < stratdepth[d1]
        AboveStrat_c = AboveStrat_lr[:-1] & AboveStrat_lr[1:]

        # Identify the grid cell that straddles the stratification:
        straddleind = np.argmax(AboveStrat_c == 0) if 0 in AboveStrat_c else -1

        Drag_c[:straddleind] = 0.5 * rho_w * (uppervelocity ** 2) * pipediameter * dragcoeff * SinTheta_c[:straddleind]
        Drag_c[straddleind:] = 0.5 * rho_w * (lowervelocity ** 2) * pipediameter * dragcoeff * SinTheta_c[straddleind:]

        # Compute drag force in the grid cell that straddles the stratification:
        if straddleind != -1:
            fractionabove = (stratdepth[d1] - Depth_lr[straddleind]) / (dx * SinTheta_c[straddleind])
            Drag_c[straddleind] = fractionabove * 0.5 * rho_w * (uppervelocity ** 2) * pipediameter * dragcoeff * \
                                  SinTheta_c[straddleind] + (1 - fractionabove) * 0.5 * rho_w * (
                                              lowervelocity ** 2) * pipediameter * dragcoeff * SinTheta_c[straddleind]

        # Interpolate drag to grid edges:
        Drag_lr = np.hstack((Drag_c[0], 0.5 * (Drag_c[:-1] + Drag_c[1:]), Drag_c[-1]))

        # Compute tension
        Tension_lr = (np.cumsum(np.hstack((endmass * g, (dx * Drag_c * CosTheta_c)[::-1]))))[::-1]

        # Compute new angle
        # Compute first cell:
        Theta_c[-1] = np.pi / 2 - 0.5 * dx * Drag_lr[-1] / Tension_lr[-1]
        # Integrate for the rest
        for ii in range(xsize - 2, -1, -1):
            Theta_c[ii] = Theta_c[ii + 1] - dx * Drag_lr[ii + 1] * np.sin(Theta_c[ii + 1]) / Tension_lr[ii + 1]

        # Compute misfit:
        misfit = np.sqrt(np.mean((Theta_c - Theta_c_last) ** 2)) / (np.pi / 2)

        # Apply damping
        Theta_c = Theta_c_last + (1 - damping) * (Theta_c - Theta_c_last)

        # Compute sin and cos
        SinTheta_c = np.sin(Theta_c)
        CosTheta_c = np.cos(Theta_c)

        # Compute new depth and distance
        Distance_lr = np.cumsum(np.hstack(([0], dx * CosTheta_c)))
        Depth_lr = np.cumsum(np.hstack(([0], dx * SinTheta_c)))

        # Optional Figure
        colorvar = (min(1, iteration / 10), 0., 0.)
        plt.plot(Distance_lr, -Depth_lr, color=colorvar)
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title('Iteration = ' + str(iteration))
        # plt.show()

        # Break from loop:
        if misfit < tolerance:
            done = 1
            results.append({'Distance_lr': Distance_lr, 'Depth_lr': Depth_lr, 'd1': d1})
        elif iteration > maxiterations:
            print(d1)
            raise Exception('Error: Unable to Converge')
        else:
            iteration += 1
            Theta_c_last = np.copy(Theta_c)
# ------------------------------------------------------

# Make a figure:
for res in results:
    c = 'r' if res['d1'] == centralind else 'k'
    plt.plot(res['Distance_lr'], -res['Depth_lr'], c)
    plt.hlines(-stratdepth[res['d1']], 0, pipelength, c, '--')
    plt.text(res['Distance_lr'][-1], -res['Depth_lr'][-1] - 2, str(stratdepth[res['d1']]), verticalalignment='top',
             horizontalalignment='center')

plt.hlines(0, 0, pipelength, 'b')
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.show()


# test_var = test_vars['Depth_lr'][0]
# np.testing.assert_array_almost_equal(test_var, Depth_lr, decimal=1)
