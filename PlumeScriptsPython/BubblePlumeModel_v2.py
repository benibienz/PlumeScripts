import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from testing import test_var

"""
Beni Bienz 9/2018

Original notes (Mike Wolovick, 7/5/2018):

This script solves the equations for a two-phase flow of bubbles and
water rising from depth to the surface.  It borrows from PlumeModel_v2, a
subscript of my Flowline model that computed the buoyant meltwater plume
underneath the floating ice shelf.

v2:  the model is now for a line source plume in an environment with an
ambient horizontal current flow.  The ambient flow contributes to a net 
diagonal plume velocity that is greater than the upwards velocity of
the plume itself.  The plume model is solved on a curvilinear coordinate
that follows the midline of the plume as it rises.  The plume is
considered to have reached the surface if the upper edge hits the
surface.

Note:  The "virtual source" is one grid cell below the model grid.  ie,
Variable_ud(1) is located a distance of ds above the (mathematically
infinitessimal) source.  The analytic solution for a single-phase plume 
in a constant stratification is used to compute B_ud(1), W_ud(1), and 
Gprime_ud(1), to avoid the problem of difficulty converging in the first 
grid cell.

The script makes a figure showing outputs of plume profile, velocity, and
percent buoyancy for source depths satisfying the following conditions:
1. lowest source
2. last before bifurcation
3. first after bifurcation
4. target buoyancy at pycnocline
5. highest source
"""

# Parameters
r0 = 1e-4  # m
alpha = .083  # unitless
da = 10  # m
rho_m = 1025  # kg/m^3
u_m = 0.30  # m/s
dp = 150  # m
nsquared = 1e-6  # s^-2
shear = u_m / 150  # s^-1   this 150 should prob be dp
gamma = 1 / 150  # m^-1
g = 9.8  # m/s^2
mu = 1e-3  # Pa*s
l = 3e2  # m
pumppower = 3e5  # W
sourcedepth = 300  # m
wsterminationfactor = 0  # unitless

# Numeric Parameters:
ssize = 3000  # integer
sfactor = 3  # unitless>1
tolerance = 1e-6  # unitless
miniterations = 2  # integer
maxiterations = 300  # integer
initialdamping = .95  # unitless [0,1)
finaldamping = .25  # unitless ]0,1)
efoldingcells = 3  # unitless

# Compute richardson number: (Ri=N^2/shear^2)
ri = nsquared / (shear ** 2)

# Check richardson number:
if ri < 0.25:
    raise Exception('Ambient Profile is Unstable to Shear Instabilities.')

# Compute density gradient:
rhograd = rho_m * nsquared / g

# Compute average density:
rhobar = rho_m + (0.5 * rhograd * (sourcedepth - dp)) * (sourcedepth - dp) / sourcedepth

# Compute source density:
rho_s = rho_m + rhograd * (sourcedepth - dp)

# Compute flow rate:
flowrate = pumppower / (rhobar * g * sourcedepth * l)  # m^2/s

# Compute buoyancy rate:
buoyancyrate = flowrate * g  # m^3/s^3

# Guess vertical velocity:
wguess = (buoyancyrate / (2 * alpha)) ** (1 / 3)

# Define along-plume coordinate:
S_ud = np.linspace(0, sourcedepth * sfactor, ssize + 1)

# Compute ds:
ds = S_ud[1] - S_ud[0]

# Set up plume model grids:
X_ud = np.zeros((ssize + 1, 1))
Depth_ud = np.zeros((ssize + 1, 1))
Z_ud = np.zeros((ssize + 1, 1))
B_ud = np.zeros((ssize + 1, 1))
W_ud = np.zeros((ssize + 1, 1))
Velocity_ud = np.zeros((ssize + 1, 1))
Gprime_ud = np.zeros((ssize + 1, 1))
F_ud = np.zeros((ssize + 1, 1))
Gprime_a_ud = np.zeros((ssize + 1, 1))
U_a_ud = np.zeros((ssize + 1, 1))
NutrientVolume_ud = np.zeros((ssize + 1, 1))
NutrientConcentration_a_ud = np.zeros((ssize + 1, 1))
Ws_ud = np.zeros((ssize + 1, 1))
BubbleRadius_ud = np.zeros((ssize + 1, 1))
Rho_a_ud = np.zeros((ssize + 1, 1))
SinTheta_ud = np.zeros((ssize + 1, 1))
CosTheta_ud = np.zeros((ssize + 1, 1))

# Initialize variables:
# geometry:
X_ud[0] = 0
Z_ud[0] = 0
Depth_ud[0] = sourcedepth

# slip velocity:
BubbleRadius_ud[0] = r0
ws_turb = np.sqrt(r0 * g)
ws_lam = (r0 ** 2) * rho_s * g / (3 * mu)
Ws_ud[0] = 1 / np.sqrt(1 / (ws_lam ** 2) + 1 / (ws_turb ** 2))

# ambient variables:
Rho_a_ud[0] = rho_m + rhograd * (sourcedepth - dp)
Gprime_a_ud[0] = 0
U_a_ud[0] = max(0, u_m - shear * (sourcedepth - dp))
NutrientConcentration_a_ud[0] = gamma * (sourcedepth - dp)

# model (water) variables:
B_ud[0] = 2 * alpha * ds
W_ud[0] = wguess
Gprime_ud[0] = 0
Velocity_ud[0] = np.sqrt(W_ud[0] ** 2 + U_a_ud[0] ** 2)

# trig terms:
SinTheta_ud[0] = U_a_ud[0] / Velocity_ud[0]
CosTheta_ud[0] = W_ud[0] / Velocity_ud[0]

# gas fraction:
F_ud[0] = flowrate / (B_ud[0] * (Velocity_ud[0] + Ws_ud[0] * CosTheta_ud[0]))


# Run Model:
# Loop through grid cells:
hasterminated = 0
reachedsurface = 0

for ii in range(ssize):

    # Use forward Euler as the first guess:
    B_ud[ii + 1] = B_ud[ii]
    W_ud[ii + 1] = W_ud[ii]
    Gprime_ud[ii + 1] = Gprime_ud[ii]
    F_ud[ii + 1] = F_ud[ii]
    Depth_ud[ii + 1] = Depth_ud[ii] - ds * W_ud[ii] / Velocity_ud[ii]
    U_a_ud[ii + 1] = max(0, u_m - shear * max(0, Depth_ud[ii + 1] - dp))
    Velocity_ud[ii + 1] = Velocity_ud[ii]
    
    # Record guesses:
    lastw = W_ud[ii + 1].copy()
    lastb = B_ud[ii + 1].copy()
    lastgprime = Gprime_ud[ii + 1].copy()
    lastf = F_ud[ii + 1].copy()
    
    # Compute this damping coefficient:
    damping = finaldamping + (initialdamping - finaldamping) * np.exp(-ii / efoldingcells)
    
    # Iterate for convergence:
    done_gridcell = 0
    numiterations = 1

    while done_gridcell == 0:
        
        # Compute trig terms:
        SinTheta_ud[ii + 1] = U_a_ud[ii + 1] / Velocity_ud[ii + 1]
        CosTheta_ud[ii + 1] = W_ud[ii + 1] / Velocity_ud[ii + 1]

        # Advance depth, elevation, and distance:
        Depth_ud[ii + 1] = Depth_ud[ii] - ds * CosTheta_ud[ii + 1]
        Z_ud[ii + 1] = Z_ud[ii] + ds * CosTheta_ud[ii + 1]
        X_ud[ii + 1] = X_ud[ii] + ds * SinTheta_ud[ii + 1]

        # Check if the plume hit the surface:
        if Depth_ud[ii + 1] - 0.5 * B_ud[ii + 1] * SinTheta_ud[ii + 1] < 0:
            hasterminated = 1
            reachedsurface = 1
            break

        # Define ambient profiles:
        Rho_a_ud[ii + 1] = rho_m + rhograd * max(0, Depth_ud[ii + 1] - dp)
        Gprime_a_ud[ii + 1] = g * (rho_s - Rho_a_ud[ii + 1]) / rho_s
        U_a_ud[ii + 1] = max(0, u_m - shear * max(0, Depth_ud[ii + 1] - dp))
        NutrientConcentration_a_ud[ii + 1] = gamma * max(0, Depth_ud[ii + 1] - dp)

        # Compute bubble radius:
        BubbleRadius_ud[ii + 1] = r0 * ((sourcedepth + da) / (da + Depth_ud[ii + 1])) ** (1 / 3)

        # Compute bubble slip velocity:
        ws_turb = np.sqrt(BubbleRadius_ud[ii + 1] * g)
        ws_lam = (BubbleRadius_ud[ii + 1] ** 2) * Rho_a_ud[ii + 1] * g / (3 * mu)
        Ws_ud[ii + 1] = 1 / np.sqrt(1 / ws_lam ** 2 + 1 / ws_turb ** 2)

        # Compute buoyancy of the plume:
        buoyancy = F_ud[ii + 1] * g + (1 - F_ud[ii + 1]) * Gprime_ud[ii + 1] - Gprime_a_ud[ii + 1]

        # Compute volume entrainment:
        entrainment = 2 * alpha * Velocity_ud[ii + 1]

        # Advance volume flux:
        influx = B_ud[ii] * Velocity_ud[ii] * (1 - F_ud[ii])
        outflux = influx + entrainment * ds

        # Advance momentum flux:
        inmomentum = influx * W_ud[ii]
        outmomentum = inmomentum + B_ud[ii + 1] * buoyancy * ds

        # Advance buoyancy flux:
        inbuoyancy = influx * Gprime_ud[ii]
        outbuoyancy = inbuoyancy + entrainment * Gprime_a_ud[ii + 1] * ds

        # Advance air flux:
        inairflux = B_ud[ii] * F_ud[ii] * (Velocity_ud[ii] + Ws_ud[ii] * CosTheta_ud[ii])
        outairflux = inairflux + ds * (
                    inairflux * CosTheta_ud[ii + 1] / (Depth_ud[ii + 1] + da) - Ws_ud[ii + 1] * F_ud[ii + 1] *
                    SinTheta_ud[ii + 1])

        # Advance primitive variables:
        W_ud[ii + 1] = outmomentum / outflux
        Velocity_ud[ii + 1] = np.sqrt(U_a_ud[ii + 1] ** 2 + W_ud[ii + 1] ** 2)
        Gprime_ud[ii + 1] = outbuoyancy / outflux
        B_ud[ii + 1] = outflux / (Velocity_ud[ii + 1] * (1 - F_ud[ii + 1]))
        F_ud[ii + 1] = outairflux / (B_ud[ii + 1] * (Velocity_ud[ii + 1] + Ws_ud[ii + 1] * CosTheta_ud[ii + 1]))

        # Check air fraction:
        if F_ud[ii + 1] < 0 or F_ud[ii + 1] > 1:
            raise Exception('Impossible air fraction')

        # Advance nutrient volume:
        NutrientVolume_ud[ii + 1] = NutrientVolume_ud[ii] + ds * entrainment * NutrientConcentration_a_ud[ii + 1]

        # Check if the plume is moving too slowly:
        if W_ud[ii + 1] < wsterminationfactor * Ws_ud[ii + 1]:
            hasterminated = 1
            reachedsurface = 0
            break

        # Display current guess:
        # print(['b=', num2str(B_ud[ii + 1]), ', w=', num2str(W_ud[ii + 1]), ', gprime=', num2str(Gprime_ud[ii + 1]),
        #       ', F=', num2str(F_ud[ii + 1])])

        # Compute misfit:
        misfit = max([np.abs(B_ud[ii + 1] - lastb) / max(np.abs([B_ud[ii + 1], lastb])),
                      np.abs(W_ud[ii + 1] - lastw) / max(np.abs([W_ud[ii + 1], lastw])),
                      np.abs(Gprime_ud[ii + 1] - lastgprime) / max(np.abs([Gprime_ud[ii + 1], lastgprime])),
                      np.abs(F_ud[ii + 1] - lastf) / max(np.abs([F_ud[ii + 1], lastf]))])

        # Break from loop:
        if misfit < tolerance and numiterations >= miniterations:
            done_gridcell = 1
        elif numiterations > maxiterations:
            raise Exception('Unable to converge')
        else:
            # Count iterations:
            numiterations = numiterations + 1
            # Apply iteration damping:
            B_ud[ii + 1] = lastb + (1 - damping) * (B_ud[ii + 1] - lastb)
            W_ud[ii + 1] = lastw + (1 - damping) * (W_ud[ii + 1] - lastw)
            Gprime_ud[ii + 1] = lastgprime + (1 - damping) * (Gprime_ud[ii + 1] - lastgprime)
            F_ud[ii + 1] = lastf + (1 - damping) * (F_ud[ii + 1] - lastf)
            # Record last guess:
            lastw = W_ud[ii + 1].copy()
            lastb = B_ud[ii + 1].copy()
            lastgprime = Gprime_ud[ii + 1].copy()
            lastf = F_ud[ii + 1].copy()

    # Check if plume terminated:
    if hasterminated:
        break

# Remember where the plume terminated:
lastplumeind = ii

# -------------------- PLOTS -----------------------------------------

# Figure parameters (NOTE: some may not be in use anymore):
figname = 'LinePlumeModel_6plot_v4.png'
pagesize = [12, 8]  # [1x2] inches
resolution = 300  # dpi
verttextbuffer = .06  # unitless
titlebuffer = .025  # unitless
horztextbuffer = .06  # unitless
horznotextbuffer = .02  # unitless
ticklength = .02  # unitless
depthlims = [0, 300]  # [1x2] m
depthtick = 50  # m
xlims = [0, 350]  # [1x2] m
xtick = 50  # m
slims = [0, 450]  # m
stick = 100  # m
stick_geom = 100  # m
vellims = [0, .5001]  # [1x2] m/s
veltick = .1  # m/s
deltarholims = [1e-7, 1e-2]  # [1x2] unitless
flims = [.999e-7, 1e-2]  # [1x2] unitless
gasfluxlims = [0, .3001]  # [1x2] m^3/s
gasfluxtick = .05  # m^3/s
nutrientfluxlims = [0, 7.5e3]  # [1x2] m^3/s
nutrientfluxtick = 2.5e3  # m^3/s
wcolor = 'b'  # valid color string or [R,G,B]
ucolor = 'g'  # valid color string or [R,G,B]
wscolor = 'r'  # valid color string or [R,G,B]
vcolor = 'k'  # valid color string or [R,G,B]

fig, [[geometry_ax, velocity_ax, bubble_ax], [buoyancy_ax, gas_ax, flux_ax]] = plt.subplots(nrows=2, ncols=3, figsize=(19, 10))

# First subplot
geometry_ax.set_title('a) Plume Geometry')
geometry_ax.invert_yaxis()
geometry_ax.set_xlabel('Downstream Distance (m)')
geometry_ax.set_ylabel('Depth (m)')

# Plot patch for plume width:
patch = Polygon([
    *zip(
        list((X_ud[:lastplumeind] + 0.5 * B_ud[:lastplumeind] * W_ud[:lastplumeind] / Velocity_ud[:lastplumeind]).flatten()),
        list((Depth_ud[:lastplumeind] + 0.5 * B_ud[:lastplumeind] * U_a_ud[:lastplumeind] / Velocity_ud[:lastplumeind]).flatten())),
    *zip(
        list(np.flipud(X_ud[:lastplumeind] - 0.5 * B_ud[:lastplumeind] * W_ud[:lastplumeind] / Velocity_ud[:lastplumeind]).flatten()),
        list(np.flipud(Depth_ud[:lastplumeind] - 0.5 * B_ud[:lastplumeind] * U_a_ud[:lastplumeind] / Velocity_ud[:lastplumeind]).flatten()))
    ], facecolor='0.9', edgecolor='0.5')
geometry_ax.add_patch(patch)

# Plot plume centerline:
geometry_ax.plot(X_ud[:ii + 1], Depth_ud[:ii + 1], 'k')

# Plot pycnocline top and 0 line:
geometry_ax.hlines(dp, *xlims, 'k', '--', linewidth=0.75)
geometry_ax.hlines(0, *xlims, 'b', '--')

# Plot zero position:
if xlims[0] < 0:
    geometry_ax.vlines(0, *depthlims, 'k', '--')

# Plot s ticks:
for ii in range(1, int(S_ud[lastplumeind] / stick_geom) + 1):
    thisx = np.interp(ii * stick_geom, S_ud[:lastplumeind].flatten(), X_ud[:lastplumeind].flatten())
    thisdepth = np.interp(ii * stick_geom, S_ud[:lastplumeind].flatten(), Depth_ud[:lastplumeind].flatten())
    geometry_ax.plot(thisx, thisdepth, 'k', marker='D', markersize=5)
    geometry_ax.text(thisx + 5, thisdepth + 5, str(ii * stick_geom) + ' m', verticalalignment='top', horizontalalignment='left')

# Second subplot
velocity_ax.set_title('b) Velocity Profiles')
velocity_ax.plot(S_ud[:lastplumeind], U_a_ud[:lastplumeind], ucolor, label='Horizontal Velocity (u)')
velocity_ax.plot(S_ud[:lastplumeind], W_ud[:lastplumeind], wcolor, label='Vertical velocity (w)')
velocity_ax.plot(S_ud[:lastplumeind], Ws_ud[:lastplumeind], wscolor, label='Bubble slip velocity (ws)')
velocity_ax.plot(S_ud[:lastplumeind], Velocity_ud[:lastplumeind], vcolor, label='Net velocity (v*)')

# Formatting
velocity_ax.set_xlim(slims)
velocity_ax.set_xlabel('Along-Plume Distance (m)')
velocity_ax.set_ylim(vellims)
velocity_ax.set_ylabel('Velocity (m/s)')
velocity_ax.legend()

# Third subplot:
bubble_ax.set_title('c) Bubble Content')

# Plot bubble fraction:
bubble_ax.plot(S_ud[:lastplumeind], F_ud[:lastplumeind], 'k')

# Formatting
bubble_ax.set_xlim(slims)
bubble_ax.set_xlabel('Along-Plume Distance (m)')
bubble_ax.set_yscale('log')
bubble_ax.set_ylim(flims)
bubble_ax.set_ylabel('Gas Fraction (unitless)')

# Fourth subplot:
buoyancy_ax.set_title('d) Aggregate Plume Buoyancy')

# Compute aggregate buoyancy:
AggregateBuoyancy_ud = F_ud + (1 - F_ud) * Gprime_ud / g - Gprime_a_ud / g

# Check if there is negative buoyancy:
if min(AggregateBuoyancy_ud) < 0:
    buoyancy_ax.plot(S_ud[:lastplumeind], max(0, AggregateBuoyancy_ud[:lastplumeind]), 'r', label='Positive Buoyancy')
    buoyancy_ax.plot(S_ud[:lastplumeind], -min(0, AggregateBuoyancy_ud[:lastplumeind]), 'b', label='Negative Buoyancy')
    buoyancy_ax.legend()
else:
    hasnegativebuoyancy = 0  # Remember
    buoyancy_ax.plot(S_ud[:lastplumeind], AggregateBuoyancy_ud[:lastplumeind], 'k')

# Formatting:
buoyancy_ax.set_xlim(slims)
buoyancy_ax.set_xlabel('Along-Plume Distance (m)')
buoyancy_ax.set_yscale('log')
buoyancy_ax.set_ylim(deltarholims)
buoyancy_ax.set_ylabel('Relative Buoyancy (Delta (rho/rho))')


# Fifth subplot:
gas_ax.set_title('e) Gas Budget')

# Plot fluxes
gas_ax.plot(0, flowrate * l, 'k', marker='o', markersize=5, label='Initial Flux')
GasFlux_ud = B_ud * F_ud * (Velocity_ud + Ws_ud * CosTheta_ud)
gas_ax.plot(S_ud[1:lastplumeind], np.cumsum(
    ds * l * CosTheta_ud[1:lastplumeind] * GasFlux_ud[:lastplumeind - 1] / (Depth_ud[1:lastplumeind] + da)), 'b',
             label='Expansion')
gas_ax.plot(S_ud[1:lastplumeind],
             np.cumsum(ds * l * Ws_ud[1:lastplumeind] * F_ud[1:lastplumeind] * SinTheta_ud[1:lastplumeind]), 'r',
             label='Detrainment')
gas_ax.plot(S_ud[:lastplumeind], GasFlux_ud[:lastplumeind] * l, 'k', label='Total Flux')

# Formatting
gas_ax.set_xlim(slims)
gas_ax.set_xlabel('Along-Plume Distance (m)')
gas_ax.set_ylim(gasfluxlims)
gas_ax.set_ylabel('Flux (m^3/s)')
gas_ax.legend()

# Sixth subplot:
flux_ax.set_title('f) Nutrient-Weighted Water Flux')

# Plot total nutrient flux:
flux_ax.plot(S_ud[:lastplumeind], NutrientVolume_ud[:lastplumeind] * l, 'k')

# Formatting
flux_ax.set_xlim(slims)
flux_ax.set_xlabel('Along-Plume Distance (m)')
flux_ax.set_ylim(nutrientfluxlims)
flux_ax.set_ylabel('Flux (m^3/s)')

plt.show()
