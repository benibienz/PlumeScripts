import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from testing import test_var, test_vars
import pickle

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

v2a:  the model now computes probability of reaching the surface and
expected nutrient flux for a sampling of source depth and gas flow rate.
For each (depth,flux) sample, the model is run for a sampling of ambient
conditions (stratification and horizontal velocity).  The total parameter
space is four dimensional, but the two dimensions corresponding to
ambient conditions are averaged over to produce a result that is
probabilistic with respect to the two parameters that are design choices.

Sampling is linear for depth and flow rate, and logarythmic for velocity
and stratification.

v2b:  The model computes a single slice through source depth/gas flow
rate space for fixed values of stratification and current velocity.  In
addition, the ambient profiles are specified differently than before.
Density has a sigmoidal profile (erf) characterized by a set density
change, a middle depth, and a transition lengthscale.  Horizontal
currents have the same form, with the transition lengthscale set so that
the minimum Richardson number is the critical value.

Nutrient flux in the output is flux to the surface (or specifically, flux
to the "nearsurfacedepth").

Things plotted:
Ambient density
ambient velocity
termination depth
downstream distance
min(w/ws)
pump power
nutrient flux
nutrient flux/pump power

Note:  The "virtual source" is one grid cell below the model grid.  ie,
Variable_ud[0] is located a distance of ds above the (mathematically
infinitessimal) source.  The analytic solution for a single-phase plume
in a constant stratification is used to compute B_ud[0], W_ud[0], and
Gprime_ud[0], to avoid the problem of difficulty converging in the first
grid cell.
"""

## Parameters:

# File names:
inputfile = 'LinePlumeModel_ParamSpace_v3i.p'  # can be commented (script is display only if uncommented)
# inputfile = None
outputfile = 'LinePlumeModel_ParamSpace_v3i.p'
# figname = LinePlumeModel_ParamSpace_v3d.png

# Constant Parameters:
r0 = 1e-4  # m
alpha = .083  # unitless
da = 10  # m
rho_m = 1025  # kg/m^3
middepth = 200  # m
transitiondepthscale = 100  # m
deltarho = 1.0  # kg/m^3
u_m = .60  # m/s
ri = .25  # unitless
g = 9.8  # m/s^2
mu = 1e-3  # Pa*s
l = 3e2  # m
nearsurfacedepth = 25  # m
ConstantParameters = {'r0': r0, 'alpha': alpha, 'da': da, 'rho_m': rho_m, 'middepth': middepth,
                      'transitiondepthscale': transitiondepthscale, 'deltarho': deltarho, 'u_m': u_m, 'ri': ri, 'g': g,
                      'mu': mu, 'l': l, 'nearsurfacedepth': nearsurfacedepth}

# Variable parameters:
sourcedepthlims = [100, 400]  # [1x2] m
flowratelims = [0, 1]  # m^3/s
VariableParameters = {'sourcedepthlims': sourcedepthlims, 'flowratelims': flowratelims}

# Numeric Parameters:
ssize = 3000  # integer
sfactor = 3  # unitless>1
asize = 1000  # integer
tolerance = 1e-6  # unitless
miniterations = 2  # integer
maxiterations = 300  # integer
initialdamping = .75  # unitless [0,1)
finaldamping = .25  # unitless ]0,1)
efoldingcells = 3  # unitless
wsterminationfactor = 0  # unitless        THIS IS THE ONLY PARAMETER REMEMBERED IF YOU ARE LOADING AN INPUT FILE
numsamples_1d = 3  # integer
interpstyle = 'linear'  # valid interp1d method
d1displayinterval = 1  # integer
NumericParameters = {'ssize': ssize, 'sfactor': sfactor, 'asize': asize, 'tolerance': tolerance,
                     'miniterations': miniterations, 'maxiterations': maxiterations, 'initialdamping': initialdamping,
                     'finaldamping': finaldamping, 'efoldingcells': efoldingcells,
                     'wsterminationfactor': wsterminationfactor, 'numsamples_1d': numsamples_1d,
                     'interpstyle': interpstyle, 'd1displayinterval': d1displayinterval}

"""
Parameter space key:
d1=source depth
d2=flow rate
"""

# Create parameter space grids:
SourceDepth = np.linspace(sourcedepthlims[0], sourcedepthlims[1], numsamples_1d)[:, None]
FlowRate = np.linspace(flowratelims[0], flowratelims[1], numsamples_1d)

# Pre-allocate output matrices:
Converged = np.ones((numsamples_1d, numsamples_1d))
ReachedSurface = np.zeros((numsamples_1d, numsamples_1d), dtype=bool)
FinalWidth = np.zeros((numsamples_1d, numsamples_1d))
NutrientFlux = np.zeros((numsamples_1d, numsamples_1d))
MinVelocityRatio = np.zeros((numsamples_1d, numsamples_1d))
InitialAngle = np.zeros((numsamples_1d, numsamples_1d))
FinalAngle = np.zeros((numsamples_1d, numsamples_1d))
MeanAngle = np.zeros((numsamples_1d, numsamples_1d))
DownstreamDistance = np.zeros((numsamples_1d, numsamples_1d))
FinalCenterDepth = np.zeros((numsamples_1d, numsamples_1d))
FinalTopDepth = np.zeros((numsamples_1d, numsamples_1d))
FinalBotDepth = np.zeros((numsamples_1d, numsamples_1d))

# Compute pump power:
PumpPower = rho_m * g * np.matlib.repmat(SourceDepth, 1, numsamples_1d) * np.matlib.repmat(FlowRate, numsamples_1d, 1)

# Pre-allocate plume model grids:
X_ud = np.zeros((ssize + 1, 1))
Depth_ud = np.zeros((ssize + 1, 1))
Z_ud = np.zeros((ssize + 1, 1))
B_ud = np.zeros((ssize + 1, 1))
W_ud = np.zeros((ssize + 1, 1))
Velocity_ud = np.zeros((ssize + 1, 1))
Gprime_ud = np.zeros((ssize + 1, 1))
F_ud = np.zeros((ssize + 1, 1))
U_ud = np.zeros((ssize + 1, 1))
NutrientFlux_ud = np.zeros((ssize + 1, 1))
Ws_ud = np.zeros((ssize + 1, 1))
BubbleRadius_ud = np.zeros((ssize + 1, 1))
SinTheta_ud = np.zeros((ssize + 1, 1))
CosTheta_ud = np.zeros((ssize + 1, 1))

# Create ambient depth grid:
Depth_a = np.linspace(0, sourcedepthlims[1], asize)

# Compute ambient density:
Rho_a = rho_m + deltarho * .5 * (erf((Depth_a - middepth) / transitiondepthscale) + 1)

# Compute central stratification:
centern2 = (g / (rho_m + .5 * deltarho)) * deltarho / (np.sqrt(np.pi) * transitiondepthscale)

# Compute central shear:
centershear = np.sqrt(centern2 / ri)

# Compute transition depth scale for velocity:
transitiondepthscale_u = u_m / (np.sqrt(np.pi) * centershear)

# Compute ambient velocity:
U_a = u_m * .5 * (1 - erf((Depth_a - middepth) / transitiondepthscale_u))

# Compute nutrient concentration (assumed propto density):
NutrientConcentration_a = (Rho_a - rho_m) / deltarho

## Run Model:

# Check if the input file exists:
    # if exist(inputfile, var)
    #     # Communicate:
    #     disp(Loading previous results)
    #     # load input:
    #     load(inputfile)
    #     # Unpack parameter structures:
    #     unpack(ConstantParameters)
    #     unpack(VariableParameters)
    #     unpack(NumericParameters)
    #     # Recompute central stratification:
    #     centern2 = (g / (rho_m + .5 * deltarho)) * deltarho / (np.sqrt(pi) * transitiondepthscale)


# Loop through parameter space:
for d1 in range(numsamples_1d):
    if inputfile is not None:
        break
    for d2 in range(numsamples_1d):
        print('d1: {}, d2: {}'.format(d1, d2))

        # Check if the flow rate or source depth is zero:
        if FlowRate[d2] == 0 or SourceDepth[d1] == 0:
            # Assign degenerate values:
            ReachedSurface[d1, d2] = 0
            NutrientFlux[d1, d2] = 0
            FinalWidth[d1, d2] = 0
            MinVelocityRatio[d1, d2] = 0
            InitialAngle[d1, d2] = 90
            FinalAngle[d1, d2] = 90
            MeanAngle[d1, d2] = 90
            DownstreamDistance[d1, d2] = 0
            FinalCenterDepth[d1, d2] = SourceDepth[d1]
            FinalTopDepth[d1, d2] = SourceDepth[d1]
            FinalBotDepth[d1, d2] = SourceDepth[d1]
            # Skip to next parameter combination:
            continue

        # Compute source density:
        rho_s = interp1d(Depth_a, Rho_a, interpstyle)(SourceDepth[d1])[0]

        # Compute buoyancy rate:
        buoyancyrate = FlowRate[d2] * g / l  # m^3/s^3

        # Guess vertical velocity:
        wguess = (buoyancyrate / (2 * alpha)) ** (1 / 3)

        # Define along-plume coordinate:
        S_ud = np.linspace(0, SourceDepth[d1] * sfactor, ssize + 1)[:, None]

        # Compute ds:
        ds = S_ud[1] - S_ud[0]

        # Initialize variables:
        # geometry:
        X_ud[0] = 0
        Z_ud[0] = 0
        Depth_ud[0] = SourceDepth[d1]

        # slip velocity:
        BubbleRadius_ud[0] = r0
        ws_turb = np.sqrt(r0 * g)
        ws_lam = (r0 ** 2) * rho_s * g / (3 * mu)
        Ws_ud[0] = 1 / np.sqrt(1 / ws_lam ** 2 + 1 / ws_turb ** 2)

        # ambient variables:
        U_ud[0] = interp1d(Depth_a, U_a, interpstyle)(SourceDepth[d1])

        # model (water) variables:
        B_ud[0] = 2 * alpha * ds
        W_ud[0] = wguess
        Gprime_ud[0] = 0
        Velocity_ud[0] = np.sqrt(W_ud[0] ** 2 + U_ud[0] ** 2)
        # trig terms:
        SinTheta_ud[0] = U_ud[0] / Velocity_ud[0]
        CosTheta_ud[0] = W_ud[0] / Velocity_ud[0]
        # gas fraction:
        F_ud[0] = FlowRate[d2] / (l * B_ud[0] * (Velocity_ud[0] + Ws_ud[0] * CosTheta_ud[0]))

        # Loop through grid cells:
        hasterminated = 0
        for ii in range(ssize):
            # Use forward Euler as the first guess:
            B_ud[ii + 1] = B_ud[ii]
            W_ud[ii + 1] = W_ud[ii]
            Gprime_ud[ii + 1] = Gprime_ud[ii]
            F_ud[ii + 1] = F_ud[ii]
            Depth_ud[ii + 1] = Depth_ud[ii] - ds * W_ud[ii] / Velocity_ud[ii]
            U_ud[ii + 1] = interp1d(Depth_a, U_a, interpstyle)(Depth_ud[ii + 1])
            Velocity_ud[ii + 1] = Velocity_ud[ii]

            # Record guesses (Note: must make copies in python!):
            lastw = W_ud[ii + 1].copy()
            lastb = B_ud[ii + 1].copy()
            lastgprime = Gprime_ud[ii + 1].copy()
            lastf = F_ud[ii + 1].copy()

            # Compute this damping coefficient:
            damping = finaldamping + (initialdamping - finaldamping) * np.exp(-ii / efoldingcells)

            # Interpolate ambient properties:
            thisrho_a = interp1d(Depth_a, Rho_a, interpstyle)(Depth_ud[ii + 1])
            thisgprime_a = g * (rho_s - thisrho_a) / rho_s
            U_ud[ii + 1] = interp1d(Depth_a, U_a, interpstyle)(Depth_ud[ii + 1])
            thisnutrientconcentration_a = interp1d(Depth_a, NutrientConcentration_a, interpstyle)(Depth_ud[ii + 1])

            # Iterate for convergence:
            done_gridcell = 0
            numiterations = 1
            while done_gridcell == 0:

                # Compute trig terms:
                SinTheta_ud[ii + 1] = U_ud[ii + 1] / Velocity_ud[ii + 1]
                CosTheta_ud[ii + 1] = W_ud[ii + 1] / Velocity_ud[ii + 1]

                # Advance depth, elevation, and distance:
                Depth_ud[ii + 1] = Depth_ud[ii] - ds * CosTheta_ud[ii + 1]
                Z_ud[ii + 1] = Z_ud[ii] + ds * CosTheta_ud[ii + 1]
                X_ud[ii + 1] = X_ud[ii] + ds * SinTheta_ud[ii + 1]

                # Check if the plume hit the surface:
                if Depth_ud[ii + 1] - .5 * B_ud[ii + 1] * SinTheta_ud[ii + 1] < 0:
                    hasterminated = 1
                    ReachedSurface[d1, d2] = 1
                    break

                # Compute bubble radius:
                BubbleRadius_ud[ii + 1] = r0 * ((SourceDepth[d1] + da) / (da + Depth_ud[ii + 1])) ** (1 / 3)

                # Compute bubble slip velocity:
                ws_turb = np.sqrt(BubbleRadius_ud[ii + 1] * g)
                ws_lam = (BubbleRadius_ud[ii + 1] ** 2) * thisrho_a * g / (3 * mu)
                Ws_ud[ii + 1] = 1 / np.sqrt(1 / ws_lam ** 2 + 1 / ws_turb ** 2)

                # Compute buoyancy of the plume:
                buoyancy = F_ud[ii + 1] * g + (1 - F_ud[ii + 1]) * Gprime_ud[ii + 1] - thisgprime_a

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
                outbuoyancy = inbuoyancy + entrainment * thisgprime_a * ds

                # Advance air flux:
                inairflux = B_ud[ii] * F_ud[ii] * (Velocity_ud[ii] + Ws_ud[ii] * CosTheta_ud[ii])
                outairflux = inairflux + ds * (
                        inairflux * CosTheta_ud[ii + 1] / (Depth_ud[ii + 1] + da) - Ws_ud[ii + 1] * F_ud[
                    ii + 1] * SinTheta_ud[ii + 1])

                # Advance primitive variables:
                W_ud[ii + 1] = outmomentum / outflux
                Velocity_ud[ii + 1] = np.sqrt(U_ud[ii + 1] ** 2 + W_ud[ii + 1] ** 2)
                Gprime_ud[ii + 1] = outbuoyancy / outflux
                B_ud[ii + 1] = outflux / (Velocity_ud[ii + 1] * (1 - F_ud[ii + 1]))
                F_ud[ii + 1] = outairflux / (
                        B_ud[ii + 1] * (Velocity_ud[ii + 1] + Ws_ud[ii + 1] * CosTheta_ud[ii + 1]))

                # Check air fraction:
                if F_ud[ii + 1] < 0 or F_ud[ii + 1] > 1:
                    raise Exception('Impossible air fraction')

                # Advance nutrient volume flux:
                NutrientFlux_ud[ii + 1] = NutrientFlux_ud[
                                              ii] + ds * entrainment * thisnutrientconcentration_a  # m^2/s

                # Check if the plume is moving too slowly:
                if W_ud[ii + 1] < wsterminationfactor * Ws_ud[ii + 1]:
                    hasterminated = 1
                    ReachedSurface[d1, d2] = 0
                    break

                # Display current guess:
                # disp([b=,num2str(B_ud[ii + 1]),, w=,num2str(W_ud[ii + 1]),, gprime=,num2str(Gprime_ud[ii + 1]),, F=,num2str(F_ud[ii + 1])])

                # Compute misfit:
                misfit = np.nanmax([np.abs(B_ud[ii + 1] - lastb) / max(np.abs([B_ud[ii + 1], lastb])),
                                    np.abs(W_ud[ii + 1] - lastw) / max(np.abs([W_ud[ii + 1], lastw])),
                                    np.abs(Gprime_ud[ii + 1] - lastgprime) / max(
                                        np.abs([Gprime_ud[ii + 1], lastgprime])),
                                    np.abs(F_ud[ii + 1] - lastf) / max(np.abs([F_ud[ii + 1], lastf]))])

                # Break from loop:
                if misfit < tolerance and numiterations >= miniterations:
                    done_gridcell = 1
                elif numiterations > maxiterations:
                    # Record failure to converge:
                    Converged[d1, d2] = 0
                    # Break from loops:
                    done_gridcell = 1
                    hasterminated = 1
                    # Throw error message:
                    # error(Unable to converge)
                else:
                    # Count iterations:
                    numiterations = numiterations + 1
                    # Apply iteration damping:
                    B_ud[ii + 1] = lastb + (1 - damping) * (B_ud[ii + 1] - lastb)
                    W_ud[ii + 1] = lastw + (1 - damping) * (W_ud[ii + 1] - lastw)
                    Gprime_ud[ii + 1] = lastgprime + (1 - damping) * (Gprime_ud[ii + 1] - lastgprime)
                    F_ud[ii + 1] = lastf + (1 - damping) * (F_ud[ii + 1] - lastf)
                    # Record last guess (MUST COPY IN PYTHON):
                    lastw = W_ud[ii + 1].copy()
                    lastb = B_ud[ii + 1].copy()
                    lastgprime = Gprime_ud[ii + 1].copy()
                    lastf = F_ud[ii + 1].copy()

            # Check if plume terminated:
            if hasterminated:
                break

        # Compute output variables:
        if ii > 1:
            NutrientFlux[d1, d2] = NutrientFlux_ud[ii] * max(0, min(1, (
                        nearsurfacedepth - (Depth_ud[ii] - .5 * B_ud[ii] * SinTheta_ud[ii])) / (                                                                                B_ud[ii] * SinTheta_ud[ii])))  # m^2/s
            FinalWidth[d1, d2] = B_ud[ii]
            MinVelocityRatio[d1, d2] = min(W_ud[1:ii + 1] / Ws_ud[1:ii + 1])
            Theta_ud = np.rad2deg(np.arcsin(SinTheta_ud))
            InitialAngle[d1, d2] = Theta_ud[1]
            FinalAngle[d1, d2] = Theta_ud[ii]
            MeanAngle[d1, d2] = np.mean(Theta_ud[1:ii + 1])
            DownstreamDistance[d1, d2] = X_ud[ii]
            FinalCenterDepth[d1, d2] = Depth_ud[ii]
            FinalTopDepth[d1, d2] = Depth_ud[ii] - .5 * B_ud[ii] * SinTheta_ud[ii]
            FinalBotDepth[d1, d2] = Depth_ud[ii] + .5 * B_ud[ii] * SinTheta_ud[ii]

    # test_var(MinVelocityRatio, 'MinVelocityRatio')

    # Save output:
    params = [ConstantParameters, VariableParameters, NumericParameters, Rho_a, Depth_a, NutrientConcentration_a,
              thisgprime_a, thisnutrientconcentration_a, thisrho_a, U_a, SourceDepth, FlowRate, Converged,
              ReachedSurface, NutrientFlux, FinalWidth, MinVelocityRatio, InitialAngle, FinalAngle, MeanAngle,
              DownstreamDistance, PumpPower, FinalCenterDepth, FinalTopDepth, FinalBotDepth, d1, d2]
    with open(outputfile, 'wb') as f:
        pickle.dump(params, f)


if inputfile is not None:
    with open(inputfile, 'rb') as f:
        (ConstantParameters, VariableParameters, NumericParameters, Rho_a, Depth_a, NutrientConcentration_a,
              thisgprime_a, thisnutrientconcentration_a, thisrho_a, U_a, SourceDepth, FlowRate, Converged,
              ReachedSurface, NutrientFlux, FinalWidth, MinVelocityRatio, InitialAngle, FinalAngle, MeanAngle,
              DownstreamDistance, PumpPower, FinalCenterDepth, FinalTopDepth, FinalBotDepth, d1, d2) = pickle.load(f)

    params = [Rho_a, Depth_a, NutrientConcentration_a,
              thisgprime_a, thisnutrientconcentration_a, thisrho_a, U_a, SourceDepth, FlowRate, Converged,
              ReachedSurface, NutrientFlux, FinalWidth, MinVelocityRatio, InitialAngle, FinalAngle, MeanAngle,
              DownstreamDistance, PumpPower, FinalCenterDepth, FinalTopDepth, FinalBotDepth]

    params_s = ['Rho_a', 'Depth_a', 'NutrientConcentration_a',
     'thisgprime_a', 'thisnutrientconcentration_a', 'thisrho_a', 'U_a', 'SourceDepth', 'FlowRate', 'Converged',
     'ReachedSurface', 'NutrientFlux', 'FinalWidth', 'MinVelocityRatio', 'InitialAngle', 'FinalAngle', 'MeanAngle',
     'DownstreamDistance', 'PumpPower', 'FinalCenterDepth', 'FinalTopDepth', 'FinalBotDepth']

    # test_vars(params, params_s)

# Figure parameters:
pagesize = [12, 15]  # [1x2] inches
resolution = 300  # dpi
verttextbuffer = .04  # unitless
titlebuffer = .03  # unitless
horztextbuffer = .06  # unitless
horznotextbuffer = .03  # unitless
ticklength = .02  # unitless
depthlims = [0, 400]  # [1x2] m
depthtick = 50  # m
rholims = [1025, 1026]  # [1x2] kg/m^3
rhotick = .25  # kg/m^3
ulims = [0, 60]  # [1x2] cm/s
utick = 10  # cm/s
flowratetick = .1  # m^3/s
distlims = [0, 500]  # [1x2] m
disttick = 100  # m
finaldepthlims = [0, 250]  # [1x2] m
finaldepthtick = 50  # m
powerlims = [0, 4]  # [1x2] MW
powertick = 1  # MW
wratiolims = [0, 20]  # [1x2] unitless
wratiotick = 5  # unitless
fluxavglims = [0, 6]  # [1x2] 10^3 m^3/s
fluxavgtick = 1  # 10^3 m^3/s
fluxratiolims = [0, 3]  # [1x2] (m^3/s)/kW
fluxratiotick = .5  # (m^3/s)/kW
numcontours = 30  # integer

fig = plt.figure(figsize=(19, 10))
# Plot density:
ax0 = plt.subplot(421)
ax0.set_title('a) Ambient Density)')
ax0.invert_yaxis()
ax0.set_ylabel('Depth (m)')
ax0.set_xlabel('Denisty (kg/m^3)')
ax0.plot(Rho_a, Depth_a, 'k')
# Label N^2:
ax0.plot(rho_m + .5 * deltarho, middepth, 'k', marker='D', markersize=5)
lowpow10 = 10 ** np.log10(centern2)
ax0.text(rho_m + .5 * deltarho, middepth - 10, 'N^2 = {:.2e}s^-2'.format(lowpow10), verticalalignment='bottom', horizontalalignment='left')

# Plot current
ax1 = plt.subplot(422, sharey=ax0)
ax1.set_title('b) Ambient Horizontal Current')
ax1.set_xlabel('Velocity (cm/s)')
ax1.plot(100 * U_a, Depth_a, 'k')
# Label Richardson number:
ax1.plot(100 * .5 * u_m, middepth, 'k', marker='D', markersize=5)
ax1.text(100 * .5 * u_m, middepth - 10, 'Ri = {}'.format(ri), verticalalignment='bottom', horizontalalignment='left'),

# Plot paramteric studies
X = FlowRate.flatten()
Y = SourceDepth.flatten()

# Plot bottom 2 subplots first to satisfy sharex argument
ax6 = plt.subplot(427)
ax6.set_title('g) Nutrient Flux to the Surface (10^3 m^3/s)')
ax6.set_ylabel('Source Depth (m)')
ax6.set_xlabel('Air Flow Rate (m^3/s)')
ax6.invert_yaxis()
ax6_c = ax6.contourf(X, Y, l * NutrientFlux / 1000, np.linspace(fluxavglims[0], fluxavglims[1], numcontours), extend='max')
fig.colorbar(ax6_c, ax=ax6, ticks=np.arange(fluxavglims[0], fluxavglims[1] + 0.01, fluxavgtick))
ax6.contour(X, Y, ReachedSurface & (FinalCenterDepth < middepth), [0.5], colors='w')  # reached surface boundary
ax6.hlines(middepth, *flowratelims, 'w', '--')  # mid depth

ax7 = plt.subplot(428, sharey=ax6)
ax7.set_title('h) Ratio of Nutrient Flux to Pump Power ((m^3/s) / kW)')
ax7.set_xlabel('Air Flow Rate (m^3/s)')
ax7_c = ax7.contourf(X, Y, 1000 * l * NutrientFlux / PumpPower, np.linspace(fluxratiolims[0], fluxratiolims[1], numcontours), extend='max')
fig.colorbar(ax7_c, ax=ax7, ticks=np.arange(fluxratiolims[0], fluxratiolims[1] + 0.01, fluxratiotick))
ax7.contour(X, Y, ReachedSurface & (FinalCenterDepth < middepth), [0.5], colors='w')  # reached surface boundary
ax7.hlines(middepth, *flowratelims, 'w', '--')  # mid depth

# Plot rest of subplots in order
ax2 = plt.subplot(423, sharex=ax6)
ax2.set_title('c) Downstream Distance (m)')
ax2.set_ylabel('Source Depth (m)')
ax2.invert_yaxis()
ax2_c = ax2.contourf(X, Y, DownstreamDistance, np.linspace(distlims[0], distlims[1], numcontours), extend='max')
fig.colorbar(ax2_c, ax=ax2, ticks=np.arange(distlims[0], distlims[1] + 0.01, disttick))
ax2.contour(X, Y, ReachedSurface & (FinalCenterDepth < middepth), [0.5], colors='w')  # reached surface boundary
ax2.hlines(middepth, *flowratelims, 'w', '--')  # mid depth

ax3 = plt.subplot(424, sharey=ax2, sharex=ax7)
ax3.set_title('d) Termination Depth (Plume Center) (m)')
ax3_c = ax3.contourf(X, Y, FinalCenterDepth, np.linspace(finaldepthlims[0], finaldepthlims[1], numcontours), extend='max')
fig.colorbar(ax3_c, ax=ax3, ticks=np.arange(finaldepthlims[0], finaldepthlims[1] + 0.01, finaldepthtick))
ax3.contour(X, Y, ReachedSurface & (FinalCenterDepth < middepth), [0.5], colors='w')  # reached surface boundary
ax3.hlines(middepth, *flowratelims, 'w', '--')  # mid depth

ax4 = plt.subplot(425, sharex=ax6)
ax4.set_title('e) Minimum w/w_s Ratio')
ax4.set_ylabel('Source Depth (m)')
ax4.invert_yaxis()
ax4_c = ax4.contourf(X, Y, MinVelocityRatio, np.linspace(wratiolims[0], wratiolims[1], numcontours), extend='max')
fig.colorbar(ax4_c, ax=ax4, ticks=np.arange(wratiolims[0], wratiolims[1] + 0.01, wratiotick))
ax4.contour(X, Y, ReachedSurface & (FinalCenterDepth < middepth), [0.5], colors='w')  # reached surface boundary
ax4.hlines(middepth, *flowratelims, 'w', '--')  # mid depth

ax5 = plt.subplot(426, sharey=ax4, sharex=ax7)
ax5.set_title('f) Pump Power (kW)')
ax5_c = ax5.contourf(X, Y, PumpPower / 1e6, np.linspace(powerlims[0], powerlims[1], numcontours), extend='max')
fig.colorbar(ax5_c, ax=ax5, ticks=np.arange(powerlims[0], powerlims[1] + 0.01, powertick))
ax5.contour(X, Y, ReachedSurface & (FinalCenterDepth < middepth), [0.5], colors='w')  # reached surface boundary
ax5.hlines(middepth, *flowratelims, 'w', '--')  # mid depth


# # Call seventh subplot:
# subplot(Position, Boxes
# {4, 1})
# # Plot nutrient flux to the surface:
# contourf(FlowRate, SourceDepth, l * NutrientFlux / 1000,
#          np.linspace(fluxavglims[0], fluxavglims(2), numcontours))
# hold
# on
# # Plot reached surface boundary:
# contour(FlowRate, SourceDepth, ReachedSurface & FinalCenterDepth < middepth, [.5, .5], Color, w,
#         LineWidth, 1)
# # Plot mid depth:
# plot(flowratelims, middepth * [1, 1], Color, w, LineStyle, --, LineWidth, 1)
# # Set up stuff:
# caxis(fluxavglims)
# set(gca, ydir, reverse)
# ylim(sourcedepthlims)
# set(gca, YTick, depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims(2) / depthtick)])
# ylabel(Source Depth (m))
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, XTick, flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims(2) * 1.0001 / flowratetick)])  # KLUDGE
# xlabel(Air Flow Rate (m^3/s))
# set(gca, TickLength, [ticklength, .025])
# set(gca, TickDir, tickdir)
# # Color bar:
# hc5 = colorbar
# set(get(hc5, Ylabel), String, 10^3 m^3/s)
# set(hc5, YTick, fluxavgtick * [ceil(fluxavglims[0] / fluxavgtick): 1:floor(fluxavglims(2) / fluxavgtick)])
# # Title:
# title(g) Nutrient Flux to the Surface)
# 
# # Call eigth subplot:
# subplot(Position, Boxes
# {4, 2})
# # Plot nutrient flux to pump power ratio:
# contourf(FlowRate, SourceDepth, 1000 * l * NutrientFlux. / PumpPower,
#          np.linspace(fluxratiolims[0], fluxratiolims(2), numcontours))
# hold
# on
# # Plot reached surface boundary:
# contour(FlowRate, SourceDepth, ReachedSurface & FinalCenterDepth < middepth, [.5, .5], Color, w,
#         LineWidth, 1)
# # Plot mid depth:
# plot(flowratelims, middepth * [1, 1], Color, w, LineStyle, --, LineWidth, 1)
# # Set up stuff:
# caxis(fluxratiolims)
# set(gca, ydir, reverse)
# ylim(sourcedepthlims)
# set(gca, YTick, depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims(2) / depthtick)])
# set(gca, YTickLabel, [])
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, XTick, flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims(2) * 1.0001 / flowratetick)])  # KLUDGE
# xlabel(Air Flow Rate (m^3/s))
# set(gca, TickLength, [ticklength, .025])
# set(gca, TickDir, tickdir)
# # Color bar:
# hc6 = colorbar
# set(get(hc6, Ylabel), String, (m^3/s)/kW)
# set(hc6, YTick, fluxratiotick * [ceil(fluxratiolims[0] / fluxratiotick): 1:floor(
#     fluxratiolims(2) / fluxratiotick)])
# # Title:
# title(h) Ratio of Nutrient Flux to Pump Power)

fig.tight_layout()
plt.show()


