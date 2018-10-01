import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from testing import test_var
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

Nutrient flux in the output is flux to the surface (or specifically, flux
to the "nearsurfacedepth").

Things plotted:
pump power
probability of reaching surface
downstream distance
mean(min(w/ws))
average nutrient flux
variability in nutrient flux

Note:  The "virtual source" is one grid cell below the model grid.  ie,
Variable_ud[0] is located a distance of ds above the (mathematically
infinitessimal) source.  The analytic solution for a single-phase plume
in a constant stratification is used to compute B_ud[0], W_ud[0], and
Gprime_ud[0], to avoid the problem of difficulty converging in the first
grid cell.
"""

# Parameters:

# File names:
# inputfile = None
inputfile = 'LinePlumeModel_ParamSpace_v2.p'  # uncomment for plot only
outputfile = 'LinePlumeModel_ParamSpace_v2.p'
# figname = 'LinePlumeModel_ParamSpace_v2.png'

# Constant Parameters:
r0 = 1e-4  # m
alpha = .083  # unitless
da = 10  # m
rho_m = 1025  # kg/m^3
dp = 150  # m
ri = .25  # unitless
g = 9.8  # m/s^2
mu = 1e-3  # Pa*s
l = 3e2  # m
nearsurfacedepth = 25  # m
ConstantParameters = {'r0': r0, 'alpha': alpha, 'da': da, 'rho_m': rho_m, 'dp': dp, 'ri': ri, 'g': g, 'mu': mu, 'l': l,
                      'nearsurfacedepth': nearsurfacedepth}

# Variable parameters:
umlims = [.1, 1]  # [1x2] m/s
nsquaredlims = [1e-6, 1e-4]  # [1x2] s^-2
sourcedepthlims = [150, 400]  # [1x2] m
flowratelims = [0, .3]  # m^3/s
VariableParameters = {'umlims': umlims, 'nsquaredlims': nsquaredlims, 'sourcedepthlims': sourcedepthlims,
                      'flowratelims': flowratelims}

# Numeric Parameters:
ssize = 3000  # integer
sfactor = 3  # unitless>1
tolerance = 1e-6  # unitless
miniterations = 2  # integer
maxiterations = 300  # integer
initialdamping = .95  # unitless [0,1)
finaldamping = .25  # unitless ]0,1)
efoldingcells = 3  # unitless
wsterminationfactor = 0  # unitless        THIS IS THE ONLY PARAMETER REMEMBERED IF YOU ARE LOADING AN INPUT FILE
numsamples_1d = 2  # integer
d1displayinterval = 1  # integer
d2displayinterval = 10  # integer
NumericParameters = {'ssize': ssize, 'sfactor': sfactor, 'tolerance': tolerance, 'miniterations': miniterations,
                     'maxiterations': maxiterations, 'initialdamping': initialdamping, 'finaldamping': finaldamping,
                     'efoldingcells': efoldingcells, 'wsterminationfactor': wsterminationfactor,
                     'numsamples_1d': numsamples_1d, 'd1displayinterval': d1displayinterval,
                     'd2displayinterval': d2displayinterval}

"""
Parameter space key:
d1 = source depth
d2 = flow rate
d3 = tratification
d4 = horizontal flow rate
"""

if 0:
    pass
    """ This is probably not worth the hassle of implementing right now. Variables will be saved at the end of
    the script. It doesn't take that long to run and does not need to be run often. """
    # Check whether to load partially completed file:
    # if exist('inputfile', 'var')
    #
    #     # Flag:
    #     continueold = 1
    #
    #     # Load old file:
    #     newwsfactor = wsterminationfactor
    #     load(inputfile)
    #
    #     # Unpack parameter structures:
    #     unpack(ConstantParameters)
    #     unpack(VariableParameters)
    #     unpack(NumericParameters)
    #
    #     # Override ws termination factor:
    #     wsterminationfactor = newwsfactor
    #
    #     # Set starting d1:
    #     startingd1 = d1
else:
    # Flag:
    continueold = 0

    # Set starting d1:
    startingd1 = 0
    
    # Create parameter space grids (this requires some dimensional reshaping for python):
    SourceDepth = np.linspace(sourcedepthlims[0], sourcedepthlims[1], numsamples_1d)[:, None]
    FlowRate = np.linspace(flowratelims[0], flowratelims[1], numsamples_1d)
    Nsquared = np.exp(
        np.transpose(np.linspace(np.log(nsquaredlims[0]), np.log(nsquaredlims[1]), numsamples_1d)[None, :, None],
                     (2, 0, 1)))
    Um = np.exp(np.transpose(np.linspace(np.log(umlims[0]), np.log(umlims[1]), numsamples_1d)[None, :, None, None],
                             (3, 2, 0, 1)))

    # Pre-allocate output matrices:
    Converged = np.ones((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    ReachedSurface = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d), dtype=bool)
    FinalWidth = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    NutrientFlux = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    MinVelocityRatio = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    InitialAngle = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    FinalAngle = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    MeanAngle = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))
    DownstreamDistance = np.zeros((numsamples_1d, numsamples_1d, numsamples_1d, numsamples_1d))

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
Gprime_a_ud = np.zeros((ssize + 1, 1))
U_a_ud = np.zeros((ssize + 1, 1))
NutrientFlux_ud = np.zeros((ssize + 1, 1))
NutrientConcentration_a_ud = np.zeros((ssize + 1, 1))
Ws_ud = np.zeros((ssize + 1, 1))
BubbleRadius_ud = np.zeros((ssize + 1, 1))
Rho_a_ud = np.zeros((ssize + 1, 1))
SinTheta_ud = np.zeros((ssize + 1, 1))
CosTheta_ud = np.zeros((ssize + 1, 1))

# Compute nutrient gradient:
gamma = 1 / (sourcedepthlims[1] - dp)
# ConstantParameters.gamma = gamma

# Run Model:
# Loop through parameter space:
for d1 in range(startingd1, numsamples_1d):
    if inputfile is not None:
        break
    for d2 in range(numsamples_1d):

        # Check if the flow rate is zero:
        if FlowRate[d2] == 0:
            # Assign degenerate values:
            ReachedSurface[d1, d2, :, :] = 0
            NutrientFlux[d1, d2, :, :] = 0
            FinalWidth[d1, d2, :, :] = 0
            MinVelocityRatio[d1, d2, :, :] = 0
            InitialAngle[d1, d2, :, :] = 90
            FinalAngle[d1, d2, :, :] = 90
            MeanAngle[d1, d2, :, :] = 90
            DownstreamDistance[d1, d2, :, :] = 0
            # Skip to next flow rate:
            continue
        
        for d3 in range(numsamples_1d):
            for d4 in range(numsamples_1d):
                print('d1: {}, d2: {}, d3: {}, d4: {}'.format(d1, d2, d3, d4))

                # Compute density gradient:
                rhograd = rho_m * Nsquared[0, 0, d3] / g

                # Compute shear:
                shear = np.sqrt(Nsquared[0, 0, d3] / ri)

                # Compute average density:
                rhobar = rho_m + (.5 * rhograd * (SourceDepth[d1] - dp)) * (SourceDepth[d1] - dp) / SourceDepth[d1]

                # Compute source density:
                rho_s = rho_m + rhograd * (SourceDepth[d1] - dp)

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
                Rho_a_ud[0] = rho_m + rhograd * (SourceDepth[d1] - dp)
                Gprime_a_ud[0] = 0
                U_a_ud[0] = max(0, Um[0, 0, 0, d4] - shear * (SourceDepth[d1] - dp))
                NutrientConcentration_a_ud[0] = gamma * (SourceDepth[d1] - dp)

                # model (water) variables:
                B_ud[0] = 2 * alpha * ds
                W_ud[0] = wguess
                Gprime_ud[0] = 0
                Velocity_ud[0] = np.sqrt(W_ud[0] ** 2 + U_a_ud[0] ** 2)

                # trig terms:
                SinTheta_ud[0] = U_a_ud[0] / Velocity_ud[0]
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
                    U_a_ud[ii + 1] = max(0, Um[0, 0, 0, d4] - shear * max(0, Depth_ud[ii + 1] - dp))
                    Velocity_ud[ii + 1] = Velocity_ud[ii]

                    # Record guesses (Note: must make copies in python!):
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
                        if Depth_ud[ii + 1] - .5 * B_ud[ii + 1] * SinTheta_ud[ii + 1] < 0:
                            hasterminated = 1
                            ReachedSurface[d1, d2, d3, d4] = 1
                            break

                        # Define ambient profiles:
                        Rho_a_ud[ii + 1] = rho_m + rhograd * max(0, Depth_ud[ii + 1] - dp)
                        Gprime_a_ud[ii + 1] = g * (rho_s - Rho_a_ud[ii + 1]) / rho_s
                        U_a_ud[ii + 1] = max(0, Um[0, 0, 0, d4] - shear * max(0, Depth_ud[ii + 1] - dp))
                        NutrientConcentration_a_ud[ii + 1] = gamma * max(0, Depth_ud[ii + 1] - dp)

                        # Compute bubble radius:
                        BubbleRadius_ud[ii + 1] = r0 * ((SourceDepth[d1] + da) / (da + Depth_ud[ii + 1])) ** (1 / 3)

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
                                    inairflux * CosTheta_ud[ii + 1] / (Depth_ud[ii + 1] + da) - Ws_ud[ii + 1] * F_ud[
                                ii + 1] * SinTheta_ud[ii + 1])

                        # Advance primitive variables:
                        W_ud[ii + 1] = outmomentum / outflux
                        Velocity_ud[ii + 1] = np.sqrt(U_a_ud[ii + 1] ** 2 + W_ud[ii + 1] ** 2)
                        Gprime_ud[ii + 1] = outbuoyancy / outflux
                        B_ud[ii + 1] = outflux / (Velocity_ud[ii + 1] * (1 - F_ud[ii + 1]))
                        F_ud[ii + 1] = outairflux / (
                                    B_ud[ii + 1] * (Velocity_ud[ii + 1] + Ws_ud[ii + 1] * CosTheta_ud[ii + 1]))

                        # Check air fraction:
                        if F_ud[ii + 1] < 0 or F_ud[ii + 1] > 1:
                            raise Exception('Impossible air fraction')

                        # Advance nutrient volume flux:
                        NutrientFlux_ud[ii + 1] = NutrientFlux_ud[ii] + ds * entrainment * NutrientConcentration_a_ud[
                            ii + 1]

                        # Check if the plume is moving too slowly:
                        if W_ud[ii + 1] < wsterminationfactor * Ws_ud[ii + 1]:
                            hasterminated = 1
                            ReachedSurface[d1, d2, d3, d4] = 0
                            break

                        # Display current guess:
                        # disp(['b=',num2str(B_ud(ii+1)),', w=',num2str(W_ud(ii+1)),', gprime=',num2str(Gprime_ud(ii+1)),', F=',num2str(F_ud(ii+1))])

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
                            Converged[d1, d2, d3, d4] = 0
                            # Break from loops:
                            done_gridcell = 1
                            hasterminated = 1
                            # Throw error message:
                            # error('Unable to converge')
                        else:
                            # Count iterations:
                            numiterations += 1
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
                    # NutrientFlux(d1,d2,d3,d4)=NutrientFlux_ud(ii)Depth_ud(ii+1)-.5*B_ud(ii+1)*SinTheta_ud(ii+1)
                    NutrientFlux[d1, d2, d3, d4] = NutrientFlux_ud[ii] * max(0, min(1, (
                                nearsurfacedepth - (Depth_ud[ii] - .5 * B_ud[ii] * SinTheta_ud[ii])) / (B_ud[ii] * SinTheta_ud[ii])))
                    FinalWidth[d1, d2, d3, d4] = B_ud[ii]
                    MinVelocityRatio[d1, d2, d3, d4] = min(W_ud[1:ii + 1]/ Ws_ud[1:ii + 1])
                    Theta_ud = np.rad2deg(np.arcsin(SinTheta_ud))
                    InitialAngle[d1, d2, d3, d4] = Theta_ud[1]
                    FinalAngle[d1, d2, d3, d4] = Theta_ud[ii]
                    MeanAngle[d1, d2, d3, d4] = np.mean(Theta_ud[1:ii + 1])
                    DownstreamDistance[d1, d2, d3, d4] = X_ud[ii]

    # Save output:
    params = [ConstantParameters, VariableParameters, NumericParameters, SourceDepth, FlowRate, Nsquared, Um, Converged,
              ReachedSurface, NutrientFlux, FinalWidth, MinVelocityRatio, InitialAngle, FinalAngle, MeanAngle,
              DownstreamDistance, PumpPower, d1, d2, d3, d4]
    with open(outputfile, 'wb') as f:
        pickle.dump(params, f)


# -------------------- PLOTS -----------------------------------------

if inputfile is not None:
    with open(inputfile, 'rb') as f:
        (ConstantParameters, VariableParameters, NumericParameters, SourceDepth, FlowRate, Nsquared, Um, Converged,
         ReachedSurface, NutrientFlux, FinalWidth, MinVelocityRatio, InitialAngle, FinalAngle, MeanAngle,
         DownstreamDistance, PumpPower, d1, d2, d3, d4) = pickle.load(f)

# Figure parameters:
pagesize = [12, 12]  # [1x2] inches
resolution = 300  # dpi
verttextbuffer = .05  # unitless
titlebuffer = .03  # unitless
horztextbuffer = .06  # unitless
horznotextbuffer = .03  # unitless
ticklength = .02  # unitless
tickdir = 'out'  # 'in' or 'out'
depthtick = 50  # m
flowratetick = .05  # m^3/s
powerlims = [0, 1200]  # [1x2] kW
powertick = 200  # kW
problims = [0, 1]  # [1x2] unitless
probtick = 0.25  # unitless
distlims = [0, 200]  # [1x2] m
disttick = 50  # m
blims = [0, 100]  # [1x2] unitless
btick = 25  # unitless
fluxavglims = [0, 3.5]  # [1x2] 10^3 m^3/s
fluxavgtick = .5  # 10^3 m^3/s
fluxvarlims = [0, 3.5]  # [1x2] 10^3 m^3/s
fluxvartick = .5  # 10^3 m^3/s
cmap = 'parula'  # valid colormap string
numcontours = 30  # integer

#
# # Set colormap:
# colormap(cmap)
#
# Plot pump power:
contourf(FlowRate, SourceDepth, PumpPower / 1000, linspace(powerlims[0], powerlims[1], numcontours))
# # Set up stuff:
# caxis(powerlims)
# set(gca, 'ydir', 'reverse')
# ylim(sourcedepthlims)
# set(gca, 'YTick', depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims[1] / depthtick)])
# ylabel('Source Depth (m)')
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, 'XTick', flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims[1] * 1.0001 / flowratetick)])  # KLUDGE
# set(gca, 'XTickLabel', [])
# set(gca, 'TickLength', [ticklength, .025])
# set(gca, 'TickDir', tickdir)
# # Color bar:
# hc1 = colorbar
# set(get(hc1, 'Ylabel'), 'String', 'kW')
# set(hc1, 'YTick', powertick * [ceil(powerlims[0] / powertick): 1:floor(powerlims[1] / powertick)])
# # Title:
# title('a) Pump Power')
#
# # Call second subplot:
# subplot('Position', Boxes
# {1, 2})
# # Plot probability of reaching surface:
# contourf(FlowRate, SourceDepth, mean(mean(ReachedSurface, 4), 3), linspace(problims[0], problims[1], numcontours))
# # Set up stuff:
# caxis(problims)
# set(gca, 'ydir', 'reverse')
# ylim(sourcedepthlims)
# set(gca, 'YTick', depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims[1] / depthtick)])
# set(gca, 'YTickLabel', [])
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, 'XTick', flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims[1] * 1.0001 / flowratetick)])  # KLUDGE
# set(gca, 'XTickLabel', [])
# set(gca, 'TickLength', [ticklength, .025])
# set(gca, 'TickDir', tickdir)
# # Color bar:
# hc2 = colorbar
# # set(get(hc2,'Ylabel'),'String','probability')
# set(hc2, 'YTick', probtick * [ceil(problims[0] / probtick): 1:floor(problims[1] / probtick)])
# # Title:
# title('b) Probability of Reaching Surface')
#
# # Call third subplot:
# subplot('Position', Boxes
# {2, 1})
# # Plot mean downstream distance:
# contourf(FlowRate, SourceDepth, mean(mean(DownstreamDistance, 4), 3), linspace(distlims[0], distlims[1], numcontours))
# # Set up stuff:
# caxis(distlims)
# set(gca, 'ydir', 'reverse')
# ylim(sourcedepthlims)
# set(gca, 'YTick', depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims[1] / depthtick)])
# ylabel('Source Depth (m)')
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, 'XTick', flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims[1] * 1.0001 / flowratetick)])  # KLUDGE
# set(gca, 'XTickLabel', [])
# set(gca, 'TickLength', [ticklength, .025])
# set(gca, 'TickDir', tickdir)
# # Color bar:
# hc3 = colorbar
# set(get(hc3, 'Ylabel'), 'String', 'm')
# set(hc3, 'YTick', disttick * [ceil(distlims[0] / disttick): 1:floor(distlims[1] / disttick)])
# # Title:
# title('c) Downstream Distance')
#
# # Call fourth subplot:
# subplot('Position', Boxes
# {2, 2})
# # Plot final width:
# contourf(FlowRate, SourceDepth, mean(mean(FinalWidth, 4), 3), linspace(blims[0], blims[1], numcontours))
# # Set up stuff:
# caxis(blims)
# set(gca, 'ydir', 'reverse')
# ylim(sourcedepthlims)
# set(gca, 'YTick', depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims[1] / depthtick)])
# set(gca, 'YTickLabel', [])
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, 'XTick', flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims[1] * 1.0001 / flowratetick)])  # KLUDGE
# set(gca, 'XTickLabel', [])
# set(gca, 'TickLength', [ticklength, .025])
# set(gca, 'TickDir', tickdir)
# # Color bar:
# hc4 = colorbar
# set(get(hc4, 'Ylabel'), 'String', 'm')
# set(hc4, 'YTick', btick * [ceil(blims[0] / btick): 1:floor(blims[1] / btick)])
# # Title:
# title('d) Final Plume Width')
#
# # Call third subplot:
# subplot('Position', Boxes
# {3, 1})
# # Plot mean nutrient flux to the surface:
# contourf(FlowRate, SourceDepth, mean(mean(NutrientFlux, 4), 3), linspace(fluxavglims[0], fluxavglims[1], numcontours))
# # Set up stuff:
# caxis(fluxavglims)
# set(gca, 'ydir', 'reverse')
# ylim(sourcedepthlims)
# set(gca, 'YTick', depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims[1] / depthtick)])
# ylabel('Source Depth (m)')
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, 'XTick', flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims[1] * 1.0001 / flowratetick)])  # KLUDGE
# xlabel('Air Flow Rate (m**3/s)')
# set(gca, 'TickLength', [ticklength, .025])
# set(gca, 'TickDir', tickdir)
# # Color bar:
# hc3 = colorbar
# set(get(hc3, 'Ylabel'), 'String', '10**3 m**3/s')
# set(hc3, 'YTick', fluxavgtick * [ceil(fluxavglims[0] / fluxavgtick): 1:floor(fluxavglims[1] / fluxavgtick)])
# # Title:
# title('e) Expected Nutrient Flux')
#
# # Call fourth subplot:
# subplot('Position', Boxes
# {3, 2})
# # Compute variability in nutrient flux:
# FluxVar = zeros(numsamples_1d, numsamples_1d)
# for d1=1:numsamples_1d
# for d2=1:numsamples_1d
# ThisFlux = NutrientFlux(d1, d2,:,:)
# FluxVar(d1, d2) = std(ThisFlux(:))
# end
# end
# # Plot variability in nutrient flux:
# contourf(FlowRate, SourceDepth, FluxVar, linspace(fluxvarlims[0], fluxvarlims[1], numcontours))
# # Set up stuff:
# caxis(fluxvarlims)
# set(gca, 'ydir', 'reverse')
# ylim(sourcedepthlims)
# set(gca, 'YTick', depthtick * [ceil(sourcedepthlims[0] / depthtick): 1:floor(sourcedepthlims[1] / depthtick)])
# set(gca, 'YTickLabel', [])
# xlim(flowratelims. * [0, 1.0001])  # KLUDGE
# set(gca, 'XTick', flowratetick * [ceil(flowratelims[0] / flowratetick): 1:floor(
#     flowratelims[1] * 1.0001 / flowratetick)])  # KLUDGE
# xlabel('Air Flow Rate (m**3/s)')
# set(gca, 'TickLength', [ticklength, .025])
# set(gca, 'TickDir', tickdir)
# # Color bar:
# hc4 = colorbar
# set(get(hc4, 'Ylabel'), 'String', '10**3 m**3/s')
# set(hc4, 'YTick', fluxvartick * [ceil(fluxvarlims[0] / fluxvartick): 1:floor(fluxvarlims[1] / fluxvartick)])
# # Title:
# title('f) Nutrient Flux Variability')
