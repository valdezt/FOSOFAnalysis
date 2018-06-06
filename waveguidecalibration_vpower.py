# -*- coding: utf-8 -*-
"""
This code is used to calibrate the RF waveguides used in the FOSOF experiment.
Previously, we fit a polynomial to the population vs electric field (or
population vs sqrt(power)) data. In this code, we've switched to using
population vs power and fitting only to the data with which we're concerned
(to around 25 V/cm electric field according to the simulations).

To estimate the error in our power calibration, we will use a few measures:
1) Calibrate the power with both 3rd and 4th order polynomials
2) Calibrate the data using simulations from on- and off-axis (with our best
guess of off-axis distance of 1.8 mm)
3) Calibrate the power by both subtracting AND not subtracting the 4% offset we
see at the pi-pulse location
4) Flag the calibration as poor if neither the 3rd nor 4th order fit to the
experimental data fits

Created on Mon May 15 12:01:36 2017

Author: Travis Valdez
"""

from __future__ import division
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib import rcParams
import scipy.optimize as opt
import scipy.optimize as scopt
from scipy.interpolate import UnivariateSpline as uvs
import scipy.stats as stats
import numpy as np
from numpy import ma
import os, sys
import fosoffunctionsv2 as ff

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

# Polynomial forcing the y-intercept to 0 W, 100% remaining population
# For use with scipy.optimize.curve_fit
def power_v_pop_fit(x, *params):
    mysum = 0.
    for p in range(1, len(params)+1):
        mysum += params[p-1]*x**p

    return 1. + mysum

# Polynomial forcing the y-intercept to 0 W, 0 V
# For use with scipy.optimize.curve_fit
def power_v_voltage_fit(x, *params):
    mysum = 0.
    for p in range(1, len(params)+1):
        mysum += params[p-1]*x**p

    return mysum

# Functions for conversion between W and dBm
def dBm_to_watts(pdBm):
    return 10**(pdBm/10.0)/1000.0

def watts_to_dBm(pW):
    return 10.0 * np.log10(1000.0*pW)

#%%

sim_frequency_column = 'Frequency [MHz]'
sim_efield_column = 'Electric Field [V/cm]'
sim_axis_column = 'Off-Axis [mm]'

# Import the simulated quench curves
sim_file = pd.read_csv("C:\\Users\\Travis\\Google Drive\\Travis Code\\" \
                       "FOSOF Data Analysis\\FOSOFDataAnalysis\\" \
                       "SOF_QuenchCurve.csv")
sim_file = sim_file.set_index([sim_axis_column,
                               sim_frequency_column,
                               sim_efield_column])
sim_on_axis = sim_file.loc[0., :, :].copy()
sim_off_axis = sim_file.loc[1.8, :, :].copy()

# Import the data to a pandas DataFrame object by searching for the timestamp
# specified.
data_location = 'Y:/data/'
data_location_list = np.array(os.listdir(data_location))
dataset_folder_timestamp = '180516-075539'
dataset_folder = [f for f in data_location_list \
                  if dataset_folder_timestamp in f]
data_location += dataset_folder[0] + '/data.txt'

wg_to_analyze = 'B'
exp_channel_column = 'Generator Channel'
exp_frequency_column = 'Waveguide Frequency Setting [MHz]'
exp_power_column = 'Waveguide Power Setting [dBm]'
exp_onoff_column = 'Digitizer DC On/Off Ratio'
exp_pdet_voltage = 'Waveguide ' + wg_to_analyze + \
                   ' Power Reading (Generator On) [V]'
exp_pdet_off = 'Waveguide ' + wg_to_analyze + \
               ' Power Reading (Generator Off) [V]'

# All comments in our files are always denoted with a hash mark
data = pd.read_csv(data_location, comment = "#")

# Select only the data for the specified waveguide
data_wg = data.loc[data[exp_channel_column] == wg_to_analyze]
data_wg = data_wg.groupby([exp_frequency_column,
                           exp_power_column])
data_wg = data_wg.agg([np.mean, lambda x: np.std(x, ddof = 1)/np.sqrt(len(x))])
data_wg = data_wg.rename(columns={'mean' : 'Average', '<lambda>' : 'Error'})

#%%

# Export fits of population vs Schottky power detector voltage
chisquare = []
chisquare4 = []

cal_curve = pd.DataFrame({}, columns = ["3rd Order", "4th Order"])

for f in data_wg.index.get_level_values(0).unique():
    toplot = data_wg.loc[(f, slice(None))].reset_index().copy()
    toplot = toplot.loc[toplot[exp_onoff_column, 'Average'] >= 0.24]

    # Subtract the "waveguide off" offset from the power detector reading
    toplot[exp_pdet_voltage,
                'Average'] = toplot[exp_pdet_voltage, 'Average'].values - \
                             toplot[exp_pdet_off, 'Average'].values

    # Make sure the fit is not weighted too much by just a few points.
    toplot[exp_onoff_column, 'Error'] = np.array([max(e, 0.003) for e in \
                                                  toplot[exp_onoff_column,
                                                            'Error'].values])

    toplot.sort_values((exp_pdet_voltage, 'Average'), inplace = True)
    cs = uvs(toplot[exp_pdet_voltage, 'Average'].values,
             toplot[exp_onoff_column, 'Average'].values,
             w = 1/toplot[exp_onoff_column, 'Error'].values**2,
             k = 3, s = len(toplot) / 2)
    poly = np.poly1d(ff.polyfit(toplot[exp_pdet_voltage, 'Average'].values,
                                toplot[exp_onoff_column, 'Average'].values, 3,
                                w = 1/toplot[exp_onoff_column, 'Error'].values**2)[0])
    poly4 = np.poly1d(ff.polyfit(toplot[exp_pdet_voltage, 'Average'].values,
                                toplot[exp_onoff_column, 'Average'].values, 4,
                                w = 1/toplot[exp_onoff_column, 'Error'].values**2)[0])

    # These curves are fitting power detector voltage vs population. Using these
    # curves, I can calibrate what power we actually used for any trace.
    norder = 3
    mycurve = scopt.curve_fit(power_v_pop_fit,
                              xdata = toplot[exp_pdet_voltage, 'Average'].values.flatten(),
                              ydata = toplot[exp_onoff_column, 'Average'].values,
                              p0 = [1. for i in range(norder)])
    mycurve = np.poly1d(np.append(np.flip(mycurve[0], 0), 1.))

    mycurve4 = scopt.curve_fit(power_v_pop_fit,
                               xdata = toplot[exp_pdet_voltage, 'Average'].values.flatten(),
                               ydata = toplot[exp_onoff_column, 'Average'].values,
                               p0 = [1. for i in range(norder+1)])
    mycurve4 = np.poly1d(np.append(np.flip(mycurve4[0], 0), 1.))
    cal_curve = cal_curve.append(pd.Series({'3rd Order' : list(mycurve),
                                            '4th Order' : list(mycurve)},
                                 name = round(f,1)))

    chisquare.append(ff.chi2(toplot[exp_onoff_column, 'Average'].values, mycurve(toplot[exp_pdet_voltage, 'Average'].values),
                             err = toplot[exp_onoff_column, 'Error'].values, dof_minus = norder)[0])
    chisquare4.append(ff.chi2(toplot[exp_onoff_column, 'Average'].values, mycurve4(toplot[exp_pdet_voltage, 'Average'].values),
                      err = toplot[exp_onoff_column, 'Error'].values, dof_minus = norder)[0])

cal_curve.index.names = ['Frequency [MHz]']
cal_curve['Waveguide'] = wg_to_analyze
cal_curve.reset_index(inplace = True)
cal_curve.set_index(['Frequency [MHz]', 'Waveguide'], inplace = True)
save_path = "C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/WCPopVsPDetV/"
if os.path.exists(save_path + "WGFits.csv"):
    old = pd.read_csv(save_path + "WGFits.csv").set_index(['Frequency [MHz]', 'Waveguide'])
    cal_curve = cal_curve.combine_first(old)

cal_curve.to_csv(save_path + "WGFits.csv")

#%%
# Fit only up to 25 V/cm
max_e_field_to_fit = 25
norder = 4 # For now
norder_sim = 4
norder_exp = 4
sim_file.sort_index(inplace = True) # Required to slice the index

cols = ['Frequency [MHz]', 'Electric Field [V/cm]', 'Fit',
        'RF generator power [dBm]', 'Power Detector Voltage [V]',
        'Power Detector Voltage Error [V]', 'Chi-Squared', 'Pr(Chi-Squared)']
calibration = pd.DataFrame([], columns = cols)


for frequency in data_wg.index.get_level_values(exp_frequency_column).unique():

    try:
        sim_to_fit = sim_file.loc[(1.8, round(frequency, 1), \
                                   slice(0,max_e_field_to_fit))]
    except:
        sim_to_fit = sim_file.loc[(1.8, 926.1, \
                                   slice(0,max_e_field_to_fit))]
    sim_efields = sim_to_fit.index.get_level_values(sim_efield_column).values
    sim_pops = sim_to_fit.values

    # Fit the simulation to the optimal order as well as one order less. This
    # gives us an estimate in the error due to our fit.
    sim_fit_order = scopt.curve_fit(power_v_pop_fit,
                                    xdata = (sim_efields**2).flatten(),
                                    ydata = sim_pops.flatten(),
                                    p0 = [1. for i in range(norder)])
    sim_func_order = np.poly1d(np.append(np.flip(sim_fit_order[0], 0), 1.))

    sim_fit_orderm1 = scopt.curve_fit(power_v_pop_fit,
                                      xdata = (sim_efields**2).flatten(),
                                      ydata = sim_pops.flatten(),
                                      p0 = [1. for i in range(norder - 1)])
    sim_func_orderm1 = np.poly1d(np.append(np.flip(sim_fit_orderm1[0], 0), 1.))

    # Fit the data with the on-axis simulations
    try:
        sim_to_fit = sim_file.loc[(0.0, round(frequency, 1), \
                                   slice(0,max_e_field_to_fit))]
    except:
        sim_to_fit = sim_file.loc[(0.0, 926.1, \
                                   slice(0,max_e_field_to_fit))]
    sim_efields = sim_to_fit.index.get_level_values(sim_efield_column).values
    sim_pops = sim_to_fit.values

    sim_fit_onaxis = scopt.curve_fit(power_v_pop_fit,
                                     xdata = (sim_efields**2).flatten(),
                                     ydata = sim_pops.flatten(),
                                     p0 = [1. for i in range(norder)])
    sim_func_onaxis = np.poly1d(np.append(np.flip(sim_fit_onaxis[0], 0), 1.))

    # Only fit the experimental data in the range of interest
    min_pop_to_fit = min(sim_pops)

    exp_to_fit = data_wg.loc[(round(frequency, 1), slice(None))].copy()
    exp_to_fit = exp_to_fit.loc[exp_to_fit[exp_onoff_column]['Average'].values \
                                >= min_pop_to_fit]
    exp_powers = exp_to_fit.index.get_level_values(exp_power_column).values
    exp_powers = dBm_to_watts(exp_powers + 40)
    exp_pops = exp_to_fit[exp_onoff_column]['Average'].values
    exp_errs = exp_to_fit[exp_onoff_column]['Error'].values
    exp_pd_wgon = exp_to_fit[exp_pdet_voltage]['Average'].values
    exp_pd_wgon -= exp_to_fit[exp_pdet_off]['Average'].values
    exp_pd_err = np.sqrt(exp_to_fit[exp_pdet_off]['Error'].values**2 +
                         exp_to_fit[exp_pdet_voltage]['Error'].values**2)

    # Fit cubic and quartic splines to the front panel power vs power detector
    # voltage. This will be used to determine the power detector voltage we are
    # looking for.
    cs_3 = uvs(exp_powers, exp_pd_wgon, w = 1/exp_pd_err**2, k = 3)
    cs_4 = uvs(exp_powers, exp_pd_wgon, w = 1/exp_pd_err**2, k = 4)

    # Fit the data without shifting and stretching by the unquenchable offset
    exp_fit_noshift = scopt.curve_fit(power_v_pop_fit,
                                      xdata = exp_powers.flatten(),
                                      ydata = exp_pops.flatten(),
                                      sigma = exp_errs.flatten(),
                                      p0 = [1. for i in range(norder)])
    exp_func_noshift = np.poly1d(np.append(np.flip(exp_fit_noshift[0], 0), 1.))

    # Compute the chi^2 statistic to check the quality of fit
    chi2noshift, prbnoshift = ff.chi2(exp_pops, exp_func_noshift(exp_powers),
                                      err = exp_errs,
                                      dof_minus = norder)

    # Shift and stretch the data
    exp_pops -= 0.04
    exp_pops = exp_pops / 0.96
    exp_errs = exp_errs / 0.96

    # Fit the data to the optimal order as well as one order less. This gives
    # us an estimate in the error due to our fit.
    exp_fit_order = scopt.curve_fit(power_v_pop_fit,
                                    xdata = exp_powers.flatten(),
                                    ydata = exp_pops.flatten(),
                                    sigma = exp_errs.flatten(),
                                    p0 = [1. for i in range(norder)])
    exp_func_order = np.poly1d(np.append(np.flip(exp_fit_order[0], 0), 1.))

    exp_fit_orderm1 = scopt.curve_fit(power_v_pop_fit,
                                      xdata = exp_powers.flatten(),
                                      ydata = exp_pops.flatten(),
                                      sigma = exp_errs.flatten(),
                                      p0 = [1. for i in range(norder - 1)])
    exp_func_orderm1 = np.poly1d(np.append(np.flip(exp_fit_orderm1[0], 0), 1.))

    # Compute the chi^2 statistic to check the quality of fit
    chi2, prb = ff.chi2(exp_pops, exp_func_order(exp_powers),
                        err = exp_errs,
                        dof_minus = norder)

    # Compute the chi^2 statistic to check the quality of fit
    chi2m1, prbm1 = ff.chi2(exp_pops, exp_func_orderm1(exp_powers),
                            err = exp_errs,
                            dof_minus = norder - 1)

    # Setting the threshold probability at 0.1%
    poor_fit = False
    if prb <= 0.001:
        poor_fit = True
        print(frequency, chi2, prb)

    # Determine the front panel power required to acheive each of the following
    # electric fields in the waveguides
    efields_to_calibrate = np.linspace(5, 25, 21)
    gen_pwr = opt.fsolve(lambda x: sim_func_order(efields_to_calibrate**2) - \
                                   exp_func_order(x),
                         [1.0 for j in range(len(efields_to_calibrate))])

    # Again for different data variations
    gen_pwr_expm1 = opt.fsolve(lambda x:
                               sim_func_order(efields_to_calibrate**2) - \
                               exp_func_orderm1(x),
                               [1.0 for j in range(len(efields_to_calibrate))])

    gen_pwr_simm1 = opt.fsolve(lambda x:
                               sim_func_orderm1(efields_to_calibrate**2) - \
                               exp_func_order(x),
                               [1.0 for j in range(len(efields_to_calibrate))])

    # Again for different data variations
    gen_pwr_onaxis = opt.fsolve(lambda x:
                                sim_func_onaxis(efields_to_calibrate**2) - \
                                exp_func_order(x),
                                [1.0 for j in \
                                 range(len(efields_to_calibrate))])

    # Again for different data variations
    gen_pwr_noshift = opt.fsolve(lambda x:
                                 sim_func_order(efields_to_calibrate**2) - \
                                 exp_func_noshift(x),
                                 [1.0 for j in \
                                  range(len(efields_to_calibrate))])

    # Determine the power (in dBm) required on the front panel of the
    # generator to obtain electric fields in the waveguides on between 5 and 24
    # V/cm
    for efield_index in range(len(efields_to_calibrate)):
        efield = efields_to_calibrate[efield_index]
        pwr = watts_to_dBm(gen_pwr[efield_index]) - 40
        pwr_expm1 = watts_to_dBm(gen_pwr_expm1[efield_index]) - 40
        pwr_simm1 = watts_to_dBm(gen_pwr_simm1[efield_index]) - 40
        pwr_onaxis = watts_to_dBm(gen_pwr_onaxis[efield_index]) - 40
        pwr_noshift = watts_to_dBm(gen_pwr_noshift[efield_index]) - 40
        to_append = pd.DataFrame([[frequency, efield, 'Best Guess', pwr,
                                   cs_4(gen_pwr[efield_index]),
                                   abs(cs_4(gen_pwr[efield_index]) - \
                                       cs_3(gen_pwr[efield_index])),
                                   chi2, prb],
                                  [frequency, efield,
                                   'Experimental Order Lower', pwr_expm1,
                                   cs_4(gen_pwr_expm1[efield_index]),
                                   abs(cs_4(gen_pwr_expm1[efield_index]) - \
                                       cs_3(gen_pwr_expm1[efield_index])),
                                   chi2m1, prbm1],
                                  [frequency, efield,
                                   'Simulation Order Lower', pwr_simm1,
                                   cs_4(gen_pwr_simm1[efield_index]),
                                   abs(cs_4(gen_pwr_simm1[efield_index]) - \
                                       cs_3(gen_pwr_simm1[efield_index])),
                                   chi2, prb],
                                  [frequency, efield,
                                   'On-Axis Simulation', pwr_onaxis,
                                   cs_4(gen_pwr_onaxis[efield_index]),
                                   abs(cs_4(gen_pwr_onaxis[efield_index]) - \
                                       cs_3(gen_pwr_onaxis[efield_index])),
                                   chi2, prb],
                                  [frequency, efield,
                                   'No Shift/Stretch', pwr_noshift,
                                   cs_4(gen_pwr_noshift[efield_index]),
                                   abs(cs_4(gen_pwr_noshift[efield_index]) - \
                                       cs_3(gen_pwr_noshift[efield_index])),
                                   chi2noshift, prbnoshift]],
                                 columns = cols)
        calibration = calibration.append(to_append, ignore_index = True)

#%%

# Save the data to a file somewhere
to_save = calibration.loc[calibration['Fit'] == 'Best Guess'].copy()

nominal_pwrs = dBm_to_watts(to_save['RF generator power [dBm]'].values)
print(len(nominal_pwrs), len(to_save))
pct_error = np.zeros(len(nominal_pwrs))
for calib_type in calibration['Fit'].unique():
    cal_thistype = calibration.loc[calibration['Fit'].values == calib_type]

    pwrs = cal_thistype['RF generator power [dBm]'].values
    pwrs = dBm_to_watts(pwrs)
    pct_error += ((pwrs - nominal_pwrs) / nominal_pwrs)**2

# pct_error = np.sqrt(pct_error)
pyerr = watts_to_dBm((1 + np.sqrt(pct_error)) * \
                     dBm_to_watts(to_save['RF generator power [dBm]'].values)) \
        - to_save['RF generator power [dBm]'].values

nyerr = to_save['RF generator power [dBm]'].values - \
        watts_to_dBm((1 - np.sqrt(pct_error)) * \
                     dBm_to_watts(to_save['RF generator power [dBm]'] \
                                              .values))

err = np.array([np.average([nyerr[i], pyerr[i]]) for i in range(len(nyerr))])

to_save['Generator Power Error [dBm]'] = err
to_save['Waveguide'] = wg_to_analyze

base_folder = 'C:/Users/Travis/Google Drive/Travis Code/Waveguide Calibration/'
this_folder = '4cm_0512_894-926'
if not os.path.exists(base_folder + this_folder):
    os.mkdir(base_folder + this_folder)

# Save errors and everything in one file
to_save.to_csv(base_folder + this_folder + '/full.csv', index = False)

for E in to_save['Electric Field [V/cm]'].unique():
    datafile = "/Waveguide_" + wg_to_analyze + " E=" + str(int(E)) + ".txt"
    datafile = base_folder + this_folder + datafile
    to_save_E = to_save.loc[to_save['Electric Field [V/cm]'].values == E].copy()
    to_save_E.set_index('Frequency [MHz]', inplace = True)
    to_save_E = to_save_E['RF generator power [dBm]']
    if os.path.exists(datafile):
        old = pd.read_csv(datafile, sep = '\t').set_index('Frequency [MHz]')

        # to_save values overwrite those from old
        to_save_E = pd.DataFrame(to_save_E).combine_first(old)

    to_save_E.to_csv(datafile, sep = '\t', header = True)

#%%

# Select which type of data to plot
efield_to_plot = 8
types_to_plot = calibration['Fit'].unique()
chi2thresh = 0.001 # chi-squared probability threshold for a "good" fit

# Plot the generator settings for different types of calculations/calibrations
plt.figure(figsize = [10, 10])
plt.title('Generator Front Panel Power [dBm]')
plt.ylabel('Front Panel Power [dBm]')
plt.xlabel('Frequency [MHz]')

# Plot each one in a different colour
for calib_type in types_to_plot:
    cal_thistype = calibration.loc[(calibration['Fit'].values == calib_type) *
                                   (calibration['Electric Field [V/cm]'].values\
                                    == efield_to_plot)]
    plt.plot(cal_thistype['Frequency [MHz]'].values,
             cal_thistype['RF generator power [dBm]'].values, '.',
             label = calib_type)
plt.legend()
plt.show()

# The nominal power is our best guess at the front panel settings
nominal_pwrs = calibration.loc[(calibration['Fit'].values == 'Best Guess') *
                               (calibration['Electric Field [V/cm]'].values \
                                == efield_to_plot),
                               'RF generator power [dBm]']
nominal_pwrs = dBm_to_watts(nominal_pwrs)
pct_error = 0.

plt.figure(figsize = [10, 10])
plt.title("$\\sqrt{\\sum (\\% Error)^2}$ As a Function of Frequency")
plt.ylabel("Quadrature Sum of % Error")
plt.xlabel('Frequency [MHz]')

# Go through and sum the % errors in quadrature
for calib_type in types_to_plot:
    cal_thistype = calibration.loc[(calibration['Fit'].values == calib_type) *
                                   (calibration['Electric Field [V/cm]'].values\
                                    == efield_to_plot)]

    pwrs = cal_thistype['RF generator power [dBm]'].values
    pwrs = dBm_to_watts(pwrs)
    if calib_type == 'No Shift/Stretch':
        pct_error += (((pwrs - nominal_pwrs) / nominal_pwrs)*.5)**2
        plt.plot(cal_thistype['Frequency [MHz]'].values,
                 abs((pwrs - nominal_pwrs) / nominal_pwrs) * 50, '.',
                 markersize = 15,
                 label = calib_type)
    else:
        pct_error += (((pwrs - nominal_pwrs) / nominal_pwrs))**2
        plt.plot(cal_thistype['Frequency [MHz]'].values,
                 abs((pwrs - nominal_pwrs) / nominal_pwrs) * 100, '.',
                 markersize = 15,
                 label = calib_type)

# Plot the total % error, highlighting the bad points
plt.plot(calibration.loc[(calibration['Pr(Chi-Squared)'].values < chi2thresh) \
                         * (calibration['Fit'].values == 'Best Guess'),
                         'Frequency [MHz]'].unique(),
         np.sqrt(pct_error[calibration['Pr(Chi-Squared)'] < chi2thresh]) * 100,
         'r.', label = '$\chi^2 < 0.1 \\%$')
plt.plot(calibration.loc[(calibration['Pr(Chi-Squared)'].values >= chi2thresh) \
                         * (calibration['Fit'].values == 'Best Guess'),
                         'Frequency [MHz]'].unique(),
         np.sqrt(pct_error[calibration['Pr(Chi-Squared)'] >= chi2thresh]) * 100,
         'b.', label = '$\chi^2 \geq 0.1 \\%$')
plt.legend(loc = 5)
plt.show()


#%%

# Plot the best guess for the power with proper error bars
cal_thistype = calibration.loc[(calibration['Fit'].values == 'Best Guess') *
                               (calibration['Electric Field [V/cm]'].values\
                                == efield_to_plot)]

# Separate data with "good" and "bad" chi-squared probabilities
good_data = cal_thistype.loc[cal_thistype['Pr(Chi-Squared)'] >= chi2thresh]
bad_data = cal_thistype.loc[cal_thistype['Pr(Chi-Squared)'] < chi2thresh]

# Calculate the error bars:
# +err = to_dbm((1 + %err) * p_watts) - p_dbm
# -err = p_dbm - to_dbm((1 - %err) * p_watts)
pyerr = watts_to_dBm((1 + np.sqrt(pct_error)) * \
                     dBm_to_watts(cal_thistype['RF generator power [dBm]'] \
                                              .values)) \
        - cal_thistype['RF generator power [dBm]'].values

nyerr = cal_thistype['RF generator power [dBm]'].values - \
        watts_to_dBm((1 - np.sqrt(pct_error)) * \
                     dBm_to_watts(cal_thistype['RF generator power [dBm]'] \
                                              .values))

# print(cal_thistype[['Frequency [MHz]','RF generator power [dBm]']])

# Plot the nominal settings, highlighting the bad values in red
plt.figure(figsize = [10, 10])
plt.title('Nominal RF Generator Front Panel Power')
plt.ylabel('Power Setting [dBm]')
plt.xlabel('Frequency [MHz]')
plt.errorbar(bad_data['Frequency [MHz]'].values,
             bad_data['RF generator power [dBm]'].values,
             yerr = [nyerr[cal_thistype['Pr(Chi-Squared)'] < chi2thresh],
                     pyerr[cal_thistype['Pr(Chi-Squared)'] < chi2thresh]],
             fmt = 'r.',
             label = '$\chi^2 \geq 0.1 \\%$')
plt.errorbar(good_data['Frequency [MHz]'].values,
             good_data['RF generator power [dBm]'].values,
             yerr = [nyerr[cal_thistype['Pr(Chi-Squared)'] >= chi2thresh],
                     pyerr[cal_thistype['Pr(Chi-Squared)'] >= chi2thresh]],
             fmt = 'b.',
             label = '$\chi^2 \geq 0.1 \\%$')
plt.legend(loc = 1)
plt.show()

# Plot the values used by the generator (with error bars)
plt.figure(figsize = [10, 10])
plt.title('Generator Power Used (Assuming 0.1 dBm Granularity)')
plt.ylabel('Power Setting [dBm]')
plt.xlabel('Frequency [MHz]')
plt.errorbar(bad_data['Frequency [MHz]'].values,
             np.round_(bad_data['RF generator power [dBm]'].values, 1),
             yerr = [nyerr[cal_thistype['Pr(Chi-Squared)'] < chi2thresh],
                     pyerr[cal_thistype['Pr(Chi-Squared)'] < chi2thresh]],
             fmt = 'r.',
             label = '$\chi^2 \geq 0.1 \\%$')
plt.errorbar(good_data['Frequency [MHz]'].values,
             np.round_(good_data['RF generator power [dBm]'].values, 1),
             yerr = [nyerr[cal_thistype['Pr(Chi-Squared)'] >= chi2thresh],
                     pyerr[cal_thistype['Pr(Chi-Squared)'] >= chi2thresh]],
             fmt = 'b.',
             label = '$\chi^2 \geq 0.1 \\%$')
plt.legend(loc = 1)
plt.show()

#%%

power_calib_dbm_column = 'RF power [dBm]'
power_calib_det_volt = 'Power detector signal [V]'
power_calib_det_volt_std = 'STD the mean of power detector signal [V]'

power_detector_calibration_data = pd.read_csv("Y:/code/instrument control/" + \
                                              "170822-130101 - RF power detector calibration/" + \
                                              "KRYTAR 109B power calibration.CSV")
power_detector_calibration_data.sort_values(power_calib_det_volt, inplace = True)
pdet_curve = uvs(power_detector_calibration_data[power_calib_det_volt],
                 power_detector_calibration_data[power_calib_dbm_column],
                 w=1./power_detector_calibration_data[power_calib_det_volt_std]**2,
                 k = 3)

efield = 24
waveguide = 'B'
oldfile = 'C:/Users/Travis/Google Drive/Travis Code/Waveguide Calibration/xlarge4cm/'
oldfile = oldfile + 'Waveguide_' + waveguide + ' E=' + str(efield) +'.txt'

# Comparing old data to new data
new_data = pd.read_csv(datafile)
new_data = new_data.set_index(['Waveguide', 'Electric Field [V/cm]'])
new_data.sort_index(inplace = True)
new_data = new_data.loc[waveguide, efield]
new_data = new_data.set_index('Frequency [MHz]')
new_data.sort_index(inplace = True)

old_data = pd.read_csv(oldfile, sep = '\t')
old_data = old_data.set_index('Frequency [MHz]')
old_data.sort_index(inplace = True)

plt.figure(figsize=[15, 10])
plt.errorbar(new_data.index, new_data['RF generator power [dBm]'],
             yerr = new_data['Generator Power Error [dBm]'],
             fmt = 'r.',
             label = 'New')
plt.plot(old_data.index, old_data['RF generator power [dBm]'], 'b.',
         label = 'Old')
plt.legend()
plt.show()

plt.figure(figsize=[15, 10])
plt.title('Power Expected on Power Sensor')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Power Detected by Power Sensor [dBm]')
plt.errorbar(new_data.index, new_data['Power Detector Voltage [V]'],
             # yerr = pdet_curve(new_data['Power Detector Voltage [V]'] + \
             #                   new_data['Power Detector Voltage Error [V]']) - \
             #        pdet_curve(new_data['Power Detector Voltage [V]']),
             fmt = 'r.')
plt.show()

#%%

waveguide = 'A'

# Visualizing electric field vs nominal power detector voltage
new_data = pd.read_csv(datafile)
new_data = new_data.set_index(['Waveguide'])
new_data.sort_index(inplace = True)
new_data = new_data.loc[waveguide]
new_data = new_data.set_index('Electric Field [V/cm]')
new_data.sort_index(inplace = True)
print(min(new_data['Frequency [MHz]']))

plt.figure(figsize=[15, 10])
plt.errorbar(new_data.loc[new_data['Frequency [MHz]'] == 894.].index,
             new_data.loc[new_data['Frequency [MHz]'] == 894.,'Power Detector Voltage [V]'],
             yerr = new_data.loc[new_data['Frequency [MHz]'] == 894.,'Power Detector Voltage Error [V]'],
             fmt = 'r.')
plt.errorbar(new_data.loc[new_data['Frequency [MHz]'] == 910.].index,
             new_data.loc[new_data['Frequency [MHz]'] == 910.,'Power Detector Voltage [V]'],
             yerr = new_data.loc[new_data['Frequency [MHz]'] == 910.,'Power Detector Voltage Error [V]'],
             fmt = 'g.')
plt.errorbar(new_data.loc[new_data['Frequency [MHz]'] == 926.].index,
             new_data.loc[new_data['Frequency [MHz]'] == 926.,'Power Detector Voltage [V]'],
             yerr = new_data.loc[new_data['Frequency [MHz]'] == 926.,'Power Detector Voltage Error [V]'],
             fmt = 'b.')
plt.show()
