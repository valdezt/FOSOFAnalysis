# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:01:36 2017

@author: travis
"""

from __future__ import division
import pandas as pd
import matplotlib.mlab as mlab
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os, sys
import scipy.optimize as scopt
import fosoffunctionsv2 as ff

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

data_directory = "Y:\\data\\"
dataset_directory = data_directory + "180519-115023 - Waveguide Calibration - 0 config, PD ON 120 V, 893.8-898 MHz\\"
data = pd.read_csv(dataset_directory + "data.txt", comment = "#")

# print(data.columns)

#%%

print(len(data))
data = data.loc[abs(data['Digitizer DC (Generator Off) [V]']) <= 16.0]
data = data.loc[abs(data['Digitizer STD (Generator Off) [V]']) <= 1.0]

data = data.loc[abs(data['Digitizer DC (Generator On) [V]']) <= 16.0]
data = data.loc[abs(data['Digitizer STD (Generator On) [V]']) <= 1.0]
print(len(data))
print(max(data['Repeat']))

#%%

# The chi-squared probability density function
def chi_squared_pdf(chisq, dof):
    if chisq > 0:
        return chisq**(dof/2-1)*np.exp(-chisq/2)/(2**(dof/2)*sp.special.gamma(dof/2))
    else:
        return 0

# Calculates the chi squared value of the function f compared to the data
# (anXArray, aYArray +/- errorsInYArray). The value functionDOF is the number
# of coefficients that were determined by the fit and hence the degrees of freedom
# for the fit that were removed by determining the best fit function.
def find_chi_squared_value(x_array, y_array, y_error, f, function_dof):
    return (1.0/float(len(x_array)-function_dof))*np.sum(((y_array - f(x_array))/y_error)**2)

# Function to calculate both chi squared and the probability of obtaining that chi squared
# This just minimizes function calls later in the code
def find_chi_squared_and_probability(x_array, y_array, y_error, f, function_dof):
    chisq = find_chi_squared_value(x_array, y_array,  y_error, f, function_dof)
    return chisq, chi_squared_probability(chisq, len(x_array) - function_dof)[0]

# Calculates the probability of obtaining the given chi squared with the number of
# degrees of freedom by numerically integrating the probability density function from
# chiSq to infinity
def chi_squared_probability(chisq, dof):
    return sp.integrate.quad(lambda chi_squared: chi_squared_pdf(chi_squared, dof), dof*chisq, np.inf)

#%%

def get_power_dbm(p, v):
    f = lambda x: abs(p(x)-v)
    soln = scopt.minimize_scalar(f,bounds=(-50.0,20.0),method='bounded')

    return soln['x']

vget_power_dbm = np.vectorize(get_power_dbm)
power_calib_dbm_column = 'RF power [dBm]'
power_calib_det_volt = 'Power detector signal [V]'
power_calib_det_volt_std = 'STD the mean of power detector signal [V]'

power_detector_calibration_data = pd.read_csv("Y:/code/instrument control/" + \
                                              "170822-130101 - RF power detector calibration/" + \
                                              "KRYTAR 109B power calibration.CSV")

poly, cov = np.polyfit(power_detector_calibration_data[power_calib_dbm_column],
                       power_detector_calibration_data[power_calib_det_volt], 7, \
                       w=1./power_detector_calibration_data[power_calib_det_volt_std]**2,
                       cov=True)
p = np.poly1d(poly)
errs = np.sqrt(np.abs(np.diag(cov)))

plt.figure(figsize = [15, 15])
plt.errorbar(power_detector_calibration_data[power_calib_dbm_column], \
             power_detector_calibration_data[power_calib_det_volt],
             xerr=power_detector_calibration_data[power_calib_det_volt_std],fmt='r.', \
             label="m = "+str(round(p[1],4)) + "0+/-" + str(round(errs[0],4)))
plt.plot(power_detector_calibration_data[power_calib_dbm_column],
         p(power_detector_calibration_data[power_calib_dbm_column]))
plt.xlabel("RF Power [dBm]")
plt.ylabel("Power Detector Voltage [V]")
plt.title("RF Power Detector Calibration")
plt.ylim(-0.373,-0.37)
plt.xlim(4.3,4.4)
plt.show()

print(get_power_dbm(p,-0.5))

#%%

def dBm_to_watts(pdBm):
    return 10**(pdBm/10.0)/1000.0

def watts_to_dBm(pW):
    return 10.0 * np.log10(1000.0*pW)

def stdm(x):
    return np.std(x) / np.sqrt(len(x) - 1)

# print(max(data['Waveguide Power Setting [dBm]'])+44)
# print(get_power_dbm(p, min(data['Waveguide A Power Reading  (Generator On) [V]']))+40)

power_dbm_column = 'Waveguide Power Setting [dBm]'
frequency_column = 'Waveguide Frequency Setting [MHz]'
waveguide_to_analyze = 'A'
power_reading_column = 'Waveguide ' + waveguide_to_analyze + ' Power Reading (Generator On) [V]'
data = data.groupby("Generator Channel").get_group(waveguide_to_analyze)
on_off_column = 'Digitizer DC On/Off Ratio'

power_sqrtw_column = 'Waveguide Power Setting [sqrtW]'
detected_power_dbm_column = 'Detected Power (RF Generator ON) [dBm]'
detected_power_sqrtw_column = 'Detected Power (RF Generator ON) [sqrtW]'

# Convert power detector voltages into dBm
data[on_off_column] = data[on_off_column].values / max(data[on_off_column].values)
data[power_sqrtw_column] = np.sqrt(dBm_to_watts(data[power_dbm_column]))
data[detected_power_dbm_column] = np.array([get_power_dbm(p,data[power_reading_column].values[i])+30
                                            for i in range(len(data.index))])
data[detected_power_sqrtw_column] = np.sqrt(dBm_to_watts(data[detected_power_dbm_column]))

if 910.0 in data[frequency_column].values:
    print("Yes it's there.")
    on_resonance = data.groupby(frequency_column).get_group(910.0)
    on_resonance = on_resonance.groupby(power_dbm_column).agg([np.mean,stdm])

    plt.title("Measured Power vs Set Power")
    plt.xlabel("Set Power [dBm]")
    plt.ylabel("Measured Power [dBm]")
    plt.errorbar(on_resonance.index, \
                on_resonance[detected_power_dbm_column]['mean'].values-30, \
                fmt='b.', \
                yerr = on_resonance[detected_power_dbm_column]['stdm'].values)
    plt.show()

#%%

# Plot fractional population vs measured power (in dBm) for a given carrier frequency
frequencies = data[frequency_column].unique()
print(np.sort(frequencies))
frequency_separated_data = data.groupby(frequency_column)
for frequency in np.sort(frequencies):
    a_frequency = frequency_separated_data.get_group(frequency)
    a_frequency_by_power = a_frequency.reset_index().groupby(power_dbm_column).agg([np.mean,stdm])
    set_power = a_frequency[power_dbm_column].unique()

#%%

# Fit the simulated quench file to a polynomial of order n
sim_file = pd.read_csv("C:\\Users\\Travis\\Google Drive\\Travis Code\\FOSOF Data Analysis\\FOSOFDataAnalysis\\SOF_QuenchCurve.csv")
sim_file = sim_file.set_index('freqs')
n = 5
sim_poly = np.polyfit(sim_file.columns.astype('float'),sim_file.loc[910.0], n)
sim_p = np.poly1d(sim_poly)

plt.title("Fractional Population\nvs.\nElectric Field Magnitude\n(Simulated)")
plt.xlabel("Electric Field Magnitude [V/cm]")
plt.ylabel("Fractional Population")
efs = np.linspace(0,60,100)
plt.plot(sim_file.columns.astype('float'),sim_file.loc[910.0],'r.')
plt.plot(efs,sim_p(efs),'g-')
plt.plot(sim_file.columns.astype('float'),sim_file.loc[912.0],'b.')
plt.show()

plt.title("Fractional Population\nvs.\nElectric Field Magnitude Squared\n(Simulated)")
plt.xlabel("Electric Field Magnitude [$V^2/cm^2$]")
plt.ylabel("Fractional Population")
plt.plot(sim_file.columns.astype('float')**2,sim_file.loc[910.0],'r.')
plt.show()

#%%

def efield_to_sqrtw(p, x):
    a = p[0]
    b = p[1]
    c = p[2]
    return a/(x+b)**2 + c

def efield_to_sqrtw_err(p, x, y):
    return efield_to_sqrtw(p, x) - y

def weighted_average(x, cols, e_cols = [None]):
    d = {}
    for col, e_col in zip(cols, e_cols):
        vals = x[col]

        if e_col != None:
            error = x[e_col]
            if len(vals) > 1:
                avg_info = ff.polyfit(range(len(vals.values)), vals.values, 0,
                                      w = 1./error.values**2)
                avg = avg_info[0][0]
                err = avg_info[3][0]
                if avg_info[1] < 1.0:
                    err = err/np.sqrt(avg_info[1])
                d[col + " Average"] = avg
                d[col + " Error"] = err
            else:
                d[col + " Average"] = vals.values[0]
                d[col + " Error"] = error.values[0]
        else:
            if len(vals) > 1:
                avg_info = ff.polyfit(range(len(vals.values)), vals.values, 0)
                avg = avg_info[0][0]
                err = avg_info[3][0]
                if avg_info[1] < 1.0:
                    err = err/np.sqrt(avg_info[1])
                d[col + " Average"] = avg
                d[col + " Error"] = err
            else:
                d[col + " Average"] = vals.values[0]
                d[col + " Error"] = np.nan

    return pd.Series(d)

function_table = pd.DataFrame()
calib_p = [None for j in range(len(frequencies))]
calib_p_inv = []
conv_function = [None for j in range(len(frequencies))]
k = 0

for frequency in np.sort(frequencies):
    # Fit the simulated quench file to a polynomial of order n
    print(str(round(frequency,1)) +" MHz")

    sim_frequency = frequency

    # Obtain the proper simulation file and fit the simulation data to an 8th order polynomial
    n = 5
    sim_poly = np.polyfit(sim_file.columns.astype('float'),
                          sim_file.loc[sim_frequency], n)
    sim_p = np.poly1d(sim_poly)

    # Get the data for the current carrier frequency from the data file
    a_frequency = frequency_separated_data.get_group(frequency)
    a_frequency[on_off_column] = a_frequency['Digitizer DC (Generator On) [V]']/a_frequency['Digitizer DC (Generator Off) [V]']
    a_frequency[on_off_column] = a_frequency[on_off_column].values
    a_frequency[on_off_column + " Error"] = \
        np.sqrt((a_frequency['Digitizer STD (Generator On) [V]'].values/a_frequency['Digitizer DC (Generator Off) [V]'].values)**2 + \
                (a_frequency['Digitizer STD (Generator Off) [V]'].values * a_frequency['Digitizer DC (Generator On) [V]'].values/a_frequency['Digitizer DC (Generator Off) [V]'].values**2)**2)

    a_frequency = a_frequency.groupby(power_sqrtw_column)
    a_frequency = a_frequency.apply(weighted_average,
                                    cols = [on_off_column, detected_power_sqrtw_column],
                                    e_cols = [on_off_column + " Error", None])
    a_frequency[on_off_column + " Average"] = a_frequency[on_off_column + " Average"].values / max(a_frequency[on_off_column + " Average"].values)
    nprime = 5
    # Fit the experimental data for on/off ratio (fractional population) vs. sqrt(power) [sqrt(W)]
    measured_poly = ff.polyfit(a_frequency[detected_power_sqrtw_column + " Average"].values,
                               a_frequency[on_off_column + " Average"].values, nprime,
                               w = 1./a_frequency[on_off_column + " Error"].values**2)
    measured_p = np.poly1d(measured_poly[0])
    pows = np.linspace(min(a_frequency[detected_power_sqrtw_column + " Average"]), \
                       max(a_frequency[detected_power_sqrtw_column + " Average"]),100)

    plt.figure(figsize=[15,15])
    # Plot the fit against the data
    plt.title("Measured Fractional Population vs Sqrt(Set RF Power)\nfor "+str(frequency)+" MHz")
    plt.ylabel("DC ON/OFF Ratio (Fractional Population)")
    plt.xlabel("$\sqrt{\\mathrm{Detected Power}}\\ \\left[ \\sqrt{W} \\right]$")
    plt.plot(pows, measured_p(pows))
    plt.errorbar(a_frequency[detected_power_sqrtw_column + " Average"], a_frequency[on_off_column + " Average"].values, \
                 yerr = a_frequency[on_off_column + " Error"].values, fmt = 'r.')
    plt.savefig('./WGCalFigs/PopVPower_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '.png')
    plt.clf()
    plt.close()

    data_shifted = a_frequency[on_off_column + " Average"].values - 0.045 * (1. + (-0.1/3.5) * a_frequency[detected_power_sqrtw_column + " Average"].values)
    data_shifted = data_shifted*max(a_frequency[on_off_column + " Average"].values)/max(data_shifted)
    print(max(data_shifted))

    # Fit the shifted data for on/off ratio (fractional population) vs. sqrt(power) [sqrt(W)]
    poly_shifted = np.polyfit(a_frequency[detected_power_sqrtw_column + " Average"].values,
                              data_shifted, nprime+1, \
                              w = 1./a_frequency[on_off_column + " Error"].values**2)
    p_shifted = np.poly1d(poly_shifted)

    plt.figure(figsize=[15,15])
    # Plot the fit against the data
    plt.title("Measured and Shifted/Normalized\nFractional Population vs Sqrt(Set RF Power)\nfor "+str(frequency)+" MHz")
    plt.ylabel("DC ON/OFF Ratio (Fractional Population)")
    plt.xlabel("$\sqrt{\\mathrm{Power}}\\ \\left[ \\sqrt{W} \\right]$")
    plt.errorbar(a_frequency[detected_power_sqrtw_column + " Average"],
                 data_shifted, fmt = 'r.', label = "Shifted Data",
                 yerr = a_frequency[on_off_column + " Error"].values)
    plt.errorbar(a_frequency[detected_power_sqrtw_column + " Average"],
                 a_frequency[on_off_column + " Average"].values,
                 fmt = 'b.', label = "Acquired Data",
                 yerr = a_frequency[on_off_column + " Error"].values)
    plt.plot(pows, p_shifted(pows))
    plt.legend()
    plt.savefig('./WGCalFigs/PopVPower_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '_SHIFTED.png')
    plt.clf()
    plt.close()

    plt.figure(figsize=[15,15])
    # Plot the fit against the data
    plt.title("Measured and Shifted/Normalized\nFractional Population vs Sqrt(Set RF Power)\nfor "+str(frequency)+" MHz")
    plt.ylabel("DC ON/OFF Ratio (Fractional Population)")
    plt.xlabel("$\sqrt{\\mathrm{Power}}\\ \\left[ \\sqrt{W} \\right]$")
    plt.errorbar(a_frequency[detected_power_sqrtw_column + " Average"],
                 data_shifted, fmt = 'r.', label = "Shifted Data",
                 yerr = a_frequency[on_off_column + " Error"].values)
    plt.plot(sim_file.columns.astype('float')/13.5,sim_file.loc[sim_frequency],'b.')
    # plt.plot(pows, p_shifted(pows))
    plt.legend()
    plt.savefig('./WGCalFigs/THEPLOTTOCHECK_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '_SHIFTED.png')
    plt.clf()
    plt.close()

    # Convert the sqrt(W) to volts per cm in the following way:
    # 1) Find a point on the measured data on/off ratio
    # 2) Match its value to a fractional population value from the simulation
    # 3) Find the electric field that yields the given fractional population
    volts_per_cm = opt.fsolve(lambda x: sim_p(x) - measured_p(a_frequency[detected_power_sqrtw_column + " Average"]), \
                              [10.0 for j in range(len(a_frequency.index))])

    a_frequency['Electric Field [V/cm]'] = volts_per_cm

    volts_per_cm_shifted = opt.fsolve(lambda x: sim_p(x) - p_shifted(a_frequency[detected_power_sqrtw_column + " Average"]), \
                                      [10.0 for j in range(len(a_frequency.index))])

    a_frequency['Electric Field (Shifted) [V/cm]'] = volts_per_cm_shifted

    # Points to disregard: any data after the amplifier appears to be saturated
    # will cause problems with the fit. If this fit doesn't work the first time,
    # look for saturation, remove some of the saturated points and try again.
    points_to_disregard = 10

    plt.figure(figsize=[15,15])
    plt.title("Measured Data @ Calibrated Electric Field vs.\nSimulation Data")
    plt.xlabel("Electric Field [V/cm]")
    plt.ylabel("Fractional Population")
    plt.plot(sim_file.columns.astype('float'),sim_file.loc[sim_frequency],'r.')
    plt.errorbar(a_frequency['Electric Field [V/cm]'].values, a_frequency[on_off_column + " Average"].values,
                 yerr = a_frequency[on_off_column + " Error"].values, fmt = 'b.')
    plt.savefig('./WGCalFigs/MeasvSim_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '.png')
    plt.clf()
    plt.close()

    plt.figure(figsize=[15,15])
    plt.title("Measured Data @ Calibrated Electric Field vs.\nSimulation Data")
    plt.xlabel("Electric Field [V/cm]")
    plt.ylabel("Fractional Population")
    plt.plot(sim_file.columns.astype('float'),sim_file.loc[sim_frequency],'r.')
    plt.errorbar(a_frequency['Electric Field (Shifted) [V/cm]'].values, data_shifted,
                 yerr = a_frequency[on_off_column + " Error"].values, fmt = 'g.')
    plt.errorbar(a_frequency['Electric Field [V/cm]'].values, a_frequency[on_off_column + " Average"].values,
                 yerr = a_frequency[on_off_column + " Error"].values, fmt = 'b.')
    plt.savefig('./WGCalFigs/MeasvSim_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '_SHIFTED.png')
    plt.clf()
    plt.close()

    if len(a_frequency['Electric Field [V/cm]'].values[:-max(1,points_to_disregard)]) > 6:
        calib_poly = np.polyfit(a_frequency['Electric Field [V/cm]'].values[:-max(1,points_to_disregard)],
                                a_frequency.index[:-max(1,points_to_disregard)], 3,
                                w = 1./a_frequency[on_off_column + " Error"].values[:-max(1,points_to_disregard)])
        calib_p = np.poly1d(calib_poly)
        power_range = np.linspace(min(a_frequency['Electric Field [V/cm]'].values[:-max(1,points_to_disregard)]), \
                                  max(a_frequency['Electric Field [V/cm]'].values[:-max(1,points_to_disregard)]), 1000)

        plt.figure(figsize=[15,15])
        plt.title("Electric Field vs Sqrt(Set Power)\nfor " + str(frequency) + "MHz")
        plt.xlabel("Electric Field [V/cm]")
        plt.ylabel("$\sqrt{\\mathrm{Power}}\\ \\left[ \\sqrt{W} \\right]$")
        plt.plot(power_range, calib_p(power_range), label = "Fit")
        plt.plot(a_frequency['Electric Field [V/cm]'].values[:-max(1,points_to_disregard)], a_frequency.index[:-max(1,points_to_disregard)], 'r.')
        plt.legend()
        plt.savefig('./WGCalFigs/Cal_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '.png')
        plt.clf()
        plt.close()

        conv_function = lambda x: watts_to_dBm((calib_p(x)**2.0))

        new_series = pd.Series([frequency, False, conv_function(5.0), conv_function(6.0), conv_function(7.0), conv_function(8.0), conv_function(9.0), \
                                conv_function(10.0), conv_function(11.0), conv_function(12.0), conv_function(13.0), conv_function(14.0), \
                                conv_function(15.0), conv_function(16.0), conv_function(17.0), conv_function(18.0), conv_function(19.0), \
                                conv_function(20.0), conv_function(21.0), conv_function(22.0), conv_function(23.0), conv_function(24.0), \
                                conv_function(25.0), conv_function(26.0), conv_function(27.0), conv_function(28.0), conv_function(29.0), \
                                conv_function(30.0)])
        function_table = function_table.append(new_series, ignore_index = True)

        calib_poly = np.polyfit(a_frequency['Electric Field (Shifted) [V/cm]'].values[:-max(1,points_to_disregard)], a_frequency.index[:-max(1,points_to_disregard)], 5)
        calib_p = np.poly1d(calib_poly)
        calib_poly_inv = np.polyfit(a_frequency.index[:-max(1,points_to_disregard)], a_frequency['Electric Field (Shifted) [V/cm]'].values[:-max(1,points_to_disregard)], 5)
        calib_p_inv.append(np.poly1d(calib_poly_inv))
        power_range = np.linspace(min(a_frequency['Electric Field (Shifted) [V/cm]'].values[:-max(1,points_to_disregard)]), \
                                  max(a_frequency['Electric Field (Shifted) [V/cm]'].values[:-max(1,points_to_disregard)]), 1000)

        plt.figure(figsize=[15,15])
        plt.title("Electric Field (Shifted and Unshifted) vs Sqrt(Set Power)\nfor " + str(frequency) + "MHz")
        plt.xlabel("Electric Field [V/cm]")
        plt.ylabel("$\sqrt{\\mathrm{Power}}\\ \\left[ \\sqrt{W} \\right]$")
        plt.plot(power_range, calib_p(power_range), label = "Fit")
        plt.plot(a_frequency['Electric Field (Shifted) [V/cm]'].values[:-max(1,points_to_disregard)], a_frequency.index[:-max(1,points_to_disregard)], 'r.', label = "Shifted Data")
        plt.legend()
        plt.savefig('./WGCalFigs/Cal_'+str(round(frequency, 2)) + '_' + waveguide_to_analyze + '_SHIFTED.png')
        plt.clf()
        plt.close()

        conv_function = lambda x: watts_to_dBm((calib_p(x)**2.0))

        new_series = pd.Series([frequency, True, conv_function(5.0), conv_function(6.0), conv_function(7.0), conv_function(8.0), conv_function(9.0), \
                                conv_function(10.0), conv_function(11.0), conv_function(12.0), conv_function(13.0), conv_function(14.0), \
                                conv_function(15.0), conv_function(16.0), conv_function(17.0), conv_function(18.0), conv_function(19.0), \
                                conv_function(20.0), conv_function(21.0), conv_function(22.0), conv_function(23.0), conv_function(24.0), \
                                conv_function(25.0), conv_function(26.0), conv_function(27.0), conv_function(28.0), conv_function(29.0), \
                                conv_function(30.0)])
        function_table = function_table.append(new_series, ignore_index = True)

function_table.columns = ["Carrier Frequency [MHz]", "Shifted?", "5.0 V/cm", "6.0 V/cm", "7.0 V/cm", "8.0 V/cm", "9.0 V/cm", \
                          "10.0 V/cm", "11.0 V/cm", "12.0 V/cm", "13.0 V/cm", "14.0 V/cm", "15.0 V/cm", "16.0 V/cm", "17.0 V/cm", \
                          "18.0 V/cm", "19.0 V/cm", "20.0 V/cm", "21.0 V/cm", "22.0 V/cm", "23.0 V/cm", "24.0 V/cm", "25.0 V/cm", \
                          "26.0 V/cm", "27.0 V/cm", "28.0 V/cm", "29.0 V/cm", "30.0 V/cm"]

function_table = function_table.set_index('Shifted?')

#%%

print(function_table.columns)
function_table = function_table.loc[True]
function_table = function_table.reset_index().set_index('Carrier Frequency [MHz]') #.loc[False].drop('index',axis=1)
print(function_table.index)
print(function_table.values)

#%%

plt.plot(function_table.loc[function_table['Shifted?'] == 0].index, function_table.loc[function_table['Shifted?'] == 0]['8.0 V/cm'].values,'r.')
plt.plot(function_table.loc[function_table['Shifted?'] == 1].index, function_table.loc[function_table['Shifted?'] == 1]['8.0 V/cm'].values,'b.')
plt.show()
#%%

output_directory = "C:\\Users\\travis\\Google Drive\\Travis Code\\Waveguide Calibration\\xlarge4cm\\"

for col in function_table.columns:
    this_efield = function_table[col].copy()
    filename = "Waveguide_" + waveguide_to_analyze + " E="+col[:col.find(".")]

    if not os.path.exists(output_directory+filename+".txt"):
        this_efield.to_csv(output_directory+filename+".txt",sep="\t", header=["RF generator power [dBm]"], \
                                            index_label="Frequency [MHz]")
    else:
        old_data = pd.read_csv(output_directory+filename+".txt",sep="\t")
        old_data = old_data.set_index('Frequency [MHz]')
        for f in this_efield.index.unique():
            old_data.at[f, "RF generator power [dBm]"] = this_efield[f]
        old_data.to_csv(output_directory+filename+".txt",sep="\t", header=["RF generator power [dBm]"], \
                        index_label="Frequency [MHz]")
