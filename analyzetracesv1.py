# This code is meant to read in traces from an agilent digitizer in binary
# format. The naming scheme of the files read in is meant to match with those
# of the hydrogen Lamb shift experiment in Eric Hessels' lab at York University
# in Toronto, ON, Canada.
#
# September 13, 2016
# Travis Valdez
# York University
# Toronto, ON, Canada
#
# v2.0:
# Added column for digitizer phase difference (syncing mixer 1 on digitizer 1
# to mixer 2 on digitizer 2). This is defined such that
# phase_(mixer 1, digi 2) + digi_phidiff = phase_(mixer 1, digi 1)
# Added a column for FOSOF phase difference with the second mixer by utilizing
# the digitizer phase difference

from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import sin, cos, pi
import os, sys, time, shutil
from datetime import datetime as dt
import scipy.optimize as scopt
import json
from defaultfolders import *

freqnum_key = 'Number of Frequency Steps'
repnum_key = 'Number of Repeats'
avgnum_key = 'Number of Averages'
digichrange_key = 'Digitizer Channel Range [V]'
digisamplingrate_key = 'Digitizer Sampling Rate [S/s]'
diginumsamples_key = 'Number of Digitizer Samples'
offsetfreqnum_key = 'Number of Offset Frequencies'
numbx_key = 'Number of B_x Steps'
numby_key = 'Number of B_y Steps'
pre910onoff_key = 'Pre-Quench 910 On/Off'
bundle_size_key = 'Number of Traces Between Switching Configurations'
nummassflow_key = 'Number of Mass Flow Rate Steps'
separation_key = "Waveguide Separation [cm]"
efield_key = "Waveguide Electric Field [V/cm]"
calibration_key = "Calibration Name"

rep_columnname = 'Repeat'
avg_columnname = 'Average'
freq_columnname = 'Waveguide Carrier Frequency [MHz]'
offsetfreq_columnname = 'Offset Frequency [Hz]'
abconfig_columnname = 'Configuration'
detectorfname_columnname = 'Detector Trace Filename'
pcomb1fname1_columnname = 'RF Power Combiner I Digi 1 Trace Filename'
pcomb2fname_columnname ='RF Power Combiner R Trace Filename'
pcomb1fname2_columnname = 'RF Power Combiner I Digi 2 Trace Filename'
cgxpressure_columnname = "Charge Exchange Pressure [torr]"
massflowrate_columnname = "Mass Flow Rate [CC]"
wg_A_power_columnname = "Waveguide A Power Reading  [V]"
wg_B_power_columnname = "Waveguide B Power Reading [V]"

new_columns = ["Repeat",
               "Average",
               "Sub-Configuration",
               "Waveguide Carrier Frequency [MHz]",
               "B_x [Gauss]",
               "B_y [Gauss]",
               "Offset Frequency [Hz]",
               "Pre-Quench 910 State",
               "Waveguide A Power Reading  [V]",
               "Waveguide B Power Reading [V]",
               "Time",
               "Detector Trace Filename",
               "RF Power Combiner I Digi 1 Trace Filename",
               "RF Power Combiner R Trace Filename",
               "RF Power Combiner I Digi 2 Trace Filename",
               "Charge Exchange Pressure [torr]",
               "Mass Flow Rate [CC]",
               "fc1a [uA]",
               "fc1b [uA]",
               "fc1c [uA]",
               "fc1d [uA]",
               "fc2i [uA]",
               "fc2ii [uA]",
               "fc2iii [uA]",
               "fc2iv [uA]",
               "fc3 [uA]",
               "fccentre [uA]",
               "Pre-Quench 910 Power Detector Reading [V]",
               "Pre-Quench 910 Attenuator Voltage Reading [V]",
               "Pre-Quench 1088 Power Detector Reading [V]",
               "Pre-Quench 1088 Attenuator Voltage Reading [V]",
               "Pre-Quench 1147 Power Detector Reading [V]",
               "Pre-Quench 1147 Attenuator Voltage Reading [V]",
               "Post-Quench 910 Power Detector Reading [V]",
               "Post-Quench 910 Attenuator Voltage Reading [V]",
               "Post-Quench 1088 Power Detector Reading [V]",
               "Post-Quench 1088 Attenuator Voltage Reading [V]",
               "Post-Quench 1147 Power Detector Reading [V]",
               "Post-Quench 1147 Attenuator Voltage Reading [V]",
               "Trace Bundle",
               "Detector Amplitude [V]",
               "Detector Phase [rad]",
               "Detector DC Offset [V]",
               "Detector SNR (Approx)",
               "Detector Amplitude (2nd Harmonic) [V]",
               "Detector Phase (2nd Harmonic) [rad]",
               "Detector DC Offset (2nd Harmonic) [V]",
               "Detector SNR 2nd Harmonic (Approx)",
               "Mixer 1 Digitizer 1 Amplitude [V]",
               "Mixer 1 Digitizer 1 Phase [rad]",
               "Mixer 1 Digitizer 1 DC Offset [V]",
               "Mixer 1 Digitizer 1 SNR (Approx)",
               "Mixer 2 Amplitude [V]",
               "Mixer 2 Phase [rad]",
               "Mixer 2 DC Offset [V]",
               "Mixer 2 SNR (Approx)",
               "Mixer 1 Digitizer 2 Amplitude [V]",
               "Mixer 1 Digitizer 2 Phase [rad]",
               "Mixer 1 Digitizer 2 DC Offset [V]",
               "Mixer 1 Digitizer 2 SNR (Approx)",
               "Phase Difference (Detector - Mixer 1) [rad]",
               "Phase Difference (Detector - Mixer 1, Corrected) [rad]",
               "Phase Difference (Detector 2nd Harmonic - Mixer 1) [rad]",
               "Phase Difference (Mixer 2 - Mixer 1) [rad]",
               "Digitizer Phase Difference (Digi 2 - Digi 1) [rad]",
               "Phase Difference (Detector - Mixer 2) [rad]",
               "Phase Difference (Detector - Mixer 2, Corrected) [rad]"
               ]

def convert_to_poly(x):
    if type(x) == str:
        return np.poly1d(eval(x))
    else:
        return np.poly1d(x)

fit_df = pd.read_csv(analysis_TV + "WCPopVsPDetV/WGFits.csv")
fit_df.set_index('Frequency [MHz]', inplace = True)
fit_df.fillna(0.0, inplace = True)
fit_df = fit_df.applymap(convert_to_poly)

def power_spectrum(y,sampling_rate):
    Fy = np.fft.fft(y)/np.sqrt(len(y) * sampling_rate)
    f = np.fft.fftfreq(len(y),1./sampling_rate)
    return f,np.abs(Fy)**2

def fit(y, dt, f):
    # Fit data to: y(t) = a cos(omega t) + b sin(omega t) + c
    #                   = A cos(phi) cos(omega t) + A sin(phi) sin(omega t) + c
    #                   = A cos(omega t - phi) + c
    # Fitting routine is simple fourier amplitude extraction
    # Input parameters are:
    # y - digitizer trace data
    # dt - time elapsed between each data point in SECONDS (we usually use 1us)
    # f - frequency of interest in HERTZ

    data_length = len(y)
    omega = 2.0*pi*f
    t = np.linspace(0,(data_length-1)*dt,data_length)

    a = 2.0*np.average(y*cos(omega*t))
    b = 2.0*np.average(y*sin(omega*t))
    c = np.average(y)

    A = np.sqrt(a**2 + b**2)
    phi = (np.arctan2(b,a) + 2.0*pi) % (2.0*pi)
        # arctan2(y,x) = tan^-1(y/x), returns values in the range (-pi,pi)
        # with the above definition, phi gets mapped to the interval (0,2*pi)

    # Returns: amplitude of the waveform A, phase phi, and DC offset c
    return np.float(A), np.float(phi), np.float(c)

def get_file_data(filename):
    # This function looks at the specified file (should be a text file) and
    # retrieves both the header and the data.
    #
    # The lines comprising the header should all begin with a # and be in the
    # format:
    # itemname = itemvalue
    # The header will be stored as a dictionary (known in this program as
    # run_dict).
    #
    # The lines after the header should consist of:
    # 1) a line with the data column names
    # 2) a table of data (mix of text, integers and floating point values)

    run_dict = {}
    num_lines_to_skip = 0
    with open(filename) as thefile:
        # Grab the next line in the file and append it to the run dictionary
        # while the new line contains a '#' at the beginning and a '=' somewhere
        # within it.
        while True:
            nextline = next(thefile)
            if nextline[0] != "#" or nextline.find(' = ') == -1:
                thefile.close()
                break
            attribute = nextline[2:nextline.find("\n")].split(" = ")
            run_dict[attribute[0]] = attribute[1]
            num_lines_to_skip += 1

    data = pd.read_csv(filename, skiprows=num_lines_to_skip)

    if freqnum_key not in run_dict.keys():
        run_dict[freqnum_key] = \
            len(data[freq_columnname].unique())

    return data, run_dict

def get_phase_shift(E_nominal, freq, pd_voltage_A, pd_voltage_B, calibration,
                    separation):
    '''
    Based on the simulated fractional population, the calibration data (V vs
    population), and the power detector voltages for the current trace, this
    function calculates the phase shift to apply due to uneven power, or
    incorrect power. The correct sign will have to be applied to account for
    configuration (0 or pi) and subconfiguration (A or B).
    '''

    freq = round(freq, 1)
    # Which fits to use; fit_df is indexed by frequency in MHz
    fits_row = fit_df.loc[freq]

    # Extract experimental fit
    sim_q_fit = fits_row["Quench Simulation Fit"]
    sim_fosof_fit = fits_row["FOSOF Simulation Fit " + str(separation) + " cm"]

    # Determine the resulting population remaining based on power detector
    # reading for each channel
    exp_pop_A = fits_row[" ".join([calibration, "A"])](pd_voltage_A)
    exp_pop_B = fits_row[" ".join([calibration, "B"])](pd_voltage_B)

    # From determined population and simulation data, find the actual E^2 that
    # was used.
    exp_pow_A = scopt.fsolve(lambda x: exp_pop_A - sim_q_fit(x), [0.])[0]
    exp_pow_B = scopt.fsolve(lambda x: exp_pop_B - sim_q_fit(x), [0.])[0]

    # Justified based on simulation results
    avg_exp_pow = np.average([exp_pow_A, exp_pow_B])

    # Phase difference between target power and actual power. This is just an
    # approximation; it works so long as avg_exp_pow is near E_nominal**2.
    actual_atomic_phase = sim_fosof_fit(avg_exp_pow)
    target_atomic_phase = sim_fosof_fit(E_nominal**2)

    # Making sure we're not getting nonsense
    if (actual_atomic_phase != 0) and (target_atomic_phase != 0):
        return actual_atomic_phase - target_atomic_phase

def read_traces(directory):

    print("Beginning analysis...")

    # Get data from the run_parameters file
    # This run_parameters file contains information about the data run. For
    # example, it contains how many carrier and offset frequencies were used.
    # It also contains all data that was not collected by the digitizer
    # (e.g. power meter voltages collected by the LabJacks, etc.).
    filename = data_LS + directory + '/data.txt'
    outfile_name = filename[filename.rfind("/")+1:filename.rfind(".txt")] + \
                    ".TV"
    outfile_directory = base_folder_TV + directory

    num_finished = 0

    data, run_dict = get_file_data(filename)
    if not (cgxpressure_columnname in data.columns):
        new_columns.remove(cgxpressure_columnname)
        new_columns.remove(massflowrate_columnname)
        new_columns.append(cgxpressure_columnname)
        new_columns.append(massflowrate_columnname)

    # If the folder does not exist, create it
    if not os.path.exists(outfile_directory):
        os.makedirs(outfile_directory)
        df = pd.DataFrame(columns = new_columns)

    # If the folder exists but not the file, create a new data frame
    elif not os.path.exists(outfile_directory+"/"+outfile_name):
        df = pd.DataFrame(columns = new_columns)
    else:
        os.chdir(outfile_directory)
        df = pd.read_csv(outfile_name.replace(".TV","_TV.csv"))
        num_finished = len(df)
        df = df.drop("Unnamed: 0",axis=1)

    with open(outfile_directory + "/rundict_parameters.txt", "w") as f:
        f.write(json.dumps(run_dict))

    print("File data acquired.")

    # Storing parameters from the run dictionary that will be used multiple
    # times throughout this code.
    num_freqs = int(run_dict[freqnum_key])
    num_reps = int(run_dict[repnum_key])
    num_avg = int(run_dict[avgnum_key])
    if offsetfreqnum_key in run_dict.keys():
        num_of = int(run_dict[offsetfreqnum_key])
    else:
        num_of = len(data[offsetfreq_columnname].unique())

    # Digitizer info. Needed to convert binary --> ADC --> float
    d1_c1_range = float(run_dict[digichrange_key])
    d1_c2_range = float(run_dict[digichrange_key])
    d2_c1_range = float(run_dict[digichrange_key])
    d2_c2_range = float(run_dict[digichrange_key])
    sampling_rate = float(run_dict[digisamplingrate_key])
    num_Bx = float(run_dict[numbx_key])
    num_By = float(run_dict[numby_key])

    if calibration_key in run_dict.keys():
        calibration = run_dict[calibration_key]
    else:
        date_of_dataset = directory[:13]
        date_of_dataset = dt.strptime(date_of_dataset, "%y%m%d-%H%M%S")

        # For legacy files
        if date_of_dataset.date() < dt(2018,4,11).date():
            calibration = "Calibration 180327"
        elif date_of_dataset.date() < dt(2018,4,30).date():
            calibration = "Calibration 180422"
        elif date_of_dataset.date() < dt(2018,5,12).date():
            calibration = "Calibration 180428"
        else:
            calibration = "Calibration 180512"

    if nummassflow_key in run_dict.keys():
        number_of_massflow_settings = float(run_dict[nummassflow_key])
    else:
        number_of_massflow_settings = 1
    bundle_size = int(run_dict[bundle_size_key])
    if run_dict[pre910onoff_key] == True or run_dict[pre910onoff_key] == 'True':
        pre910onoff = 2
    else:
        pre910onoff = 1

    separation = int(run_dict[separation_key])
    electric_field = int(run_dict[efield_key])

    # I use the number of lines to determine when the analysis has finished
    # Note that this is the number of lines that will exist in the data file
    # once the acquisition is complete. This value allows this program to
    # run concurrently with the acquisition.
    num_lines = num_freqs * num_reps*num_avg * \
                    num_of * 2 * \
                    pre910onoff*num_Bx * \
                    num_By * number_of_massflow_settings

    # Recording the binary directory
    bin_dir = 'Z:/' + directory

    # Since not every data set will include all columns, I just set everything
    # here to a default null value so as not to get errors later on.
    filename_d1_c1 = filename_d1_c2 = filename_d2_c1 = \
    filename_d2_c2 = 'N/A'

    # Used for error checking later (if nan --->)
    fosof_A = fosof_phi = fosof_c = \
    fosof_A2 = fosof_phi2 = fosof_c2 = \
    mixer_A = mixer_phi = mixer_c = \
    detector_mixer_phidiff = detector_mixer_phidiff2 = \
    mixer_2_A = mixer_2_c = mixer_2_phi = \
    mixer_1_A = mixer_1_c = mixer_1_phi = mixer_phidiff = np.nan

    num_finished_last = 0
    waited = 0

    # While there is still data that has been or will be acquired but has not
    # been analyzed...
    while num_finished < num_lines:
        # Start "where we left off". By this, I mean if the program is running
        # at the same time as data collection, it will stop when it has finished
        # analyzing all acquired data and pause for a time to allow data to be
        # collected. Instead of starting from the beginning, this program will
        # pick up where it left off.
        data, run_dict = get_file_data(filename)
        line_n = num_finished
        while line_n < len(data):
            os.chdir(bin_dir)

            # The repeat, average, carrier frequency and offset frequency
            # will be consistent throughout all data relevant to the current
            # line.
            repeat = data[rep_columnname][line_n]
            average = data[avg_columnname][line_n]
            tracebundle = (average - 1) // bundle_size + 1
            carrier_freq = data[freq_columnname][line_n]
            wg_A_power = float(data[wg_A_power_columnname][line_n])
            wg_B_power = float(data[wg_B_power_columnname][line_n])

            # Some of the older data sets do not have the offset frequency in
            # the columns. If it is not there, it will be in the header.
            offsetFrequency = data[offsetfreq_columnname][line_n]

            int_of = int(offsetFrequency) # For use with "fit()"

            subconfig = data[abconfig_columnname][line_n]

            # Acquire data from digitizer traces for FOSOF detector and
            # mixer.
            filename_d1_c1 = data[detectorfname_columnname][line_n]
            filename_d1_c1 += '.digi.npy'
            filename_d1_c2 = data[pcomb1fname1_columnname][line_n]
            filename_d1_c2 += '.digi.npy'

            print(filename_d1_c1)
            V1 = [None]
            # Sometimes the network takes awhile to update with the new file.
            # Error checking to wait for the files to show up rather than crash.
            while V1[0] == None:
                try:
                    V1 = np.load(filename_d1_c1)*d1_c1_range/32767.0
                    V2 = np.load(filename_d1_c2)*d1_c2_range/32767.0
                except:
                    V1 = [None]
                    print("Could not find one of the files. Sleeping then " \
                          "trying again.")
                    time.sleep(5)

            freqs,S = power_spectrum(V1,sampling_rate)

            min_60 = int_of // 60

            # The noise indices are those within the 60 Hz band that contains
            # the offset frequency, but not the index of the offset frequency
            # itself. The idea here is to average the noise at the other
            # components in the 60 Hz band to get an idea of the noise level.
            snr_noise_index = np.intersect1d(np.where(freqs - min_60 * 60 > 0),
                                             np.where(freqs - (min_60 + 1) * 60\
                                                      <= -(freqs[1] - freqs[0]))
                                            )

            # The signal index is self explanatory
            signal_index = np.where(abs(freqs - float(int_of)) \
                                    == min(abs(freqs-float(int_of)))
                                   )

            # Here I delete the signal index from the noise index
            snr_noise_index = np.delete(snr_noise_index,
                                        np.where(freqs[snr_noise_index] \
                                                 == freqs[signal_index]))
            print("Signal frequency in FFT: " + str(freqs[signal_index]))

            # Calculating the signal to noise
            det_snr = np.sqrt(S[signal_index]) / \
                            np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Atoms): " + str(det_snr))

            # Same thing for second harmonic. Maybe I'll make this a function
            # at some point
            min_602nd = int_of*2 // 60

            snr_noise_index_2nd = np.intersect1d(
                                    np.where(freqs - min_602nd * 60 > 0),
                                    np.where(freqs - (min_602nd+1) * 60 \
                                             <= -(freqs[1]-freqs[0]))
                                                )
            signal_index_2nd = np.where(abs(freqs - float(int_of*2)) \
                                        == min(abs(freqs-float(int_of*2))))
            snr_noise_index_2nd = np.delete(snr_noise_index_2nd, \
                                            np.where(freqs[snr_noise_index_2nd]\
                                                     == freqs[signal_index_2nd])
                                           )
            det_snr_2nd = np.sqrt(S[signal_index_2nd]) / \
                            np.average(np.sqrt(S[snr_noise_index_2nd]))

            # Same things for other channels. I should almost certainly make
            # this a function
            freqs,S = power_spectrum(V2,sampling_rate)

            m1d1_snr = np.sqrt(S[signal_index]) / \
                            np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Mixer 1.1): " + str(m1d1_snr))

            fosof_A, fosof_phi, fosof_c = fit(V1, 1.0/sampling_rate, int_of)
            fosof_A2, fosof_phi2, fosof_c2 = fit(V1, 1.0/sampling_rate,
                                                 2*int_of)
            mixer_A, mixer_phi, mixer_c = fit(V2, 1.0/sampling_rate, int_of)

            print("FOSOF Current: " + str(fosof_c))

            # Determine amount by which the phase should be shifted due to
            # incorrect power.
            phase_shift = get_phase_shift(electric_field, carrier_freq,
                                          wg_A_power, wg_B_power, calibration,
                                          separation)
            if "pi config" in directory:
                sign_0_pi = -1
            else:
                sign_0_pi = 1

            if subconfig == "A":
                sign_A_B = 1
            else:
                sign_A_B = -1

            phase_shift = phase_shift * sign_0_pi * sign_A_B

            # Taking differences and shifting back to the [0, 2*pi) range
            detector_mixer_phidiff = (fosof_phi - mixer_phi + 2.0 * np.pi) \
                                                    % (2.0 * np.pi)
            detector_mixer_phidiff2 = (fosof_phi2 - mixer_phi + 2.0 * np.pi) \
                                                % (2.0 * np.pi)

            detector_mixer_phidiff_c = detector_mixer_phidiff + phase_shift
            detector_mixer_phidiff_c += 2. * np.pi
            detector_mixer_phidiff_c = detector_mixer_phidiff_c % (2. * np.pi)

            # Digitizer 2
            filename_d2_c1 = data[pcomb2fname_columnname][line_n]
            filename_d2_c1 += '.digi.npy'
            filename_d2_c2 = data[pcomb1fname2_columnname][line_n]
            filename_d2_c2 += '.digi.npy'

            V3 = [None]
            while V3[0] == None:
                try:
                    V3 = np.load(filename_d2_c1)*d2_c1_range/32767.0
                    V4 = np.load(filename_d2_c2)*d2_c2_range/32767.0
                except:
                    V3 = [None]
                    print("Could not find one of the files. Waiting and " \
                          "trying again.")
                    time.sleep(5)

            freqs, S = power_spectrum(V3,sampling_rate)

            m2_snr = np.sqrt(S[signal_index]) / \
                            np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Mixer 2): " + str(m2_snr))

            freqs,S = power_spectrum(V4,sampling_rate)

            m1d2_snr = np.sqrt(S[signal_index]) / \
                            np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Mixer 1.2): " + str(m1d2_snr))

            mixer_2_A, mixer_2_phi, mixer_2_c = fit(V3, 1.0/sampling_rate,
                                                    int_of)
            mixer_1_A, mixer_1_phi, mixer_1_c = fit(V4, 1.0/sampling_rate,
                                                    int_of)


            mixer_phidiff = (mixer_2_phi - mixer_1_phi + 2.0 * np.pi) \
                                            % (2.0 * np.pi)
            digi_phidiff = (mixer_1_phi - mixer_phi) + 2.0 * np.pi
            digi_phidiff = digi_phidiff % (2. * np.pi) - np.pi

            detector_mixer2_phidiff = (fosof_phi - (mixer_2_phi - \
                                                             digi_phidiff) + \
                                                             2.0 * np.pi) \
                                                             % (2.0 * np.pi)

            detector_mixer2_phidiff_c = detector_mixer2_phidiff + phase_shift
            detector_mixer2_phidiff_c += 2. * np.pi
            detector_mixer2_phidiff_c = detector_mixer2_phidiff_c % (2. * np.pi)

            analyzed_values = [tracebundle, fosof_A, fosof_phi, fosof_c,
                               det_snr, fosof_A2, fosof_phi2, fosof_c2,
                               det_snr_2nd, mixer_A, mixer_phi, mixer_c,
                               m1d1_snr, mixer_2_A, mixer_2_phi, mixer_2_c,
                               m2_snr, mixer_1_A, mixer_1_phi, mixer_1_c,
                               m1d2_snr, detector_mixer_phidiff,
                               detector_mixer_phidiff_c,
                               detector_mixer_phidiff2, mixer_phidiff,
                               digi_phidiff, detector_mixer2_phidiff,
                               detector_mixer2_phidiff_c]

            if not (cgxpressure_columnname in data.columns):
                analyzed_values.append(1.6e-6)
                analyzed_values.append(0.25)

            newData = data.ix[line_n].values
            newData = np.append(newData, analyzed_values)
            newData = pd.Series(newData, index = new_columns)

            df = df.append(newData, ignore_index = True)

            # Save every thousand files (just in case of a power outage or \
            # something similar)
            line_n += 1
            if line_n % 1000 == 0:
                os.chdir(outfile_directory)
                df.to_csv(outfile_name.replace(".TV","_TV.csv"),
                          header = df.columns)
                print("Saved data, finished "+str(line_n)+" files.")
                print("Finished "+str(float(line_n)/float(num_lines)))

        num_finished = len(df)
        if num_finished == num_finished_last:
            waited += 1
        else:
            waited = 0
            num_finished_last = num_finished

        # Assuming new data will occur within two minutes, otherwise assume the
        # dataset was aborted for some reason.
        if waited == 24:
            print("Assuming this dataset was cancelled. Finished at " + \
                  str(100.*float(line_n)/float(num_lines)) + "%")
            num_finished = num_lines
            os.chdir(outfile_directory)
            f = open('aborted.txt', 'w')
            f.write('Ended after ' + \
                    str(100.*float(line_n)/float(num_lines)) + '% completed.')
            f.close()

        os.chdir(outfile_directory)
        df.to_csv(outfile_name.replace(".TV","_TV.csv"),
                  header = df.columns)
        print("Saved and waiting 5 seconds for more files...")
        print("Finished "+str(float(line_n)/float(num_lines)))
        time.sleep(5)

    os.chdir(outfile_directory)
    df.to_csv(outfile_name.replace(".txt",".csv"), header = df.columns)
    print("Done.")
    print("Saved to: " + outfile_name)

def main():

    # All arguments except filenames must contain a '-' at the beginning.
    # All filenames must have the same arguments (for now)
    args = []
    directoryNames = []
    list_of_files = open("./fosof_file_list.txt").read().split('\n')[:-1]
    print(list_of_files)
    for item in list_of_files:
        read_traces(item)

        print("Analysis finished for:")
        print(item)

if __name__ == '__main__':
    main()
