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
# phase_(mixer 1, digi 2) + digitizer_phase_difference = phase_(mixer 1, digi 1)
# Added a column for FOSOF phase difference with the second mixer by utilizing
# the digitizer phase difference

from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import sin, cos, pi
import os, sys, time, shutil
import json

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

# To make this program as general as possible, I've included all the columns
# necessary for every type of data we've taken so far. This includes:
# Regular FOSOF (no calibration)
# AM Modulation Calibration
# Switching Offset Cavity Calibration
newColumns = ["Repeat",
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
              "Phase Difference (Detector 2nd Harmonic - Mixer 1) [rad]",
              "Phase Difference (Mixer 2 - Mixer 1) [rad]",
              "Digitizer Phase Difference (Digi 2 - Digi 1) [rad]",
              "Phase Difference (Detector - Mixer 2) [rad]"
              ]

def powerSpectrum(y,samplingRate):
    #win = signal.get_window( ('gaussian',600/10),600 )
    Fy = np.fft.fft(y)/np.sqrt(len(y) * samplingRate)
    f = np.fft.fftfreq(len(y),1./samplingRate)
    return f,np.abs(Fy)**2

def fit(y,dt,f):
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

def read_traces(directory):

    print("Beginning analysis...")

    # Get data from the run_parameters file
    # This run_parameters file contains information about the data run. For
    # example, it contains how many carrier and offset frequencies were used.
    # It also contains all data that was not collected by the digitizer
    # (e.g. power meter voltages collected by the LabJacks, etc.).
    filename = "Y:/data/" + directory + '/data.txt'
    runParametersDirectory = directory + '/run parameters/'
    outFilename = filename[filename.rfind("/")+1:filename.rfind(".txt")] + ".TV"
    outFileDirectory = "C:/Users/Travis/Google Drive/Travis Code/"+directory
    originalDirectory = os.getcwd()

    numberOfFilesFinished = 0

    data, runDict = get_file_data(filename)
    if not (cgxpressure_columnname in data.columns):
        newColumns.remove(cgxpressure_columnname)
        newColumns.remove(massflowrate_columnname)
        newColumns.append(cgxpressure_columnname)
        newColumns.append(massflowrate_columnname)

    if not os.path.exists(outFileDirectory):
        os.makedirs(outFileDirectory)
        # Creating the data table to be saved to a txt and csv file at the end of the
        # analysis
        dataTable = pd.DataFrame(columns = newColumns)
    elif os.path.exists(outFileDirectory) and os.path.exists(outFileDirectory+"/"+outFilename):
        os.chdir(outFileDirectory)
        dataTable = pd.read_csv(outFilename.replace(".TV","_TV.csv"))
        numberOfFilesFinished = len(dataTable)
        dataTable = dataTable.drop("Unnamed: 0",axis=1)
    else:
        dataTable = pd.DataFrame(columns = newColumns)

    with open(outFileDirectory + "/rundict_parameters.txt", "w") as f:
        f.write(json.dumps(runDict))

    print("File data acquired.")

    # Storing parameters from the run dictionary that will be used multiple
    # times throughout this code.
    numberOfFrequencies = int(runDict[freqnum_key])
    numberOfRepeats = int(runDict[repnum_key])
    numberOfAverages = int(runDict[avgnum_key])
    if offsetfreqnum_key in runDict.keys():
        numberOfOffsetFrequencies = int(runDict[offsetfreqnum_key])
    else:
        numberOfOffsetFrequencies = len(data[offsetfreq_columnname].unique())
    digitizer1CH1Range = float(runDict[digichrange_key])
    digitizer1CH2Range = float(runDict[digichrange_key])
    digitizer2CH1Range = float(runDict[digichrange_key])
    digitizer2CH2Range = float(runDict[digichrange_key])
    samplingRate = float(runDict[digisamplingrate_key])
    numberOfSamples = float(runDict[diginumsamples_key])
    numberOfMagneticFieldsx = float(runDict[numbx_key])
    numberOfMagneticFieldsy = float(runDict[numby_key])
    if nummassflow_key in runDict.keys():
        number_of_massflow_settings = float(runDict[nummassflow_key])
    else:
        number_of_massflow_settings = 1
    bundle_size = int(runDict[bundle_size_key])
    if runDict[pre910onoff_key] == True or runDict[pre910onoff_key] == 'True':
        pre910onoff = 2
    else:
        pre910onoff = 1

    # I use the number of lines to determine when the analysis has finished
    # Note that this is the number of lines that will exist in the data file
    # once the acquisition is complete. This value allows this program to
    # run concurrently with the acquisition.
    numberOfLines = numberOfFrequencies * numberOfRepeats*numberOfAverages * \
                    numberOfOffsetFrequencies * 2 * \
                    pre910onoff*numberOfMagneticFieldsx * \
                    numberOfMagneticFieldsy * number_of_massflow_settings

    # Recording the binary directory
    binaryDirectory = 'Z:/' + directory

    # Since not every data set will include all columns, I just set everything
    # here to a default null value so as not to get errors later on.
    filenameDigi1CH1 = filenameDigi1CH2 = filenameDigi2CH1 = \
    filenameDigi2CH2 = 'N/A'

    fosofAmplitude = fosofPhase = fosofCurrent = \
    fosofAmplitude2 = fosofPhase2 = fosofCurrent2 = \
    mixerAmplitude = mixerPhase = mixerCurrent = \
    detectorMixerPhaseDifference = detectorMixerPhaseDifference2 = \
    mixer2Amplitude = mixer2Current = mixer2Phase = \
    mixer1Amplitude = mixer1Current = mixer1Phase = mixerPhaseDifference = \
    preQuench910Power = preQuench1088Power = preQuench1147Power = \
    postQuench910Power = postQuench1088Power = postQuench1147Power = \
    faradayCup1a = faradayCup1b = faradayCup1c = \
    faradayCup1d = faradayCup2i = faradayCup2ii = \
    faradayCup2iii = faradayCup2iv = faradayCup3 = \
    faradayCupCentre = timeAcquired =  np.nan

    # While there is still data that has been or will be acquired but has not
    # been analyzed...
    while numberOfFilesFinished < numberOfLines:
        # Start "where we left off". By this, I mean if the program is running
        # at the same time as data collection, it will stop when it has finished
        # analyzing all acquired data and pause for a time to allow data to be
        # collected. Instead of starting from the beginning, this program will
        # pick up where it left off.
        data, runDict = get_file_data(filename)
        lineNumber = numberOfFilesFinished
        while lineNumber < len(data):
            os.chdir(binaryDirectory)

            # The repeat, average, carrier frequency and offset frequency
            # will be consistent throughout all data relevant to the current
            # line.
            repeat = data[rep_columnname][lineNumber]
            average = data[avg_columnname][lineNumber]
            tracebundle = (average - 1) // bundle_size + 1
            carrierFrequency = data[freq_columnname][lineNumber]

            # Some of the older data sets do not have the offset frequency in
            # the columns. If it is not there, it will be in the header.
            offsetFrequency = data[offsetfreq_columnname][lineNumber]

            intOffsetFrequency = int(offsetFrequency) # For use with "fit()"

            offsetChannel = data[abconfig_columnname][lineNumber]

            # Acquire data from digitizer traces for FOSOF detector and
            # mixer.
            filenameDigi1CH1 = data[detectorfname_columnname][lineNumber]
            filenameDigi1CH1 += '.digi.npy'
            filenameDigi1CH2 = data[pcomb1fname1_columnname][lineNumber]
            filenameDigi1CH2 += '.digi.npy'
            #print("FILE " + str(filenameDigi1CH1))

            print(filenameDigi1CH1)
            V1 = [None]
            while V1[0] == None:
                try:
                    V1 = np.load(filenameDigi1CH1)*digitizer1CH1Range/32767.0
                    V2 = np.load(filenameDigi1CH2)*digitizer1CH2Range/32767.0
                except:
                    V1 = [None]
                    print("Could not find one of the files. Sleeping then trying again.")
                    time.sleep(5)

            freqs,S = powerSpectrum(V1,samplingRate)

            min60Hz = intOffsetFrequency // 60

            snr_noise_index = np.intersect1d(np.where(freqs - min60Hz * 60 > 0), np.where(freqs - (min60Hz+1) * 60 <= -(freqs[1]-freqs[0])))
            signal_index = np.where(abs(freqs - float(intOffsetFrequency)) == min(abs(freqs-float(intOffsetFrequency))))
            snr_noise_index = np.delete(snr_noise_index,np.where(freqs[snr_noise_index] == freqs[signal_index]))
            print("Signal frequency in FFT: " + str(freqs[signal_index]))

            det_snr = np.sqrt(S[signal_index]) / np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Atoms): " + str(det_snr))

            min60Hz2nd = intOffsetFrequency*2 // 60

            snr_noise_index_2nd = np.intersect1d(np.where(freqs - min60Hz2nd * 60 > 0), np.where(freqs - (min60Hz2nd+1) * 60 <= -(freqs[1]-freqs[0])))
            signal_index_2nd = np.where(abs(freqs - float(intOffsetFrequency*2)) == min(abs(freqs-float(intOffsetFrequency*2))))
            snr_noise_index_2nd = np.delete(snr_noise_index_2nd,np.where(freqs[snr_noise_index_2nd] == freqs[signal_index_2nd]))
            det_snr_2nd = np.sqrt(S[signal_index_2nd]) / np.average(np.sqrt(S[snr_noise_index_2nd]))

            freqs,S = powerSpectrum(V2,samplingRate)

            m1d1_snr = np.sqrt(S[signal_index]) / np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Mixer 1.1): " + str(m1d1_snr))

            fosofAmplitude, fosofPhase, fosofCurrent = fit(V1, 1.0/samplingRate, intOffsetFrequency)
            fosofAmplitude2, fosofPhase2, fosofCurrent2 = fit(V1, 1.0/samplingRate, 2*intOffsetFrequency)
            mixerAmplitude, mixerPhase, mixerCurrent = fit(V2, 1.0/samplingRate, intOffsetFrequency)

            print("FOSOF Current: " + str(fosofCurrent))

            detectorMixerPhaseDifference = (fosofPhase - mixerPhase + 2.0 * np.pi) % (2.0 * np.pi)
            detectorMixerPhaseDifference2 = (fosofPhase2 - mixerPhase + 2.0 * np.pi) % (2.0 * np.pi)

            filenameDigi2CH1 = data[pcomb2fname_columnname][lineNumber]
            filenameDigi2CH1 += '.digi.npy'
            filenameDigi2CH2 = data[pcomb1fname2_columnname][lineNumber]
            filenameDigi2CH2 += '.digi.npy'

            V3 = [None]
            while V3[0] == None:
                try:
                    V3 = np.load(filenameDigi2CH1)*digitizer2CH1Range/32767.0
                    V4 = np.load(filenameDigi2CH2)*digitizer2CH2Range/32767.0
                except:
                    V3 = [None]
                    print("Could not find one of the files. Waiting and trying again.")
                    time.sleep(5)

            freqs,S = powerSpectrum(V3,samplingRate)

            m2_snr = np.sqrt(S[signal_index]) / np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Mixer 2): " + str(m2_snr))

            freqs,S = powerSpectrum(V4,samplingRate)

            m1d2_snr = np.sqrt(S[signal_index]) / np.average(np.sqrt(S[snr_noise_index]))
            print("SIGNAL TO NOISE (Mixer 1.2): " + str(m1d2_snr))

            mixer2Amplitude, mixer2Phase, mixer2Current = fit(V3, 1.0/samplingRate, intOffsetFrequency)
            mixer1Amplitude, mixer1Phase, mixer1Current = fit(V4, 1.0/samplingRate, intOffsetFrequency)

            mixerPhaseDifference = (mixer2Phase - mixer1Phase + 2.0 * np.pi) % (2.0 * np.pi)
            digitizer_phase_difference = (mixer1Phase - mixerPhase)
            if digitizer_phase_difference < -np.pi:
                digitizer_phase_difference = digitizer_phase_difference + 2*np.pi
            elif digitizer_phase_difference > np.pi:
                digitizer_phase_difference = digitizer_phase_difference - 2*np.pi
            detector_mixer2_phase_difference = (fosofPhase - (mixer2Phase - digitizer_phase_difference) + 2.0 * np.pi) % (2.0 * np.pi)

            analyzed_values = [tracebundle,
                               fosofAmplitude, fosofPhase, fosofCurrent, det_snr,
                               fosofAmplitude2, fosofPhase2, fosofCurrent2, det_snr_2nd,
                               mixerAmplitude, mixerPhase, mixerCurrent, m1d1_snr,
                               mixer2Amplitude, mixer2Phase, mixer2Current, m2_snr,
                               mixer1Amplitude, mixer1Phase, mixer1Current, m1d2_snr,
                               detectorMixerPhaseDifference, detectorMixerPhaseDifference2,
                               mixerPhaseDifference, digitizer_phase_difference,
                               detector_mixer2_phase_difference]

            if not (cgxpressure_columnname in data.columns):
                analyzed_values.append(1.6e-6)
                analyzed_values.append(0.25)

            newData = data.ix[lineNumber].values
            newData = np.append(newData, analyzed_values)
            newData = pd.Series(newData, index = newColumns)

            dataTable = dataTable.append(newData, ignore_index = True)
            #print("Mixer 1 Amplitude:")
            #print(dataTable["Mixer Amplitude [V]"][len(dataTable["Repeat"])-1])
            #print("Mixer 2 Amplitude:")
            #print(dataTable["Mixer 2 Amplitude [V]"][len(dataTable["Repeat"])-1])
            #print("Mixer Phase Difference:")
            #print(mixerPhaseDifference)

            # Save every thousand files (just in case of a power outage or something similar)
            lineNumber += 1
            if lineNumber % 1000 == 0:
                os.chdir(outFileDirectory)
                dataTable.to_csv(outFilename.replace(".TV","_TV.csv"), header = dataTable.columns)
                print("Saved data, finished "+str(lineNumber)+" files.")
                print("Finished "+str(float(lineNumber)/float(numberOfLines)))

        numberOfFilesFinished = len(dataTable)
        os.chdir(outFileDirectory)
        dataTable.to_csv(outFilename.replace(".TV","_TV.csv"), header = dataTable.columns)
        print("Saved and waiting 5 seconds for more files...")
        print("Finished "+str(float(lineNumber)/float(numberOfLines)))
        time.sleep(5)

    os.chdir(outFileDirectory)
    dataTable.to_csv(outFilename.replace(".txt",".csv"), header = dataTable.columns)
    print("Done.")
    print("Saved to: " + outFilename)

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
