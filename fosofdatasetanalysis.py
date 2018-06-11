'''
New (01/2018) data analysis for FOSOF data sets with new FOSOFtware. To be
blunt, includes less bullshit and a more streamlined summary than its
predecessor, fosof_data_into_summary_filesv2.py.

First just to clarify some current FOSOF lingo:
    - "Frequency" refers to the waveguide carrier frequency; usually represented
    in MHz.
    - "Repeat" is the iteration over the entire frequency range
    - "Configuration" is the rotational configuration of the waveguides. The
    naming scheme is arbitrary, but important. First, note that the waveguides
    have numbers carved into their sides (1 and 2) to denote which waveguide is
    which. "0 configuration" denotes when Waveguide 2 is further upstream than
    Waveguide 1. "pi configuration" is the opposite scenario.
    - "Sub-Configuration" denotes which RF channel carries the offset frequency
    in addition to the carrier frequency, blind offset and jitter (blind and
    jitter are for un-biased data collection). The Sub-Configurations are
    denoted "A" and "B" as are the channels on the RF generator. Currently,
    Waveguide 1 is hooked up to Channel B and Waveguide 2 is hooked up to
    Channel A. Why? I don't know. Just because. Sometimes, we refer to Waveguide
    2 as Waveguide A and vice versa. We switch which waveguide carries the
    offset frequency because by taking data in both sub-configurations and
    subtracting the results, we can get rid of some contributions to the phase
    from the detection system that can be approximated as an RC filter on the
    phase.
    - "Average" refers to the trace number within one repeat of the frequency
    domain. For each repeat we take N averages at A and N averages at B.
    - "Trace Bundle" refers to a bundle of traces within one repeat that were
    taken sequentially. We may take 8 averages at each subconfiguration every
    repeat, but we may not take 8 traces at A followed by 8 traces at B. We
    may take 2 traces at A, then 2 traces at B and switch randomly (2 then 2)
    until we take 8 traces at each subconfiguration in total. The Trace bundle
    will then refer to which of the "bundles" of 2 consecutive traces a
    particular trace belongs. Confusing? Yes. Useful? I think so. Confidence,
    right?

Other than being compatible with new FOSOF data, the changes are as follows:
    - No longer outputs useless values to a csv file (i.e. slope of the A
    subconfiguration lineshape).
    - Exports plots of various data to the TravisCode data folder (monitor
    values vs time, FOSOF lineshape vs freq).
    - Exports a csv of the [freq, phases, phase_error] values of the lineshape
    so it can be re-plotted easier. Will also help with 0 - pi data as the
    averaging will not have to be redone.
    - Averages data together in multiple ways:
        1) Uses the rms uncertainty for each repeat as the uncertainty in all
        frequencies for that repeat (i.e. all frequencies in repeat 1 have the
        same uncertainty, as do all those in repeat 2, etc.).
        2) Uses the calculated standard deviation in the mean for each repeat
        and frequency.
    - Rejects data for which the spread of the phases in the trace bundle or
    repeat is greater than 60 degrees (pi/3 radians). This is a safety net to
    guard against our 2 pi problem for phases (they are a modular quantity). By
    spread, I'm referring to the maximum difference between two traces in a set.
    The spread will be checked within each trace bundle (if bundle size > 1) and
    again for each frequency in a repeat (if bundle size != number of averages
    and number of averages != 1). If the number of averages is 1, this cannot
    be done at all and the data is (frankly) untrustworthy.
'''

import sys
sys.path.insert(0, 'C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fosoffunctionsv2 as ff
import pickle as p
import json, os, ast, itertools
from datetime import datetime as dt

# For single (0 or pi) data
summary_file = 'C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/data_summary.csv'
# For full (0 - pi) data
fosof_summary_file = 'C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/fosof_summary.csv'
simulation_file = 'C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/simulationshifts.csv'
analyzed_data_location = 'C:/Users/Travis/Google Drive/Travis Code/'
phaseaverage_column = "Average Phase [rad]"
phaseerr_column = "Phase Error [rad]"
phasen_column = "Number of Points Averaged"
unperturbed_datatype = (800, 0.0, 0.0, 'off') #, 0.25)

class FOSOFSummary(object):
    '''
    Class to visualize the results from all FOSOF data sets in the fosof summary
    file.
    '''

    def __init__(self, fosof = True):
        self.fosof = fosof
        if self.fosof:
            self.fosof_summary = pd.read_csv(fosof_summary_file)
            self.fosof_summary = self.fosof_summary.set_index('Dataset Timestamp')
            self.simulations = pd.read_csv(simulation_file)
            self.simulations = self.simulations.set_index('Separation [cm]')
        else:
            self.fosof_summary = pd.read_csv(summary_file)
            self.fosof_summary = self.fosof_summary.set_index('Dataset Timestamp')

    def plotlineshape(self, timestamp, c_h = None, averaging_method = None,
                      corrected = None):
        '''
        Plots the FOSOF lineshape and its fit for the data at the specified
        timestamp, combiner/harmonic (c_h), and averaging method.
        '''

        if self.fosof:
            fosof_data_folder = self.fosof_summary.loc[timestamp]['Folder'].values[0]
        else:
            fosof_data_folder = self.fosof_summary.loc[timestamp]['Folder']
        fosof_data = pd.read_csv(fosof_data_folder + 'phasedata.csv')
        fit_data = pd.read_csv(fosof_data_folder + 'fitdata.csv',
                               converters = {"Least-Squares Fit" : ast.literal_eval,
                                             "Parameter Error" : ast.literal_eval})

        if c_h == None:
            ch_list = fosof_data['Combiner/Harmonic'].unique()
        else:
            ch_list = [c_h]

        if averaging_method == None:
            avg_method_list = fosof_data['Averaging Method'].unique()
        else:
            avg_method_list = [averaging_method]

        if corrected == None:
            corrected_list = fosof_data['Corrected'].unique()
        else:
            corrected_list = [corrected]

        # Sort the data into different data types
        fosof_data = fosof_data.set_index(['Averaging Method',
                                           'Combiner/Harmonic', 'Corrected'])
        fit_data = fit_data.set_index(['Averaging Method', 'Combiner/Harmonic',
                                       'Corrected'] + ff.type_groupby_columns)

        for ch in ch_list:
            for avg_method in avg_method_list:
                for corr in corrected_list:
                    self.plotchavgcorr_lineshape(ch, avg_method, corr,
                                                 fosof_data, fit_data)
        plt.show()

    def plotchavgcorr_lineshape(self, ch, avg_method, corr, fosof_data,
                                fit_data):
        '''
        Creates a series of plots for each data type, given a combiner/harmonic
        (ch) pair, an averaging method, the FOSOF data points, and the fit
        data. Mainly used by plotlineshape.
        '''

        # Count the different number of data types (B fields, 910,
        # offset freq, mass flow)
        ch_av_data = fosof_data.loc[avg_method, ch, corr].reset_index()
        # if self.fosof:
        #     ch_av_data = ch_av_data.set_index(ff.type_groupby_columns[1:])
        # else:
        ch_av_data = ch_av_data.set_index(ff.type_groupby_columns)
        n_plots = len(ch_av_data) / \
                  len(ch_av_data[ff.frequency_columnname].unique())
        indices = ch_av_data.index.unique()

        # Make sure that there's enough plot space. Python rounds
        # down when performing division of integers.
        npo2 = (n_plots) // 2 + n_plots % 2

        plt.figure()

        for plot_num in range(n_plots):
            # ind = indices[plot_num]
            # print(fit_data.index)
            # print(ind)
            # print(fit_data.loc[:,:,ind[0], ind[1], ind[2], ind[3]])
            ch_av_fit = fit_data.loc[(avg_method, ch, corr) + indices[plot_num]]
            fit_func = np.poly1d(ch_av_fit.loc['Least-Squares Fit'])
            fit_err = ch_av_fit.loc['Parameter Error']
            chi2 = ch_av_fit.loc['Chi Squared']
            prob_chi2 = ch_av_fit.loc['Probability of Chi Squared']
            if 'Resonant Frequency [MHz]' in ch_av_fit.index:
                res_freq = ch_av_fit.loc['Resonant Frequency [MHz]']
                res_freq_err = ch_av_fit.loc['Error in Resonant Frequency [MHz]']
                res_freq_string = 'Resonant Frequency = ' + \
                                  str(round(res_freq,4)) + '+/-' + \
                                  str(round(res_freq_err,4)) + ' MHz\n'
            else:
                phase_offset = fit_func[0]
                phase_offset_error = fit_err[0]
                res_freq_string = 'Phase Offset = ' + \
                                  str(round(phase_offset, 5)) + " +/- " + \
                                  str(round(phase_offset_error, 5)) + " rad\n"

            if 'Subtracted' in avg_method:
                slope_string = ''
            else:
                slope_string = 'Slope = ' + str(round(fit_func[1],5)) + \
                               '+/-' + str(round(fit_err[1],5)) + ' rad/MHz\n'

            if corr == True:
                corrected = ', Corrected'
            else:
                corrected = ''

            plt.subplot(npo2, 2, plot_num+1)
            plt.title(ch + ' (' + avg_method + ')' + corrected + '\n' + \
                      ','.join(np.array(indices[plot_num]).astype(str)))
            freqs = ch_av_data.loc[indices[plot_num]][ff.frequency_columnname].values
            phases = ch_av_data.loc[indices[plot_num]]['Average Phase [rad]'].values
            errors = ch_av_data.loc[indices[plot_num]]['Phase Error [rad]'].values
            label = slope_string + res_freq_string + \
                    '$\chi^2$ = ' + str(round(chi2,3)) + '\n' \
                    'P($\chi^2$) = ' + str(round(prob_chi2,3))
            plt.plot(freqs, fit_func(freqs-910.), 'r', label = label)
            plt.errorbar(freqs, phases, yerr = errors, fmt = 'b.')
            plt.legend()

    def plotlinecentres(self, separation, electric_field, accel_v,
                        c_h = None, averaging_method = None,
                        freq_range = None, plotslope = False,
                        corrected = None):
        '''
        Plots all of the line centres for a single separation and electric field
        amplitude.
        '''
        if plotslope:
            val_column = 'Slope [rad/MHz]'
            err_column = 'Error in Slope [rad/MHz]'
        else:
            val_column = 'Resonant Frequency [MHz]'
            err_column = 'Error in Resonant Frequency [MHz]'

        # Check that all lists or choices are valid
        summary = self.fosof_summary.copy()
        summary = summary.set_index('Waveguide Separation [cm]')

        if not (separation in summary.index.unique()):
            print("Could not find the listed waveguide separation.")
            return

        summary = summary.loc[separation]
        summary = summary.set_index('Peak RF Field ' \
                                                    'Amplitude [V/cm]')

        if not (electric_field in summary.index.unique()):
            print("Could not find the listed peak electric field amplitude.")
            return

        summary = summary.loc[electric_field]
        summary = summary.set_index('Accelerating Voltage [kV]')

        if not (accel_v in summary.index.unique()):
            print("Could not find the listed accelerating voltage.")
            print(summary.index.unique())
            return

        summary = summary.loc[accel_v]

        if not (freq_range == None):
            try:
                summary = summary.loc[summary['Frequency Range [MHz]'] == str(freq_range)]
            except:
                print('Could not filter to frequency range specified.')

        if c_h == None:
            ch_list = list(summary['Combiner/Harmonic'].unique())
        elif not (c_h in summary['Combiner/Harmonic'].unique()):
            print("Could not find the combiner/harmonic specified.")
            return
        else:
            ch_list = [c_h]

        if averaging_method == None:
            avgmethod_list = list(summary['Averaging Method'].unique())
        elif not (averaging_method in summary['Averaging Method'].unique()):
            print("Could not find the averaging method specified.")
            return
        else:
            avgmethod_list = [averaging_method]

        # Create a different figure for each combiner/harmonic and averaging
        # method combination
        for ch in ch_list:
            summary_ch = summary.copy()
            summary_ch = summary_ch.set_index('Combiner/Harmonic')
            summary_ch = summary_ch.loc[ch]
            for avgmethod in avgmethod_list:
                summary_to_plot = summary_ch.copy()
                summary_to_plot = summary_to_plot.set_index('Averaging Method')
                summary_to_plot = summary_to_plot.loc[avgmethod]
                summary_to_plot = summary_to_plot \
                                    .set_index(ff.type_groupby_columns[1:])

                # Determine the number of subplots in each figure
                n_plots = len(summary_to_plot.index.unique()) + 1
                npo2 = n_plots // 2

                plt.figure()
                plt.title(ch + '\n' + avgmethod)
                typelist = list(summary_to_plot.index.unique())

                counter = 1
                for type in typelist:
                    # Get the data to plot on this chart
                    this_plot = summary_to_plot.loc[type].copy()

                    # Find the statistical information
                    avg_std = ff.polyfit(range(len(this_plot)),
                                         this_plot[val_column].values,
                                         0,
                                         w = 1./this_plot[err_column].values**2)
                    print(avg_std)
                    avg = avg_std[0][0]
                    std = avg_std[3][0]
                    avg_upper = avg + std # For plotting errors
                    avg_lower = avg - std
                    chi2 = avg_std[1]
                    probchi2 = avg_std[2]
                    maxval = len(this_plot)
                    val_name = val_column[:val_column.find('[') - 1]
                    val_units = val_column[val_column.find('[') + 1: \
                                           val_column.find(']')]
                    label = val_name + ' = ' + str(round(avg,4)) + \
                            ' +/- ' + str(round(std,4)) + ' ' + \
                            val_units + '\n$\chi^2$ = ' + \
                            str(round(chi2,4)) + '\nP($\chi^2$) = ' + \
                            str(round(probchi2,4))

                    # Plot the individual points nicely.
                    plt.subplot(npo2, npo2, counter)
                    plt.errorbar(range(maxval),
                                 this_plot[val_column].values,
                                 yerr = this_plot[err_column].values,
                                 fmt = 'r.'
                                )

                    plt.plot([-1, maxval], [avg, avg], 'b-',
                             label = label)
                    plt.plot([-1, maxval], [avg_upper, avg_upper], 'b--',
                             [-1, maxval], [avg_lower, avg_lower], 'b--')
                    plt.xlabel('Trial #')
                    plt.ylabel(val_column)
                    plt.legend()
                    counter += 1
        plt.show()

    def plotlinecentres_e(self, separation, accel_vs = None,
                          electric_fields = None, c_h = None,
                          averaging_method = None, freq_range = None,
                          corrected = None):
        '''
        Plots all of the line centres for a single separation and electric field
        amplitude. This function will first average the resonant frequencies to
        E points (one point for each electric field value) and find the
        chi-square statistic for that distribution (for all N measurements to
        fit to E points). The error bars of all N points will then be expanded
        by the square root of the chi-square to reflect our lack of
        understanding of our error. The final plot contain the statistics for
        all N shifted measurements with expanded error bars to fit to a single
        frequency.
        '''

        # Check that all lists or choices are valid
        summary = self.fosof_summary.copy()
        summary = summary.set_index('Waveguide Separation [cm]')
        sims = self.simulations.copy()

        if not (separation in summary.index.unique()):
            print("Could not find the listed waveguide separation.")
            return

        elif not (separation in sims.index.unique()):
            print("No simulation data for this separation.")
            return

        # Fit simulated f vs E^2 to a polynomial
        sims = sims.loc[separation].set_index('E [V/cm]')
        f_v_esq = ff.polyfit(sims.index.values**2,
                             sims['Shift [MHz]'], 2)
        f_v_esq = np.poly1d(f_v_esq[0])

        summary = summary.loc[separation]
        summary = summary.set_index('Peak RF Field ' \
                                                    'Amplitude [V/cm]')

        if electric_fields != None:
            for electric_field in electric_fields:
                if not (electric_field in summary.index.unique()):
                    electric_fields.remove(electric_field)
            if len(electric_fields) == 0:
                print('Could not find any of the electric fields.')
                return
            elif len(electric_fields) == 1:
                return plotlinecentres(separation, electric_fields[0],
                                       c_h = c_h,
                                       averaging_method = averaging_method)
        else:
            electric_fields = summary.index.unique()

        summary = summary.reset_index()

        if c_h == None:
            ch_list = list(summary['Combiner/Harmonic'].unique())
        elif not (c_h in summary['Combiner/Harmonic'].unique()):
            print("Could not find the combiner/harmonic specified.")
            return
        else:
            ch_list = [c_h]

        if averaging_method == None:
            avgmethod_list = list(summary['Averaging Method'].unique())
        elif not (averaging_method in summary['Averaging Method'].unique()):
            print("Could not find the averaging method specified.")
            return
        else:
            avgmethod_list = [averaging_method]

        if corrected == None:
            corrected_list = [True, False]
        elif not (corrected in summary['Corrected'].unique()):
            print("Could not find the corrected value specified. Should be " \
                  "True or False.")
            return
        else:
            corrected_list = [corrected]

        if not (type(freq_range) == type(None)):
            try:
                print("SELECTING FREQ RANGE")
                print(summary['Frequency Range [MHz]'])
                summary = summary.loc[summary['Frequency Range [MHz]'] == str(freq_range)]
            except:
                print("Could not find frequency range specified.")

        all_chavg = [zip(x, avgmethod_list) \
                      for x in \
                      itertools.permutations(ch_list, len(avgmethod_list))]
        all_chavg = [item for sublist in all_chavg for item in sublist]

        all_chavgcorr = [zip(x, all_chavg) for x in \
                         itertools.permutations(corrected_list, len(all_chavg))]
        all_chavgcorr = [item for sublist in all_chavgcorr for item in sublist]

        type_list = summary.set_index(ff.type_groupby_columns[1:]).index.unique()

        if len(type_list) > 1:
            all_combos = [zip(x, all_chavgcorr) \
                          for x in \
                          itertools.permutations(type_list, len(all_chavgcorr))]
            all_combos = [item for sublist in all_combos for item in sublist]
        else:
            all_combos = [list(chavco) + [type_list[0]] for chavco in all_chavgcorr]
        print(all_combos)

        # Create a different figure for each combiner/harmonic and averaging
        # method combination
        for (ch, avgmethod, corr, type) in all_combos:
            summary_ch = summary.copy()
            summary_ch.set_index('Combiner/Harmonic', inplace = True)
            summary_ch = summary_ch.loc[ch]

            summary_corr = summary_ch.copy()
            summary_corr.set_index('Corrected', inplace = True)
            sumary_corr = summary_corr.loc[corr]

            summary_to_plot = summary_corr.copy()
            summary_to_plot.set_index('Averaging Method', inplace = True)
            summary_to_plot = summary_to_plot.loc[avgmethod]
            summary_to_plot = summary_to_plot \
                                .set_index(ff.type_groupby_columns[1:])

            # Determine the number of subplots in each figure
            n_plots = len(summary_to_plot.index.unique())
            npo2 = n_plots // 2 + n_plots % 2

            counter = 1
            # Get the data to plot on this chart
            this_plot = summary_to_plot.loc[type].copy()
            this_plot = this_plot.set_index('Peak RF Field Amplitude [V/cm]')
            this_plot = this_plot.loc[electric_fields]

            # For each electric field, compute the average of all the
            # resonant frequencies and the chi-square statistic of the
            # distribution. Add this chi-square value to the total
            # chi-squared for the entire set of data to fit to N
            # values corresponding to N electric fields.
            total_chi2 = 0.0
            for index in this_plot.index.unique():
                try:
                    avg_std = ff.polyfit(range(len(this_plot.loc[index])),
                                         this_plot.at[index,'Resonant Frequency [MHz]'],
                                         0, w = 1./this_plot.at[index,'Error in Resonant Frequency [MHz]']**2)
                    total_chi2 += avg_std[1] * (len(this_plot.loc[index]) - 1)
                except TypeError:
                    pass

                # Shift the resonant frequencies down by the predicted
                # shift value.
                this_plot.at[index,'Resonant Frequency [MHz]'] = this_plot.at[index,'Resonant Frequency [MHz]'] - \
                                                                 f_v_esq(index**2)

            # Compute the total REDUCED chi-square statistic
            total_chi2 = total_chi2 / (len(this_plot) - len(electric_fields))

            # Expand the error bars of each individual point by the
            # sqrt of the total chi-square
            this_plot['Error in Resonant Frequency [MHz]'] = this_plot['Error in Resonant Frequency [MHz]'] * np.sqrt(total_chi2)

            # Find the statistical information for all the shifted points
            # to fit to a single value. This tests our consistency with
            # the predicted shift.
            avg_std = ff.polyfit(range(len(this_plot)),
                                 this_plot['Resonant Frequency [MHz]'].values,
                                 0,
                                 w = 1./this_plot['Error in Resonant Frequency [MHz]'].values**2)

            # Shortcuts to values for plotting
            avg = avg_std[0][0]
            std = avg_std[3][0]
            avg_upper = avg + std # For plotting errors
            avg_lower = avg - std
            chi2 = avg_std[1]
            probchi2 = avg_std[2]
            maxval = len(this_plot)
            label = 'Resonant Frequency = ' + str(round(avg,4)) + \
                    ' +/- ' + str(round(std,4)) + ' MHz\n$\chi^2$ = ' + \
                    str(round(chi2,4)) + '\nP($\chi^2$) = ' + \
                    str(round(probchi2,4))

            plt.figure()
            plt.title(ch + '\n' + avgmethod)
            minval = 0

            # Plot the resonant frequencies for each of the electric field
            # values.
            for index in this_plot.index.unique():
                try:
                    maxval_e = len(this_plot.loc[index])
                    freqs = this_plot.loc[index]['Resonant Frequency [MHz]'].values
                    errors = this_plot.loc[index]['Error in Resonant Frequency [MHz]'].values
                except AttributeError:
                    maxval_e = 1
                    freqs = np.array([this_plot.loc[index]['Resonant Frequency [MHz]']])
                    errors = np.array([this_plot.loc[index]['Error in Resonant Frequency [MHz]']])
                plt.errorbar(range(minval, minval + maxval_e),
                             freqs,
                             yerr = errors,
                             fmt = '.', label = str(index) + " V/cm"
                            )
                minval += maxval_e

            # Plot the average and confidence interval
            plt.plot([-1, maxval], [avg, avg], 'b-',
                     label = label)
            plt.plot([-1, maxval], [avg_upper, avg_upper], 'b--',
                     [-1, maxval], [avg_lower, avg_lower], 'b--')
            plt.xlabel('Trial #')
            plt.ylabel('Resonant Frequency (+ Blind) [MHz]')
            plt.legend()
            counter += 1
        plt.show()

class FOSOFDataSet(object):
    '''
    Base class for 0 - pi FOSOF data sets. Will plot, save, etc.
    '''

    def __init__(self, locations, descrim = np.pi/3, splittime = 10):
        '''
        If the datasets passed have already been analyzed, pull the summary
        data from their files. Otherwise, analyze the data sets and save them
        to the summary file.
        Then, subtract the two sets frequency by frequency and create a set of
        0 - pi files containing the phase vs frequency data points, the fit
        parameters for the FOSOF lineshape, and time series data.
        '''

        # Analyze or read in the data sets if already analyzed
        self.dataset_0 = DataSet(locations[0][0], descrim = descrim,
                                 splittime = splittime)
        if len(locations[0]) > 1:
            for loc in locations[0][1:]:
                ds0 = DataSet(loc, descrim = descrim, splittime = splittime)
                self.dataset_0.timestamp = self.dataset_0.timestamp + ", " + ds0.timestamp
                self.dataset_0.fitdata = self.dataset_0.fitdata.append(ds0.fitdata)
                self.dataset_0.phasedata = self.dataset_0.phasedata.append(ds0.phasedata)
                self.dataset_0.t_avg = self.dataset_0.t_avg.append(ds0.t_avg)
                self.dataset_0.tf_avg = self.dataset_0.tf_avg.append(ds0.tf_avg)
                self.dataset_0.tfab_avg = self.dataset_0.tfab_avg.append(ds0.tfab_avg)

        self.dataset_pi = DataSet(locations[1][0], descrim = descrim,
                                  splittime = splittime)
        if len(locations[1]) > 1:
            for loc in locations[1][1:]:
                dspi = DataSet(loc, descrim = descrim, splittime = splittime)
                self.dataset_pi.timestamp = self.dataset_pi.timestamp + ", " + dspi.timestamp
                self.dataset_pi.fitdata = self.dataset_pi.fitdata.append(dspi.fitdata)
                self.dataset_pi.phasedata = self.dataset_pi.phasedata.append(dspi.phasedata)
                self.dataset_pi.t_avg = self.dataset_pi.t_avg.append(dspi.t_avg)
                self.dataset_pi.tf_avg = self.dataset_pi.tf_avg.append(dspi.tf_avg)
                self.dataset_pi.tfab_avg = self.dataset_pi.tfab_avg.append(dspi.tfab_avg)

        if len(self.dataset_0.timestamp) > 92:
            self.timestamp = self.dataset_0.timestamp[:89] + ",,, - " + \
                             self.dataset_pi.timestamp[:89] + ",,,"
        else:
            self.timestamp = self.dataset_0.timestamp + " - " + \
                             self.dataset_pi.timestamp
        if os.path.exists(fosof_summary_file):
            summary_data = pd.read_csv(fosof_summary_file) \
                             .set_index('Dataset Timestamp')
        else:
            summary_data = pd.DataFrame()
        self.location = analyzed_data_location + self.timestamp + \
                        " - Zero - Pi/"

        self.dataset_0.phasedata.reset_index(inplace = True)
        self.dataset_0.phasedata.set_index(ff.frequency_columnname, inplace = True)
        self.dataset_pi.phasedata.reset_index(inplace = True)
        self.dataset_pi.phasedata.set_index(ff.frequency_columnname, inplace = True)

        # Determine integer frequency range for summary file
        min_freq = min(self.dataset_0.phasedata.index.values)
        max_freq = max(self.dataset_0.phasedata.index.values)
        print(min_freq)
        self.freq_range = max_freq - min_freq
        self.freq_range = str(int(round(self.freq_range,0)))

        # If the data has already been analyzed, ask the user if they want to
        # analyze it again.
        if self.timestamp in summary_data.index.unique():
            yn = raw_input('This data set has already been analyzed. Do you' \
                           ' want to analyze it again? [Y/N]\t\t')

            if yn != 'Y' and yn != 'y':
                self.fosof_data = pd.read_csv(self.location + "phasedata.csv")
                self.fitdata = pd.read_csv(self.location + "fitdata.csv",
                                           converters = {"Least-Squares Fit" : ast.literal_eval,
                                                         "Parameter Error" : ast.literal_eval})

                self.t_avg = pd.read_csv(self.location + "t_avgd_data.csv",
                                         header = [0,1], index_col = range(6))

                self.tf_avg = pd.read_csv(self.location + "tf_avgd_data.csv",
                                          header = [0,1], index_col = range(7))

                self.tfab_avg = pd.read_csv(self.location + "tfab_avgd_data.csv",
                                            header = [0,1], index_col = range(8))

                self.summary_data = summary_data.loc[self.timestamp]
                return

        # Subtract 0 - pi
        self.fosof_data = self.analyze_fosof(self.dataset_0, self.dataset_pi)
        self.fosof_data = self.fosof_data.reset_index() \
                                         .set_index(['Combiner/Harmonic',
                                                     'Averaging Method',
                                                     'Corrected'] + \
                                                     ff.type_groupby_columns[:])

        # Fit data for both combiners, as well as unique and RMS uncertainties
        fitdata_columns = ["Least-Squares Fit", "Chi Squared",
                           "Probability of Chi Squared", "Parameter Error",
                           "Resonant Frequency [MHz]",
                           "Error in Resonant Frequency [MHz]"]
        self.fitdata = pd.DataFrame(columns = ['Combiner/Harmonic',
                                               'Averaging Method',
                                               'Corrected'] + \
                                               ff.type_groupby_columns[:] + \
                                               fitdata_columns[:])

        for ind in self.fosof_data.index.unique():
            data_to_fit = self.fosof_data.loc[ind].copy()
            data_to_fit = data_to_fit.reset_index().set_index(ff.frequency_columnname)

            # This fit method expands errors by sqrt(chi^2)
            fitdata = ff.polyfit(data_to_fit.index - 910.0,
                                 data_to_fit[phaseaverage_column].values, 1,
                                 w = 1./data_to_fit[phaseerr_column].values**2,
                                 find_x0 = True)

            fitdata = pd.Series(fitdata, index = fitdata_columns)
            fitdata = fitdata.append(pd.Series(ind, index = ['Combiner/Harmonic',
                                                             'Averaging Method',
                                                             'Corrected'] + \
                                                             ff.type_groupby_columns[:]))
            fitdata = fitdata.append(pd.Series())
            self.fitdata = self.fitdata.append(fitdata, ignore_index = True)

        # Make sure all arrays are lists so they can be parsed upon calling
        # this information later.
        self.fitdata['Least-Squares Fit'] = map(list, self.fitdata['Least-Squares Fit'].values)
        self.fitdata['Parameter Error'] = map(list, self.fitdata['Parameter Error'].values)
        self.fitdata['Resonant Frequency [MHz]'] = map(lambda x: x + 910, self.fitdata['Resonant Frequency [MHz]'].values)

        # Find the difference in start times
        t_0 = dt.strptime(self.dataset_0.timestamp[:13],'%y%m%d-%H%M%S')
        t_pi = dt.strptime(self.dataset_pi.timestamp[:13],'%y%m%d-%H%M%S')
        delta_t = t_0 - t_pi

        # Determine which dataset time data to shift by delta_t
        if delta_t == abs(delta_t):
            index_names = list(self.dataset_0.t_avg.index.names)
            t_avg = self.dataset_0.t_avg.copy().reset_index()
            t_avg['Time'] = map(lambda x: x + delta_t.seconds/60., t_avg['Time'].values)
            self.dataset_0.t_avg = t_avg.set_index(index_names)

            index_names = list(self.dataset_0.tf_avg.index.names)
            tf_avg = self.dataset_0.tf_avg.copy().reset_index()
            tf_avg['Time'] = map(lambda x: x + delta_t.seconds/60., tf_avg['Time'].values)
            self.dataset_0.tf_avg = tf_avg.set_index(index_names)

            index_names = list(self.dataset_0.tfab_avg.index.names)
            tfab_avg = self.dataset_0.tfab_avg.copy().reset_index()
            tfab_avg['Time'] = map(lambda x: x + delta_t.seconds/60., tfab_avg['Time'].values)
            self.dataset_0.tfab_avg = tfab_avg.set_index(index_names)
        else:
            index_names = list(self.dataset_pi.t_avg.index.names)
            t_avg = self.dataset_pi.t_avg.copy().reset_index()
            t_avg['Time'] = map(lambda x: x + delta_t.seconds/60., t_avg['Time'].values)
            self.dataset_pi.t_avg = t_avg.set_index(index_names)

            index_names = list(self.dataset_pi.tf_avg.index.names)
            tf_avg = self.dataset_pi.tf_avg.copy().reset_index()
            tf_avg['Time'] = map(lambda x: x + delta_t.seconds/60., tf_avg['Time'].values)
            self.dataset_pi.tf_avg = tf_avg.set_index(index_names)

            index_names = list(self.dataset_pi.tfab_avg.index.names)
            tfab_avg = self.dataset_pi.tfab_avg.copy().reset_index()
            tfab_avg['Time'] = map(lambda x: x + delta_t.seconds/60., tfab_avg['Time'].values)
            self.dataset_pi.tfab_avg = tfab_avg.set_index(index_names)

        # Loading and compiling all the time series data
        self.t_avg = pd.concat([self.dataset_0.t_avg, self.dataset_pi.t_avg])
        self.tf_avg = pd.concat([self.dataset_0.tf_avg, self.dataset_pi.tf_avg])
        self.tfab_avg = pd.concat([self.dataset_0.tfab_avg, self.dataset_pi.tfab_avg])

        # Saving all the compiled and calculated data
        if not os.path.exists(self.location):
            os.mkdir(self.location)

        self.fosof_data.to_csv(self.location + "phasedata.csv")
        self.fitdata.to_csv(self.location + "fitdata.csv")
        self.t_avg.to_csv(self.location + "t_avgd_data.csv")
        self.tf_avg.to_csv(self.location + "tf_avgd_data.csv")
        self.tfab_avg.to_csv(self.location + "tfab_avgd_data.csv")

        self.summary_data = pd.DataFrame()
        for i in range(len(self.fitdata)):
            ch_summary_data = {'Folder' : self.location,
                               'Peak RF Field Amplitude [V/cm]' : self.dataset_0 \
                                                                      .rundict_parameters["Waveguide Electric Field [V/cm]"],
                               'Waveguide Separation [cm]' : self.dataset_0 \
                                                                 .rundict_parameters["Waveguide Separation [cm]"],
                               'Dataset Type' : 'Normal',
                               'Resonant Frequency [MHz]' : self.fitdata.iloc[i]['Resonant Frequency [MHz]'],
                               'Error in Resonant Frequency [MHz]' : self.fitdata.iloc[i]['Error in Resonant Frequency [MHz]'],
                               'Slope [rad/MHz]' : self.fitdata.iloc[i]['Least-Squares Fit'][0],
                               'Error in Slope [rad/MHz]' : self.fitdata.iloc[i]['Parameter Error'][0],
                               'Y-Intercept [rad]' : self.fitdata.iloc[i]['Least-Squares Fit'][1],
                               'Error in Y-Intercept [rad]' : self.fitdata.iloc[i]['Parameter Error'][1],
                               'Chi Squared' : self.fitdata.iloc[i]['Chi Squared'],
                               'Combiner/Harmonic' : self.fitdata.iloc[i]['Combiner/Harmonic'],
                               'Averaging Method' : self.fitdata.iloc[i]['Averaging Method'],
                               'Offset Frequency [Hz]' : self.fitdata.iloc[i]['Offset Frequency [Hz]'],
                               'B_x [Gauss]' : self.fitdata.iloc[i]['B_x [Gauss]'],
                               'B_y [Gauss]' : self.fitdata.iloc[i]['B_y [Gauss]'],
                               'Pre-Quench 910 State' : self.fitdata.iloc[i]['Pre-Quench 910 State'],
                               'Mass Flow Rate [CC]' : self.fitdata.iloc[i]['Mass Flow Rate [CC]'],
                               'Frequency Range [MHz]' : self.freq_range,
                               'Accelerating Voltage [kV]' : self.dataset_0 \
                                                                 .rundict_parameters["Accelerating Voltage [kV]"]
                               }

            ch_summary_data = pd.Series(ch_summary_data,
                                        name = self.timestamp,
                                       )

            self.summary_data = self.summary_data.append(ch_summary_data)
            self.summary_data.index.name = 'Dataset Timestamp'

        if self.timestamp in summary_data.index.unique():
            summary_data.loc[self.timestamp] = self.summary_data
        else:
            summary_data = summary_data.append(self.summary_data)

        summary_data.to_csv(fosof_summary_file)

    def analyze_fosof(self, dataset_0, dataset_pi):
        '''
        Sorts data by type and averaging method, the performs subtraction before
        returning data.
        '''

        # Group all data by data type and averaging method
        new_type = ff.type_groupby_columns[:] + ['Averaging Method',
                                                 'Combiner/Harmonic',
                                                 'Corrected']
        dataset_0_phase = dataset_0.phasedata.reset_index().groupby(new_type)
        dataset_pi_phase = dataset_pi.phasedata.reset_index().groupby(new_type)

        groups_to_subtract = dataset_0_phase.groups.keys()
        fosof_data = pd.DataFrame()

        # Subtract 0 phases - pi phases for every data type
        for group in groups_to_subtract:
            # Combiner 1, Harmonic 1
            zero_group = dataset_0_phase.get_group(group)
            pi_group = dataset_pi_phase.get_group(group)

            zero_minus_pi = self.zero_minus_pi(zero_group, pi_group, new_type)
            fosof_data = fosof_data.append(zero_minus_pi)

        return fosof_data

    def zero_minus_pi(self, zero_data, pi_data, type_columns):
        '''
        Subtracts zero and pi phases, being sure to divide everything by two,
        including the error.
        '''

        # Sort by frequency
        zero_data[ff.frequency_columnname] = [round(f, 4) for f in zero_data[ff.frequency_columnname].values]
        pi_data[ff.frequency_columnname] = [round(f, 4) for f in pi_data[ff.frequency_columnname].values]
        x_0 = zero_data.set_index(ff.frequency_columnname)
        x_pi = pi_data.set_index(ff.frequency_columnname)

        # Calculate
        x_zero_minus_pi = pd.DataFrame({ff.frequency_columnname : [],
                                        phaseaverage_column : [],
                                        phaseerr_column : [],
                                        phasen_column : []})
        for frequency in x_0.index.unique():
            if frequency in x_pi.index.unique():
                phaseavg = (x_0.at[frequency, phaseaverage_column] - x_pi.at[frequency, phaseaverage_column])/2
                phaseerr = np.sqrt(x_0.at[frequency, phaseerr_column]**2 + x_pi.at[frequency, phaseerr_column]**2)/2
                phasen = x_0.at[frequency, phasen_column] + x_pi.at[frequency, phasen_column]
                freq_dict = {ff.frequency_columnname : frequency,
                             phaseaverage_column : phaseavg,
                             phaseerr_column : phaseerr,
                             phasen_column : phasen}
                x_zero_minus_pi = x_zero_minus_pi.append(freq_dict,
                                                         ignore_index = True)

        # Append type data and return
        for col in type_columns:
            x_zero_minus_pi[col] = x_0[col].values[0]

        # x_zero_minus_pi[phaseaverage_column] = x_zero_minus_pi[phaseaverage_column].values
        x_zero_minus_pi = x_zero_minus_pi.set_index(ff.frequency_columnname)

        return x_zero_minus_pi

class DataSet(object):
    '''
    Base class for dataset analysis results. Will contain data frames, fit
    functions, plotted images, etc.
    '''

    def __init__(self, location, descrim = np.pi, splittime = 10):
        '''
        Reads and analyzes the data from the data set located at
        C:/Users/Travis/Google Drive/Travis Code/location/data_TV.csv.
        '''

        self.descrim = descrim
        self.splittime = splittime
        self.timestamp = location[:13]
        self.location = analyzed_data_location + location + '/'
        self.filename = self.location + "data_TV.csv"
        self.errorfile = self.location + "errors.txt"
        self.to_summary_file = pd.DataFrame()

        # Open summary file to check if this data set has already been
        # analyzed
        if os.path.exists(summary_file):
            summary_data = pd.read_csv(summary_file)
        else:
            summary_data_columns = ['Folder',
                                    'Configuration',
                                    'Peak RF Field Amplitude [V/cm]',
                                    'Dataset Type']
            summary_data = pd.DataFrame(columns = summary_data_columns)
            summary_data.index.name = 'Dataset Timestamp'
            summary_data = summary_data.reset_index()

        with open(self.location + "/rundict_parameters.txt") as f:
            self.rundict_parameters = json.load(f)

        print(location)
        if location[:13] in summary_data['Dataset Timestamp'].values:
            yn = raw_input("This data set has already been analyzed. Do you" \
                           " want to analyze it again? [Y/N]\t\t")
            if yn == 'N':
                self.fitdata = pd.read_csv(self.location + 'fitdata.csv',
                                           converters = {"Least-Squares Fit" : ast.literal_eval,
                                                         "Parameter Error" : ast.literal_eval})

                self.phasedata = pd.read_csv(self.location + 'phasedata.csv')
                self.phasedata = self.phasedata.set_index(ff.frequency_columnname)

                self.t_avg = pd.read_csv(self.location + 't_avgd_data.csv',
                                         header = [0,1], index_col = range(6))
                self.tf_avg = pd.read_csv(self.location + 'tf_avgd_data.csv',
                                          header = [0,1], index_col = range(7))
                self.tfab_avg = pd.read_csv(self.location + 'tfab_avgd_data.csv',
                                            header = [0,1], index_col = range(8))

                return

        # Clearing the error file from previous runs
        f = open(self.errorfile,'w')
        f.close()

        # Setting type of dataset
        n_Bfield = int(self.rundict_parameters['Number of B_y Steps']) + \
                   int(self.rundict_parameters['Number of B_x Steps']) - 1
        onoff_910 = str(self.rundict_parameters["Pre-Quench 910 On/Off"]) == 'True'

        self.subtract = False
        self.datatype = 'Normal'

        # self.subtract will affect the analyze_fosof_phases function
        if n_Bfield > 1 and onoff_910:
            self.datatype = 'Hybrid'
            self.subtract = True
        elif n_Bfield > 1:
            self.datatype = 'B Field'
            self.subtract = True
        elif onoff_910:
            self.datatype = '910 On/Off'
            self.subtract = True

        print('Reading data...')
        self.full_data = pd.read_csv(self.filename)

        print('Analyzing data from first power combiner...')
        # Obtain detector - power combiner phase/fit data for both combiners
        # and for 2nd harmonic on the first (forward) combiner
        c1_h1 = self.analyze_fosof_phases(self.full_data.copy(),
                                          ff.fosofphasediff_columnname,
                                          subtract = self.subtract)

        c1_h1[0]['Combiner/Harmonic'] = 'Combiner 1, Harmonic 1'
        c1_h1[0]['Corrected'] = False
        c1_h1[1]['Combiner/Harmonic'] = 'Combiner 1, Harmonic 1'
        c1_h1[1]['Corrected'] = False

        c1_h1_c = self.analyze_fosof_phases(self.full_data.copy(),
                                            ff.fosofphasediff_c_columnname,
                                            subtract = self.subtract)

        c1_h1_c[0]['Combiner/Harmonic'] = 'Combiner 1, Harmonic 1'
        c1_h1_c[0]['Corrected'] = True
        c1_h1_c[1]['Combiner/Harmonic'] = 'Combiner 1, Harmonic 1'
        c1_h1_c[1]['Corrected'] = True
        print('Done first power combiner.')

        print('Analyzing data from second power combiner...')
        c2_h1 = self.analyze_fosof_phases(self.full_data.copy(),
                                          ff.fosofphasediffm2_columnname,
                                          subtract = self.subtract)
        c2_h1[0]['Combiner/Harmonic'] = 'Combiner 2, Harmonic 1'
        c2_h1[0]['Corrected'] = False
        c2_h1[1]['Combiner/Harmonic'] = 'Combiner 2, Harmonic 1'
        c2_h1[1]['Corrected'] = False

        c2_h1_c = self.analyze_fosof_phases(self.full_data.copy(),
                                            ff.fosofphasediffm2_c_columnname,
                                            subtract = self.subtract)

        c2_h1_c[0]['Combiner/Harmonic'] = 'Combiner 2, Harmonic 1'
        c2_h1_c[0]['Corrected'] = True
        c2_h1_c[1]['Combiner/Harmonic'] = 'Combiner 2, Harmonic 1'
        c2_h1_c[1]['Corrected'] = True
        print('Done second power combiner.')

        # Creating a complete dataframe (both combiners)
        self.fitdata = pd.concat([c1_h1[0], c2_h1[0], c1_h1_c[0], c2_h1_c[0]])
        self.fitdata.to_csv(self.location + 'fitdata.csv')

        self.phasedata = pd.concat([c1_h1[1], c2_h1[1], c1_h1_c[1], c2_h1_c[1]])
        self.phasedata.to_csv(self.location + 'phasedata.csv')

        # Average data over time and plot
        print('Analyzing time series data.')
        self.t_avg = self.average_by(self.full_data.copy(),
                                     ff.type_groupby_columns + \
                                     [ff.time_columnname],
                                     ff.time_columnname,
                                     'Time',
                                     columns = ff.bytime_columns)
        self.t_avg.to_csv(self.location + 't_avgd_data.csv')
        print('Analyzing time and frequency series data.')
        self.tf_avg = self.average_by(self.full_data.copy(),
                                      ff.type_groupby_columns + \
                                      [ff.time_columnname,
                                       ff.frequency_columnname],
                                       ff.time_columnname,
                                       'Time_Freq',
                                      columns = ff.bytimefreq_columns)
        self.tf_avg.to_csv(self.location + 'tf_avgd_data.csv')
        print('Analyzing time and frequency series data.')
        self.tfab_avg = self.average_by(self.full_data.copy(),
                                        ff.type_groupby_columns + \
                                        [ff.time_columnname,
                                         ff.frequency_columnname,
                                         ff.subconfig_columnname],
                                         ff.time_columnname,
                                         'Time_Freq_AB',
                                        columns = ff.bytimefreqab_columns)
        self.tfab_avg.to_csv(self.location + 'tfab_avgd_data.csv')
        print('Done.')
        if not ('Accelerating Voltage [kV]' in self.rundict_parameters.keys()):
            self.rundict_parameters['Accelerating Voltage [kV]'] = 49.87

        newdata_dict = {'Folder' : self.location,
                        'Configuration' : self.rundict_parameters["Configuration"],
                        'Peak RF Field Amplitude [V/cm]' : self.rundict_parameters["Waveguide Electric Field [V/cm]"],
                        'Waveguide Separation [cm]' : self.rundict_parameters['Waveguide Separation [cm]'],
                        'Accelerating Voltage [kV]' : self.rundict_parameters['Accelerating Voltage [kV]'],
                        'Dataset Type' : self.datatype
                        }
        newdata = pd.Series(newdata_dict,
                            name = self.timestamp,
                            )

        summary_data = summary_data.set_index('Dataset Timestamp')
        if self.timestamp in summary_data.index.unique():
            summary_data.loc[self.timestamp] = newdata
            summary_data.to_csv(summary_file)
        else:
            summary_data = summary_data.append(newdata)
            summary_data.to_csv(summary_file)

    def analyze_fosof_phases(self, df, columnname, subtract = False,
                             is_subtracted = False):
        '''
        Performs the following algorithm:
            1) Separate by data type if the data is a systematic test (e.g. B
            field, on/off, etc.) as well as by repeat, average, frequency,
            subconfiguration and trace bundle.
            2) Average each trace bundle as long as the spread is not greater
            than the self.descrim value.
            3) Subtract the A and B subconfigurations for each repeat, frequency
            and trace bundle.
            4) Average together the different avg(A) - avg(B) values for each
            repeat (if there's more than one trace bundle in the repeat).
            5) Find the rms uncertainty for the repeat (averaging over all
            frequencies), and duplicate the DataFrame.
            6) Average together all the repeats in two ways:
                a) A weighted average of the data from each repeat.
                b) A weighted average using the rms uncertainty from each
                repeat.
            7) Fit the data in each case, and save the results. Also, save the
            averaged data vs frequency.
        Note: in the case that the phase spread is greater than self.descrim,
        the corresponding data will be disregarded and noted down.
        '''

        print('Sorting by AB')
        data = df.groupby(ff.tracebundle_ab_groupby_columns)

        # Averaging data within trace bundles
        data = data.apply(self.descrim_average,
                          col = columnname)
        print('Sorting by trace bundle')
        data = data.reset_index().groupby(ff.tracebundle_groupby_columns)
        print('Sorted. Subtracting.')
        data = data.apply(self.a_minus_b)
        data = data.dropna(axis = 0, how = 'any')
        print("LENGTH " + str(len(df)))

        print('Sorting by repeat')
        # Averaging all the a_data[i] - b_data[i] values together for each
        # frequency and each repeat
        data = data.reset_index().groupby(ff.repeat_groupby_columns)
        data = data.apply(self.descrim_average,
                          col = phaseaverage_column,
                          e_col = phaseerr_column)

        print('Calculating RMS Error')
        # Finding the rms error for each repeat and type and making a copy of the
        # dataframe with constant error over each repeat and type.
        data = data.reset_index()
        data_by_repeat = data.groupby(ff.repeat_columnname)

        # Separate data by type and repeat
        data_rms = data.copy().reset_index().set_index(ff.type_groupby_columns + [ff.repeat_columnname])

        # For every repeat
        for repeat in data[ff.repeat_columnname].unique():

            # Retrieve the data for that repeat and group data by type only
            data_by_type = data_by_repeat.get_group(repeat).reset_index().groupby(ff.type_groupby_columns)

            # For every type
            for datatype in data_by_type.groups.keys():
                # Retrieve data for that type, calculate the rms error and apply
                # the rms error to all frequencies in the repeat
                data_to_rms = data_by_type.get_group(datatype)
                data_rms_error = np.sqrt(np.mean(data_to_rms[phaseerr_column].values**2))
                data_rms.at[datatype + (repeat,), phaseerr_column] = data_rms_error

        print('Separating by frequency only...')
        # Averaging all data together for each frequency by keeping individual
        # uncertainties
        # print(data[phaseerr_column].values)
        # plt.plot(data[ff.frequency_columnname].values,
        #          data[phaseaverage_column].values, 'r.')
        # plt.show()
        data = data.groupby(ff.freq_groupby_columns)
        data = data.apply(self.descrim_average,
                          col = phaseaverage_column,
                          e_col = phaseerr_column)

        # Averaging all data together for each frequency using the RMS
        # uncertainty
        data_rms = data_rms.groupby(ff.freq_groupby_columns)
        data_rms = data_rms.apply(self.descrim_average,
                                  col = phaseaverage_column,
                                  e_col = phaseerr_column)

        print('Ordering and fitting data.')
        type_without_massflow = ff.type_groupby_columns[:]
        type_without_massflow.remove(ff.massflow_columnname)

        # If the data was already subtracted, the phase uncertainties for
        # the unperturbed data will be zero (subtracting trace by trace). This
        # will cause problems, so we drop it.
        if is_subtracted:
            data = data.reset_index().set_index(type_without_massflow)
            data = data.drop(unperturbed_datatype)

            data_rms = data_rms.reset_index().set_index(type_without_massflow)
            data_rms = data_rms.drop(unperturbed_datatype)

        # Set up to fit the data to a straight line
        grouped_data = data.groupby(ff.type_groupby_columns)
        grouped_data_rms = data_rms.groupby(ff.type_groupby_columns)
        data = data.reset_index().set_index(ff.type_groupby_columns)

        data_rms = data_rms.reset_index().set_index(ff.type_groupby_columns)

        fitdata_columns = ["Least-Squares Fit", "Chi Squared",
                           "Probability of Chi Squared", "Parameter Error",
                           "Averaging Method"] + \
                           ff.type_groupby_columns[:]
        fitdata_to_return = pd.DataFrame(columns = fitdata_columns)
        convert_to_lists = lambda x: list(x) if type(x) == np.ndarray else x

        # Fit to a flat line if the data is subtracted data
        if is_subtracted:
            order = 0
        else:
            order = 1

        for group in grouped_data.groups.keys():
            group_data = grouped_data.get_group(group).reset_index().set_index(ff.frequency_columnname)
            group_data[phaseaverage_column] = ff.mod_wrap(group_data[phaseaverage_column].values, -1)
            phases = ff.unwrap_fosof_line(group_data[phaseaverage_column].values, group_data.index.values)[:, 1]
            group_data[phaseaverage_column] = phases / 2.
            data.at[group, phaseaverage_column] = phases / 2.
            fitdata = ff.polyfit(group_data.index.values - 910.,
                                 phases,
                                 order,
                                 w = 1./group_data[phaseerr_column].values**2)
            fitdata = [convert_to_lists(x) for x in fitdata]
            fitdata = pd.Series(fitdata + ['Unique Uncertainty'] + list(group),
                                index = fitdata_columns)
            fitdata_to_return = fitdata_to_return.append(fitdata, ignore_index = True)

            data_rms.at[group,phaseaverage_column] = ff.mod_wrap(data_rms.at[group,phaseaverage_column], n = -1)
            group_data_rms = grouped_data_rms.get_group(group).reset_index().set_index(ff.frequency_columnname)
            group_data_rms[phaseaverage_column] = ff.mod_wrap(group_data_rms[phaseaverage_column].values, -1)
            phases = ff.unwrap_fosof_line(group_data_rms[phaseaverage_column].values, group_data_rms.index.values)[:, 1]
            group_data_rms[phaseaverage_column] = phases / 2.
            data_rms.at[group, phaseaverage_column] = phases / 2.
            fitdata_rms = ff.polyfit(group_data_rms.index.values - 910.,
                                     phases,
                                     order,
                                     w = 1./group_data_rms[phaseerr_column].values**2)

            fitdata_rms = [convert_to_lists(x) for x in fitdata_rms]
            fitdata_rms = pd.Series(fitdata_rms + ['RMS Uncertainty'] + list(group),
                                    index = fitdata_columns)
            fitdata_to_return = fitdata_to_return.append(fitdata_rms, ignore_index = True)

        fitdata_to_return = fitdata_to_return.set_index(ff.type_groupby_columns)
        data = data.reset_index().set_index(ff.frequency_columnname)
        data_rms = data_rms.reset_index().set_index(ff.frequency_columnname)

        if subtract:
            print('Attempting to subtract the perturbed data and re-analyze.')

            # Group the data
            data_sub = df.set_index(type_without_massflow) \
                         .loc[unperturbed_datatype,:].reset_index()

            # data_sub = data_sub.set_index([ff.repeat_columnname,
            #                                ff.average_columnname,
            #                                ff.subconfig_columnname,
            #                                ff.tracebundle_columnname,
            #                                ff.massflow_columnname,
            #                                ff.frequency_columnname])
            # isnanlist = np.isnan([list(v[:2]) + list(v[3:]) for v in data_sub.index.values])
            # print(np.where([x.any() == True for x in isnanlist]))
            data_sub = data_sub.groupby([ff.repeat_columnname,
                                         ff.average_columnname,
                                         ff.subconfig_columnname,
                                         ff.tracebundle_columnname,
                                         ff.massflow_columnname,
                                         ff.frequency_columnname])

            # Remove the unperturbed data
            df_dropped = df.set_index(type_without_massflow) \
                           .drop(unperturbed_datatype).reset_index()
            df_dropped = df_dropped.groupby([ff.repeat_columnname,
                                             ff.average_columnname,
                                             ff.subconfig_columnname,
                                             ff.tracebundle_columnname,
                                             ff.massflow_columnname,
                                             ff.frequency_columnname])
            data_sub = self.difference(data_sub, df_dropped, columnname)
            print('Finished subtracting... re-analyzing')

            # Analyze the FOSOF traces of the altered data set as normal
            fitdata_sub, phase_values_sub = self.analyze_fosof_phases(data_sub, columnname, is_subtracted = True)

            fitdata_sub['Averaging Method'] = [x + ' (Subtracted)' for x in fitdata_sub['Averaging Method'].values]
            fitdata_to_return['Averaging Method'] = [x + ' (Normal)' for x in fitdata_to_return['Averaging Method'].values]
            fitdata_df = pd.concat([fitdata_sub, fitdata_to_return])
            phase_values_sub_unique = phase_values_sub.ix['Unique Uncertainty']#.reset_index()
            phase_values_sub_rms = phase_values_sub.ix['RMS Uncertainty']#.reset_index()
            phasedata_df = pd.concat([data, data_rms,
                                      phase_values_sub_unique,
                                      phase_values_sub_rms],
                                      keys = ['Unique Uncertainty (Normal)',
                                              'RMS Uncertainty (Normal)',
                                              'Unique Uncertainty (Subtracted)',
                                              'RMS Uncertainty (Subtracted)'],
                                      names = ['Averaging Method', ff.frequency_columnname])

            return fitdata_df, phasedata_df

        return fitdata_to_return, pd.concat([data, data_rms],
                                            keys = ["Unique Uncertainty", "RMS Uncertainty"],
                                            names = ['Averaging Method', ff.frequency_columnname])

    def difference(self, to_subtract, df, column_to_subtract, is_phase = True):
        '''Subtracts unperturbed data from perturbed data
        '''

        new_df = pd.DataFrame()

        # Grouped by: offset freq, repeat, average, subconfig, tracebundle,
        # and frequency
        print('To subtract: ' + str(len(df.groups.keys())))
        finished_groups = 0
        for group in df.groups.keys():
            this_group = df.get_group(group)
            this_group = this_group.set_index(ff.type_groupby_columns)

            try:
                unperturbed = to_subtract.get_group(group)[column_to_subtract].values[0]
            except KeyError as e:
                print('Could not find unperturbed data. Try performing the calculations manually.')
                return None

            # Indexed by B_x, B_y, 910 On/Off State
            for index in this_group.index.unique():
                # Subtract data
                this_group.at[index, column_to_subtract] = this_group.at[index, column_to_subtract] - unperturbed

                # Keep between 0 and 2*pi if it's a phase
                if is_phase:
                    this_group.at[index, column_to_subtract] = ff.mod_wrap(this_group.at[index, column_to_subtract])

                # Append to the new data frame
                new_index = list(index)
                new_series = pd.Series(this_group.loc[index], name = tuple(new_index))
                new_df = new_df.append(new_series)

            finished_groups += 1
            if finished_groups % 100 == 0:
                print('Finished ' + str(finished_groups))

        # Set names of the index columns
        new_df.index = pd.MultiIndex.from_tuples(new_df.index)
        new_df.index.names = ff.type_groupby_columns

        # Reset the index and return the data frame
        return new_df.reset_index()

    def descrim_average(self, x, col, e_col = None):
        '''
        Takes the average of a set of data as long as the dataset has more than
        one element and as long as its spread is not greater than the
        descrim_average value for the DataSet object.
        '''


        # Retrieve the data column of the structure x
        v = x[col].values
        f = x[ff.frequency_columnname].values

        e = None
        if e_col:
            # Retrieve the error column of the structure x
            e = x[e_col].values

            # If only some of the errors are NaN values, remove the x and e
            # indices corresponding to those NaN e values
            if not np.isnan(e).all():
                wherenane = np.isnan(e)

                # Remove x and e values where e is not a number
                v = v[~wherenane]
                e = e[~wherenane]

            # If there are no error values, we just take a normal average
            else:
                e = None

        # Remove values from v (and e if it exists) where v is NaN
        wherenanv = np.isnan(v)
        v = v[~wherenanv]
        if not isinstance(e, type(None)):
            e = e[~wherenanv]

        # Make sure the values are not larger than the speciefied descrim
        # value.
        spread = self.check_spread(v)
        if spread < self.descrim:
            n = len(v)

            # If there's more than one item in the group, calculate the
            # average and stdev
            if n > 1:
                # Weighted average if errors are given
                if not isinstance(e, type(None)):
                    fosofphaseavg, fosofphasestd = ff.average_and_std(v, w = 1/e**2)
                else:
                    fosofphaseavg, fosofphasestd = ff.average_and_std(v)
                    fosofphasestd = fosofphasestd / np.sqrt(n) # Standard dev in the mean

            # If there's only one item in the group, cannot calculate the
            # standard deviation or average
            else:
                fosofphaseavg = v[0]
                if not isinstance(e, type(None)):
                    fosofphasestd = e[0]
                else:
                    fosofphasestd = np.nan
        else:
            print(v)
            print(spread)
            print(self.descrim)
            f = open(self.errorfile,'a')
            for c in ff.tracebundle_ab_groupby_columns:
                if c in x.columns:
                    f.write(c + ', ' + str(col) + ' : ' + str(x[c].values) + '\n')
            f.close()

            fosofphaseavg = np.nan
            fosofphasestd = np.nan
            n = 0

        data = {phaseaverage_column : fosofphaseavg,
                phaseerr_column : fosofphasestd,
                phasen_column : n}

        return pd.Series(data)

    def a_minus_b(self, x):
        '''
        Subtracts the phases at the B subconfiguration from those at the A
        subconfiguration, so long as both groups have valid averages/values.
        '''

        x = x.groupby(ff.subconfig_columnname)
        x_a = x.get_group("A")
        x_b = x.get_group("B")

        x_a_minus_b = {}
        x_a_minus_b[phaseaverage_column] = (x_a[phaseaverage_column].values - x_b[phaseaverage_column].values)[0]
        #x_a_minus_b[phaseaverage_column] = ff.mod_wrap(x_a_minus_b[phaseaverage_column],0)
        x_a_minus_b[phaseerr_column] = (np.sqrt(x_a[phaseerr_column].values**2 + x_b[phaseerr_column].values**2))[0]/2.
        x_a_minus_b[phasen_column] = min(x_a[phasen_column].values, x_b[phasen_column].values)[0]

        return pd.Series(x_a_minus_b)

    def check_spread(self, x, mod = True):
        '''Calculates the maximum spread of a set of data.'''

        # Check the spread by wrapping the values from [0, 2*pi) and from
        # [-pi, pi) in case values are right near the edge of one of the ranges
        if mod:
            tocheck_1 = ff.mod_wrap(x, -1)
            tocheck_2 = ff.mod_wrap(x, 0)
            return min(abs(max(tocheck_1) - min(tocheck_1)),
                       abs(max(tocheck_2) - min(tocheck_2)))
        else:
            tocheck = x
        return abs(max(tocheck) - min(tocheck))

    def average_by(self, df, by, plotby, avgbyname, columns = "ALL"):
        '''
        Calculates the average and standard error of a grouped subset of a data
        frame. Also plots the averaged data.

        Parameters:
            df : DataFrame to average
            by : Name of columns by which to group the data to be averaged.
            plotby : Name of column(s) to put on the x axis of the plots.
            columns (default ALL) : The name of the columns to average.
        '''

        # If no columns are specified, assume all are to be averaged
        if columns == "ALL":
            columns = df.columns

        # Copy the data frame, just to be safe.
        to_avg = df[columns + by].copy()

        # If averaging by the time, break it up into 1/self.splittime of the
        # data set
        if ff.time_columnname in by:
            totaltime = self.check_spread(to_avg[ff.time_columnname].values, mod = False)
            ti = min(to_avg[ff.time_columnname].values)
            # Fraction of time completed, split into self.splittime groups
            to_avg[ff.time_columnname] = (to_avg[ff.time_columnname].values - ti) // (totaltime/self.splittime)

            # Convert to number of minutes completed
            to_avg[ff.time_columnname] = to_avg[ff.time_columnname] * (totaltime/600)

        to_avg = to_avg.groupby(by) \
                       .agg([np.average, lambda x: np.std(x, ddof=1)/np.sqrt(len(x))]) \
                       .rename(columns = {'average' : 'Average',
                                          '<lambda>' : 'Error'})

        # Create plots for each of the items in by EXCEPT for plotby
        # self.plot_average_by(to_avg, by[:by.index(plotby)] + by[by.index(plotby)+1:], plotby, avgbyname)

        return to_avg

    def plot_average_by(self, df, different_types, index, avgbyname):
        '''
        Plots the data in df against the df[index] for each of the categories
        specified in different_plots.

        Parameters:
            df : DataFrame to be plotted.
            different_types : Indexes by which a new figure will be created.
            index : Index column to plot on the x axis.
        '''

        # Separate data into different types
        columns = df.columns.get_level_values(0).unique()
        df = df.reset_index().set_index(different_types)

        # Each column will be on its own set of axes
        num_plots = len(columns)
        if num_plots % 2 == 1:
            num_plots += 1
        fig_number = 1

        # For every data type, set up a new figure.
        for ind in df.index.unique():
            current_plot_number = 1
            title = self.make_title_from_multiindex(df.index, ind)
            fig, axs = plt.subplots(int(num_plots/2), 2,figsize = (10,10))
            plt.suptitle(title)
            axs = axs.flatten()
            # On each figure, plot each of the columns as a subplot.
            for column in columns:
                toplot = df.ix[ind].set_index(index)
                axs[current_plot_number-1].set_title(column)
                axs[current_plot_number-1].errorbar(toplot.index, toplot[column]['Average'].values,
                                                  yerr = toplot[column]['Error'].values, fmt = 'r.')
                current_plot_number += 1

            plt.savefig(self.location + "/avg_by_" + avgbyname + "_" + str(fig_number) + ".pdf", format = 'pdf')
            plt.clf()
            plt.close()
            fig_number += 1

    def make_title_from_multiindex(self, full_index, this_index):
        title = ""

        for name, val in zip(full_index.names,this_index):
            title = title + name + ": " + str(val) + ", "

        # The [:-2] gets rid of the last ", "
        return title[:-2]

def main():
    f = open('fosofdataset_file_list.txt', 'r')
    files_to_read = f.readlines()
    f.close()


    if sys.argv[1] == '-r':
        for filename in files_to_read:
            ds = DataSet(filename[:-1])
    elif sys.argv[1] == '-zp':
        for filename in files_to_read:
            filename = filename.split(';')
            filename[0] = eval(filename[0])
            filename[1] = eval(filename[1])
            ds = FOSOFDataSet([filename[0], filename[1]], descrim = np.pi)
    elif sys.argv[1] == 'ls':
        dataset = sys.argv[2] # dataset_timestamp in fosof_summary.csv
        type = sys.argv[3] # r or zp
        if type == 'r':
            summary = FOSOFSummary(fosof = False)
        elif type == 'zp':
            summary = FOSOFSummary()
        else:
            print(type)
            sys.exit('Could not decide which type of summary to open. ' \
                     'Try using "zp" or "r" as the type. See documentation ' \
                     'for more informaton.')
        summary.plotlineshape(dataset)
    elif sys.argv[1] == 'lc':
        separation = int(sys.argv[2]) # Waveguide separation [cm]
        e_field = float(sys.argv[3]) # Electric field amplitude [V/cm]
        accel_v = float(sys.argv[4])
        plot_slope = False
        if len(sys.argv) > 5:
            plot_slope = eval(sys.argv[5])

        freq_range = None
        if len(sys.argv) > 6:
            freq_range = str(sys.argv[6])
        summary = FOSOFSummary()
        summary.plotlinecentres(separation, e_field, accel_v,
                                plotslope = plot_slope, freq_range = freq_range)
    elif sys.argv[1] == 'lc_e':
        separation = int(sys.argv[2]) # Waveguide separation [cm]
        if len(sys.argv) > 3:
            freq_range = str(sys.argv[3])
        summary = FOSOFSummary()
        summary.plotlinecentres_e(separation, freq_range = freq_range)

if __name__ == '__main__':
    main()
