import pandas as pd
import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt
import fosoffunctionsv2 as ff
import os

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

def power_v_pop_fit(x, *params):
    mysum = 0.
    for p in range(1, len(params)+1):
        mysum += params[p-1]*x**p

    return 1. + mysum

norder = 4

def main():
    folder = "180517-165808 - FOSOF Acquisition - pi config, 8 V per cm PD ON 120 V, 894-898 MHz/"
    to_save = folder[:13] + "c" + folder[13:]

    if folder.find('0 config') > -1:
        zeropi_sign = 1
    else:
        zeropi_sign = -1

    sign = {'A': 1 * zeropi_sign, 'B' : -1 * zeropi_sign}

    df = pd.read_csv("C:/Users/Travis/Google Drive/Travis Code/" + folder + "data_TV.csv")
    df = df.rename(columns = {'Waveguide A Power Reading  [V]' : \
                              'Waveguide A Power Reading [V]'})

    phases_freqs = pd.read_csv("C:/Users/Travis/Google Drive/Travis Code/" + folder + "phasedata.csv")
    phases_freqs['Rounded Frequency [MHz]'] = [round(f, 4) for f in phases_freqs['Waveguide Carrier Frequency [MHz]'].values]
    phases_freqs.set_index(['Averaging Method', 'Rounded Frequency [MHz]', 'Combiner/Harmonic'], inplace = True)
    phases_freqs['Shifted Phase [rad]'] = phases_freqs['Average Phase [rad]'].values

    popvpdet_fits = pd.read_csv("C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/WCPopVsPDetV/WGFits.csv").set_index(['Frequency [MHz]', 'Waveguide'])
    popvpdet_fits['3rd Order'] = [np.poly1d(eval(fit)) for fit in popvpdet_fits['3rd Order'].values]
    popvpdet_fits['4th Order'] = [np.poly1d(eval(fit)) for fit in popvpdet_fits['4th Order'].values]

    sim_vals = pd.read_csv("C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/SOF_QuenchCurve.csv").set_index(["Off-Axis [mm]","Frequency [MHz]"])
    sim_fits = pd.DataFrame([], columns = ["Frequency [MHz]", "Fit"])

    sim_vals.sort_index(inplace = True)
    for freq in sim_vals.index.get_level_values(1).unique():
        sim_efields = sim_vals.loc[(1.8, freq)]['Electric Field [V/cm]']
        sim_pops = sim_vals.loc[(1.8, freq)]['Fractional Population']
        fit = scopt.curve_fit(power_v_pop_fit,
                              xdata = sim_efields.values**2,
                              ydata = sim_pops.values,
                              p0 = [1. for i in range(norder)])
        fit = np.poly1d(np.append(np.flip(fit[0], 0), 1))
        sim_fits = sim_fits.append({'Frequency [MHz]' : freq, 'Fit' : fit}, ignore_index = True)

    sim_fits.set_index('Frequency [MHz]', inplace = True)

    phase_shifts = pd.read_csv("C:/Users/Travis/Google Drive/Travis Code/FOSOF Data Analysis/FOSOFDataAnalysis/WCPopVsPDetV/PhaseShifts.csv").set_index('Freq (MHz)')

    for indx in df.index:
        freq = df.loc[indx,'Waveguide Carrier Frequency [MHz]']
        a_or_b = df.loc[indx,'Sub-Configuration']

        exp_pop_A = popvpdet_fits.loc[(round(freq,1), 'A'), '4th Order'](df.loc[indx, 'Waveguide A Power Reading [V]'])
        exp_pop_B = popvpdet_fits.loc[(round(freq,1), 'B'), '4th Order'](df.loc[indx, 'Waveguide B Power Reading [V]'])

        power_A = scopt.fsolve(lambda x: exp_pop_A - sim_fits.loc[round(freq, 1), 'Fit'](x), [1])[0]
        power_B = scopt.fsolve(lambda x: exp_pop_B - sim_fits.loc[round(freq, 1), 'Fit'](x), [1])[0]
        avg_power = np.average([power_A, power_B])

        phase_diff_prediction = phase_shifts.loc[round(freq, 1), 'Phase Difference [rad]']
        dp = (avg_power/64.)*(phase_diff_prediction)/1.01
        phases_freqs.at[('RMS Uncertainty', round(freq, 4), 'Combiner 1, Harmonic 1'), 'Shifted Phase [rad]'] += dp
        df.at[indx, ff.fosofphasediff_columnname] += sign[a_or_b] * dp
        print(freq, phase_diff_prediction, avg_power/64.)

    plt.errorbar(phases_freqs.index.get_level_values(1).unique(),
                 phases_freqs.loc[('RMS Uncertainty', slice(None), 'Combiner 1, Harmonic 1'),'Average Phase [rad]'].values,
                 yerr = phases_freqs.loc[('RMS Uncertainty', slice(None), 'Combiner 1, Harmonic 1'),'Phase Error [rad]'].values,
                 fmt = 'r.')
    plt.errorbar(phases_freqs.index.get_level_values(1).unique(),
                 phases_freqs.loc[('RMS Uncertainty', slice(None), 'Combiner 1, Harmonic 1'),'Shifted Phase [rad]'].values,
                 yerr = phases_freqs.loc[('RMS Uncertainty', slice(None), 'Combiner 1, Harmonic 1'),'Phase Error [rad]'].values,
                 fmt = 'b.')
    plt.show()

    # if not os.path.exists('C:/Users/Travis/Google Drive/Travis Code/' + to_save):
    #     os.mkdir('C:/Users/Travis/Google Drive/Travis Code/' + to_save)
    #
    # df.to_csv('C:/Users/Travis/Google Drive/Travis Code/' + to_save + 'data_TV.csv')

if __name__ == '__main__':
    main()
