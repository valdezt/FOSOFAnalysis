import pandas as pd
import fosoffunctionsv2 as ff
import numpy as np
import scipy.optimize as scopt
from defaultfolders import *
import matplotlib.pyplot as plt

%matplotlib inline

norder = 4
col = 'Quench Simulation Fit'

# Polynomial forcing the y-intercept to 0 W, 100% remaining population
# For use with scipy.optimize.curve_fit
def power_v_pop_fit(x, *params):
    mysum = 0.
    for p in range(1, len(params)+1):
        mysum += params[p-1]*x**p

    return 1. + mysum

def main():
    df = pd.read_csv("SOF_QuenchCurve.csv").set_index(['Off-Axis [mm]'])
    df = df.loc[1.8].set_index('Frequency [MHz]')
    print(len(df))

    tofilename = analysis_TV + "/WCPopVsPDetV/WGFits.csv"
    tofile = pd.read_csv(tofilename).set_index('Frequency [MHz]')

    toappend = pd.DataFrame({}, columns = [col])

    for f in df.index.unique().values:
        df_f = df.loc[f]
        df_f = df_f.loc[df_f['Electric Field [V/cm]'] <= 25]

        fit = scopt.curve_fit(power_v_pop_fit,
                              xdata = df_f['Electric Field [V/cm]'].values**2,
                              ydata = df_f['Fractional Population'].values,
                              p0 = [1. for i in range(norder)])

        fit = np.poly1d(np.append(np.flip(fit[0], 0), 1.0))
        toappend = toappend.append(pd.Series({col : list(fit)},
                                             name = round(f, 1)))


    tofile = toappend.combine_first(tofile)
    tofile.to_csv(tofilename)

if __name__ == '__main__':
    main()
