import pandas as pd
import numpy as np
import scipy.optimize as scopt
import fosoffunctionsv2 as ff
from defaultfolders import *

phasefile = analysis_TV + "WCPopVsPDetV/linearphases4cm.csv"
tofilename = analysis_TV + "WCPopVsPDetV/WGFits.csv"

def main():
    df = pd.read_csv(phasefile).set_index('Frequency [MHz]')
    df.sort_index(inplace=True)

    outfile = pd.read_csv(tofilename).set_index('Frequency [MHz]')

    toappend = pd.DataFrame({}, columns = ["FOSOF Simulation Fit 4 cm"])

    for f in df.index.unique():
        df_f = df.loc[f]
        fit = ff.polyfit(np.array([64, 64.64]), df_f.values, 1)
        fit = np.poly1d(fit)

        if (f < 908.0) or (f > 912.0):
            toappend = toappend.append(pd.Series([list(fit)],
                                                 index = ["FOSOF Simulation Fit 4 cm"],
                                                 name = f))

    toappend.index.names = ['Frequency [MHz]']
    outfile = toappend.combine_first(outfile)
    outfile.to_csv(tofilename)

if __name__ == '__main__':
    main()
