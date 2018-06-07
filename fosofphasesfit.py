import pandas as pd
import fosoffunctionsv2 as ff
import numpy as np
import scipy.optimize as scopt
from defaultfolders import *
import matplotlib.pyplot as plt

%matplotlib inline

col = "FOSOF Simulation Fit"
separations = ["4", "5", "6", "7"]
cols = [" ".join([col, x]) for x in [" ".join([sep, "cm"]) for sep in separations]]
fromfile = analysis_TV + "WCPopVsPDetV/phases.csv"

def main():
    df = pd.read_csv(fromfile).set_index('Frequency [MHz]')

    tofilename = analysis_TV + "/WCPopVsPDetV/WGFits.csv"
    tofile = pd.read_csv(tofilename).set_index('Frequency [MHz]')

    toappend = pd.DataFrame({}, columns = cols)

    for f in df.index.unique().values:
        df_f = df.loc[f]
        df_f = df_f.set_index('Separation [cm]')

        df_f.sort_index(inplace = True)
        fitlist = []
        for s in df_f.index.unique():
            df_s = df_f.loc[s]
            fit = ff.polyfit(df_s['P [V/cm]^2'].values,
                             df_s['Phase [rad]'].values, 2)

            fit = np.poly1d(fit[0])
            fitlist.append(list(fit))
        toappend = toappend.append(pd.Series({cols[0] : fitlist[0],
                                              cols[1] : fitlist[1],
                                              cols[2] : fitlist[2],
                                              cols[3] : fitlist[3]},
                                             name = round(f, 1)))

    toappend.index.names = ["Frequency [MHz]"]
    tofile = toappend.combine_first(tofile)
    tofile.to_csv(tofilename)

if __name__ == '__main__':
    main()
