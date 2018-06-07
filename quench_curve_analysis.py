# -*- coding: utf-8 -*-
"""
Created on Tue May 02 13:22:17 2017

@author: travis
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as scopt
from scipy.interpolate import UnivariateSpline as uvs

%matplotlib inline

foldername = "180531-172629 - Quench Cavity Calibration - post-1088 PD ON"
att_v_setting_column = "Attenuator Voltage Setting [V]"
cavity_name = "Post-Quench 1088"
att_v_reading_column = cavity_name + " Attenuator Voltage Reading (Quenches On) [V]"
on_off_column = 'Digitizer DC On/Off Ratio'
data = pd.read_csv("y:\\data\\"+foldername+"\\data.txt",comment="#")

# Sort by set voltage on V_attenuators
avgd_data = data.reset_index().groupby(att_v_setting_column) \
                              .agg([np.average,
                                    lambda x: np.std(x)/np.sqrt(len(x-1))])

mx = max(avgd_data[on_off_column]['average'].values)
mn = min(avgd_data[on_off_column]['average'].values)

poly = np.polyfit(avgd_data[att_v_reading_column]['average'].values,
                  avgd_data[on_off_column]['average'].values, 10,
                  w=1.0/avgd_data[on_off_column]['<lambda>'].values)

p = np.poly1d(poly)

spline = uvs(avgd_data[att_v_reading_column]['average'].values,
             avgd_data[on_off_column]['average'].values,
             w=1.0/avgd_data[on_off_column]['<lambda>'].values**2,
             k = 3, s = len(avgd_data)/2)

min_ain = min(avgd_data[att_v_reading_column].index.unique())
max_ain = max(avgd_data[att_v_reading_column].index.unique())

rn = np.linspace(min_ain,max_ain,1000)

plt.figure(figsize=[20,20])
plt.plot(rn,p(rn))
# plt.plot(rn, spline(rn))
plt.errorbar(avgd_data[att_v_reading_column]['average'].values,
             avgd_data[on_off_column]['average'].values,
             xerr=avgd_data[att_v_reading_column]['<lambda>'].values,
             yerr=avgd_data[on_off_column]['<lambda>'].values,fmt='r.')
plt.show()
soln = scopt.minimize_scalar(p,bounds=(1., 6.),method='bounded')
print(soln['x'])
x = soln['x']
print(p(x))
print(min(avgd_data[on_off_column]['average'].values))
print(avgd_data[att_v_reading_column]['average'].values[np.where(avgd_data[on_off_column]['average'].values==mn)])

tenpercentremaining = mx - 0.8 * (mx-mn)
print(np.roots(p-tenpercentremaining))

#print(max(avgd_data['Digitizer DC (Quenches On) [V]']['average']))
# print(mx-0.9*(mx-mn))
# print(np.roots(p-(mx-0.9*(mx-mn))))
