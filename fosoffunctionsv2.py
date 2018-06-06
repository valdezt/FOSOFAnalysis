# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:59:15 2017

@author: travis
"""

# -*- coding: utf-8 -*-

# Necessary functions
from __future__ import division
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sqlite3 as lite
import os, time
from datetime import datetime as dt

# Columns that are important for FOSOF phase/resonant frequency analysis
repeat_columnname = "Repeat"
average_columnname = "Average"
subconfig_columnname = "Sub-Configuration"
frequency_columnname = "Waveguide Carrier Frequency [MHz]"
offsetfreq_columnname = "Offset Frequency [Hz]"
mixerphasediff_columnname = "Phase Difference (Mixer 2 - Mixer 1) [rad]"
fosofphasediff_columnname = "Phase Difference (Detector - Mixer 1) [rad]"
fosofphasediffh2_columnname = "Phase Difference (Detector 2nd Harmonic - Mixer 1) [rad]"
fosofphasediffm2_columnname = "Phase Difference (Detector - Mixer 2) [rad]"
bxfield_columnname = "B_x [Gauss]"
byfield_columnname = "B_y [Gauss]"
pre910_columnname = "Pre-Quench 910 State"
tracebundle_columnname = "Trace Bundle"
massflow_columnname = "Mass Flow Rate [CC]"
cgxpressure_columnname = "Charge Exchange Pressure [torr]"

# Group by everything (for initial averaging A and B)
tracebundle_ab_groupby_columns = [offsetfreq_columnname, bxfield_columnname,
                                  byfield_columnname, pre910_columnname,
                                  massflow_columnname, repeat_columnname,
                                  frequency_columnname, tracebundle_columnname,
                                  subconfig_columnname]

# Group by everything but sub-configuration (for subtraction A_avg - B_avg)
tracebundle_groupby_columns = [offsetfreq_columnname, bxfield_columnname,
                               byfield_columnname, pre910_columnname,
                               massflow_columnname, repeat_columnname,
                               frequency_columnname, tracebundle_columnname]

# Group by everything down to repeat & frequency (for averaging A_avg - B_avg)
repeat_groupby_columns = [offsetfreq_columnname, bxfield_columnname,
                          byfield_columnname, pre910_columnname,
                          massflow_columnname, repeat_columnname,
                          frequency_columnname]

# Group into type and frequency (for averaging together repeat_data)
freq_groupby_columns = [offsetfreq_columnname, bxfield_columnname,
                        byfield_columnname, pre910_columnname,
                        massflow_columnname, frequency_columnname]

# Group into types (for plotting and saving)
# offsetfreq_columnname,
type_groupby_columns = [offsetfreq_columnname, bxfield_columnname,
                        byfield_columnname, pre910_columnname,
                        massflow_columnname]

# Columns for time, frequency averaging; monitor columns
time_columnname = "Time"
waveguidepower_columnnames = {"A" : "Waveguide A Power Reading  [V]",
                              "B" : "Waveguide B Power Reading [V]"}
faradaycup_columnanmes = {"1a" : "fc1a [uA]", "1b" : "fc1b [uA]",
                          "1c" : "fc1c [uA]", "1d" : "fc1d [uA]",
                          "2i" : "fc2i [uA]", "2ii" : "fc2ii [uA]",
                          "2iii" : "fc2iii [uA]", "2iv" : "fc2iv [uA]",
                          "3" : "fc3 [uA]", "c" : "fccentre [uA]"}
prequenchpower_columnnames = {"910" : "Pre-Quench 910 Power Detector Reading [V]",
                              "1088" : "Pre-Quench 1088 Power Detector Reading [V]",
                              "1147" : "Pre-Quench 1147 Power Detector Reading [V]"
                             }
postquenchpower_columnnames = {"910" : "Post-Quench 910 Power Detector Reading [V]",
                               "1088" : "Post-Quench 1088 Power Detector Reading [V]",
                               "1147" : "Post-Quench 1147 Power Detector Reading [V]"
                              }
preattenuation_columnnames = {"910" : "Pre-Quench 910 Attenuator Voltage Reading [V]",
                              "1088" : "Pre-Quench 1088 Attenuator Voltage Reading [V]",
                              "1147" : "Pre-Quench 1147 Attenuator Voltage Reading [V]"
                             }
postattenuation_columnnames = {"910" : "Post-Quench 910 Attenuator Voltage Reading [V]",
                               "1088" : "Post-Quench 1088 Attenuator Voltage Reading [V]",
                               "1147" : "Post-Quench 1147 Attenuator Voltage Reading [V]"
                              }
amplitude_columnnames = {"Detector H1" : "Detector Amplitude [V]",
                         "Detector H2" : "Detector Amplitude (2nd Harmonic) [V]",
                         "Combiner 1.1" : "Mixer 1 Digitizer 1 Amplitude [V]",
                         "Combiner 1.2" : "Mixer 1 Digitizer 2 Amplitude [V]",
                         "Combiner 2" : "Mixer 2 Amplitude [V]"
                        }
dc_columnnames = {"Detector H1" : "Detector DC Offset [V]",
                  "Combiner 1.1" : "Mixer 1 Digitizer 1 DC Offset [V]",
                  "Combiner 1.2" : "Mixer 1 Digitizer 2 DC Offset [V]",
                  "Combiner 2" : "Mixer 2 DC Offset [V]"
                 }
snr_columnnames = {"Detector H1" : "Detector SNR (Approx)",
                   "Detector H2" : "Detector SNR 2nd Harmonic (Approx)",
                   "Combiner 1.1" : "Mixer 1 Digitizer 1 SNR (Approx)",
                   "Combiner 1.2" : "Mixer 1 Digitizer 2 SNR (Approx)",
                   "Combiner 2" : "Mixer 2 SNR (Approx)"
                  }
otherphase_columnnames = {"Mixer Difference" : "Phase Difference (Mixer 2 - Mixer 1) [rad]",
                          "Digitizer Difference" : "Digitizer Phase Difference (Digi 2 - Digi 1) [rad]"
                         }

bytime_columns = list(prequenchpower_columnnames.values()) + \
                 list(postquenchpower_columnnames.values()) + list(preattenuation_columnnames.values()) + \
                 list(postattenuation_columnnames.values()) + [otherphase_columnnames["Digitizer Difference"]]

bytimefreq_columns = list(waveguidepower_columnnames.values()) + list(amplitude_columnnames.values()) + \
                     list(dc_columnnames.values()) + list(snr_columnnames.values())

bytimefreqab_columns = [otherphase_columnnames["Mixer Difference"]]

def mod_wrap(x, n = 0, dsize = 2*np.pi):
    '''
    Wraps values in an array x that are constricted to a domain of size
    dsize to lie within [n*dsize/2,(n+2)*dsize/2). For instance, it will wrap
    phase values (constricted to a 2*pi domain) to lie within [-pi,pi) if run
    as wrap(x, -1) or wrap(x).

    Input:
        x: array-like
            List of values to wrap to the specified domain.
        n: integer, Optional (default = 0)
            Lower bound of the domain modulo dsize/2.
        dsize: float, Optional
            The size of the domain to which the values x are restricted.

    Output:
        wrapped: array-like
            The array x wrapped into the domain [n*dsize/2, (n+2)*dsize/2]
    '''

    return (x-n*dsize/2) % (dsize) + n*dsize/2

def mod_average(x, w = None, n = 0, dsize = 2 * np.pi):
    '''
    Correctly calculates the average of an array of values constricted to a
    domain of finite size. Note that this will not work for a normal
    distribution with a standard deviation of larger than ~dsize/4.

    Input:
        x: array-like
            Array of values of which the standard deviation should be
            calculated.
        w: array-like, Optional
            Array of weights to apply to the average.
        n: integer, Optional (default = 0)
            Lower bound of the domain in which to contain the average modulo
            dsize/2.
        dsize: float, Optional
            The size of the domain to which the values x are restricted.

    Output:
        avg: Average of the values in x, weighted by the values in w.
    '''

    x = np.array(x)

    # Shift to be centered about dsize/2 and wrap to the range [0,dsize)
    x_shift = x[0] - dsize/2
    x = mod_wrap(np.array(x) - x_shift, 0, dsize = dsize)
    avg = np.average(x, weights = w)

    # Take the average and shift back, keeping the average in the range
    # [0,dsize)
    return mod_wrap(avg + x_shift, 0, dsize = dsize)

def mod_std(x, w = None, dsize = 2 * np.pi):
    '''
    Correctly calculates the standard deviation of an array of values
    constricted to a domain of finite size. Note that this will not work for a
    normal distribution with a standard deviation of larger than ~dsize/4.

    Input:
        x: array-like
            array of values of which the standard deviation should be
            calculated.
        w: array-like, Optional
            array of weights to apply to the standard deviation. Weighted
            standard deviation is calculated as outlined in J. R. Taylor, Error
            Analysis.
        dsize: float, Optional
            The size of the domain to which the values x are restricted.

    Output:
        std: float
            The standard deviation of the values in x, weighted by the values in
            w.
    '''

    x = np.array(x)

    # Shift to be centered about dsize/2 and wrap to the range [0,dsize)
    x_shift = x[0] - dsize/2
    x = mod_wrap(np.array(x) - x_shift, 0, dsize = dsize)

    # Take the average and shift back, keeping the average in the range
    # [0,dsize)
    return std(x, w)

def std(x, w = None):
    '''
    Calculates the standard deviation of values in a normal distribution.
    Formula for the weighted standard deviation found in Introduction to Error
    Analysis by J. R. Taylor.

    Input:
        x: array-like
            Array of values of which to take the standard deviation.
        w: array-like, Optional
            Array of values by which to weight the standard deviation. Must be
            the same length as x.

    Output:
        std: float
            The standard deviation of the values in x. If w is given, the
            weighted standard deviation is returned.
    '''

    if isinstance(w, type(None)):
        stdv = np.std(x, ddof = 1)
    else:
        try:
            assert(len(x)==len(w))
        except AssertionError:
            print("AssertionError:\tx and w must be the same length.")
            return
        stdv = 1./np.sqrt(np.sum(w))

    return stdv

def unwrap_fosof_line(phase_data, freq_data, discont = np.pi):
    '''
    Given an array of phases and an array of frequencies (of equal length), this
    function returns a list of phases that is not constrained to a 2*pi interval
    but forms a continuous line. Essentially, this is the same as numpy.unwrap
    but accounts for sorting our phase data by frequency.

    Input:
        phase_data: array-like
            List of phases constrained to some 2*pi interval.
        freq_data: array-like
            List of frequencies used.
        discont: float, Optional
            The maximum absolute discontinuity that will not be unwrapped.
    Output:
        unwrapped: array-like
            A 2d numpy array of [freq, phase] pairs where the phases
            are unwrapped.
    '''

    # "Unwraps" the data by changing discontinuities larger than discont to
    # their 2*pi compliment
    data = dict(zip(freq_data, phase_data))
    central_value = data[sorted(data.keys())[(len(data.keys())+1)//2]]
    for freq, phase in data.iteritems():
        if abs(phase - central_value) > discont:
            n = (central_value - phase) / (2 * np.pi)
            n = int(round(n,0))
            data[freq] = phase + 2. * np.pi * n
    unwrapped = np.array(sorted(data.items()))

    # plt.plot(unwrapped[:,0],unwrapped[:,1],'r.')
    # plt.show()

    return unwrapped

def average_and_std(x, w = None, n = 0, phases = True):
    '''
    Calculates an average and standard deviation for a list of values.

    Input:
        x: array-like
            List of values to average
        w, Optional: array-like
            List of weights to apply to the values in x. Must be the same size
            as x.
        phases, Optional: bool
            If True, the values will be assumed to lie in a domain of size 2*pi.
            The values will be shifted and the average calculated as in the
            function mod_average above.
        n: integer, Optional (default = 0)
            Lower bound of the domain in which to contain the average modulo
            dsize/2. Only used if phases = True.

    Output:
        avg: float
            Average of values in x, weighted by values in w (if given).
        std: float
            Standard deviation of the values in x, weighted by values in w (if
            given).

    '''

    x = np.array(x)

    if not isinstance(w, type(None)):
        try:
            assert(len(x) == len(w))
        except AssertionError:
            print("AssertionError:\tx and w must be the same length.")
            return
        except TypeError:
            print("TypeError:\tw must be an array of floats the same length " \
                  "as x, or None.")
            return

    if phases:
        avg = mod_average(x, w, n)
        stdv = mod_std(x, w = w)
    else:
        avg = np.average(x, weights = w)
        stdv = std(x, w = w)

    return avg, stdv

def polyfit(x, y, o, w = None, find_x0 = False):
    '''
    Fits a polynomial of order o to the data points described by x and y using
    least squares. If given weights w, will perform a weighted least squares
    fit to the data. The length of the arrays x, y and w (if given) must be
    equal.

    Input:
        x: array-like
            Data for the independent variable.
        y: array-like
            Data for the dependent variable. Must be the same length as x.
        o: integer
            Order of the polynomial to fit.
        w: array-like, Optional
            Array of weights for the data points. If using the error for a
            normal distribution, give w = 1/error**2.

    Returns:
        poly: array-like
            Array of polynomial coefficients from highest to lowest order.
        chi2: float
            The reduced chi2 statistic, only present if len(x) > (o + 1)
        chi2_prob: float
            The probability of obtaining the chi2 value for a fit with
            (len(x) - (o + 1)) degrees of freedom, only present if
            len(x) > (o + 1)
        err: array-like
            Errors in the polynomial coefficients from the covariance matrix
            estimate, only present if len(x) > (o + 1).
    '''

    # Check inputs for errors
    order = int(o)
    assert(len(x) == len(y))
    assert(len(x) >= order+1)

    x = np.array(x)
    y = np.array(y)
    if not isinstance(w, type(None)):
        w = np.array(w)
        assert(len(x) == len(w))
        W = np.eye(len(x))*w
    else:
        W = np.eye(len(x))

    # Set up matrices for least squares estimation
    W12 = np.sqrt(W)
    Xp = np.matrix(np.power(np.array([x]),
                            np.array([np.arange(order,-1,-1)]).transpose()))
    X = W12*Xp.T
    Xp = X.T
    Y = W12*np.matrix(y).T

    # Least squares estimation
    b = (Xp*X).I*Xp*Y

    # Calculate error & chi2 statistics if there are enough data points
    if len(x) > order+1:
        residsq = np.array((Y - X*b).T)[0]**2
        rsum = np.sum(residsq)
        dof = len(x) - (order+1)
        chi2 = rsum/dof
        chi2dist = stats.chi2(dof)
        chi2prob = chi2dist.sf(rsum)
        if find_x0 == True:
            x0 = -np.array(b.T)[0][1]/np.array(b.T)[0][0]
            error_x0 = dx0(x, y, w = w, chi2 = chi2)
            return np.array(b.T)[0], chi2, chi2prob, np.sqrt(np.diag((Xp*X).I*chi2)), x0, error_x0
        else:
            return np.array(b.T)[0], chi2, chi2prob, np.sqrt(np.diag((Xp*X).I*chi2))
    else:
        print('Order of polynomial is equal to number of points. No error ' \
              'estimate or chi^2 calculation possible.')
        if find_x0 == True:
            x0 = -np.array(b.T)[0][1]/np.array(b.T)[0][0]
            return np.array(b.T)[0], x0
        else:
            return np.array(b.T)[0]

def dx0(x, y, w = None, chi2 = 1):
    '''
    Calculates error in the zero crossing of a linear least squares fit.
    Accounts for possible correlation between m and b (if y = m*x + b). This
    formula was derived from error propagating the least squares summation
    for -b/m with respect to y (see J. R. Taylor, p 201 Q 8.9 for related
    formulas).

    Input:
        x: array-like
            Samples of the independent variable.
        y: array-like
            Samples of the dependent variable
        w: array-like, Optional
            Relative weights of the dependent variable samples. For data with
            standard errors es, this should be 1/es**2. If not provided,
            defaults to even weights (array of 1s). Note: must have the same
            length as x.
        chi2: float, Optional
            The chi squared statistic for the least squares fit to
            the data. If provided, the error dx0 will be scaled by
            np.sqrt(chi2). Defaults to 1 (no scaling), must be greater than 0.

    Output:
        dx0: float
            The error in the zero crossing of the linear least squares fit. May
            be weighted and/or scaled by chi2 depending on the parameters
            provided.
    '''

    x = np.array(x)
    y = np.array(y)
    try:
        assert(len(x)==len(y))
    except AssertionError:
        print("AssertionError:\tx and y arrays must be the same size.")

    try:
        assert(chi2 > 0)
    except AssertionError:
        print("AssertionError:\tchi2 value must be greater than 0.")

    # Make sure the weights have values (all 1 or otherwise) and that the arrays
    # are of the same length
    if not isinstance(w, type(None)):
        w = np.array(w)
        try:
            assert(len(x) == len(w))
        except AssertionError:
            print("AssertionError:\tx and w arrays must be the same size.")
    else:
        w = np.zeros(len(x)) + 1.

    # Important values for convenience
    a = np.sum(w)
    b = np.sum(w*x)
    c = np.sum(w*y)
    d = np.sum(w*x**2)
    e = np.sum(w*x*y)

    # Error derived from error-propagating least-squares parameter estimation
    # for x0.
    dx0_squared = (b**2 - a*d)**2 * (a*e**2 - 2*b*e*c + d*c**2)/(b*c-a*e)**4

    return np.sqrt(chi2*dx0_squared)

def chi2(y, yhat, err = None, dof_minus = 1):
    '''
    Calculates the chi^2 statistic for a set of predicted values yhat with error
    err vs measured values y. The total degrees of freedom will be decreased by
    dof_minus when returning the chi^2. This function will also calculate the
    probability of obtaining this chi^2 given the number of observations and
    degrees of freedom.
    '''

    if type(err) == type(None):
        err = np.ones(len(y))

    sumsq = np.sum(((y-yhat)/err)**2)
    chi2dist = stats.chi2(len(y) - dof_minus)
    chi2prob = chi2dist.sf(sumsq)

    return sumsq / (len(y) - dof_minus), chi2prob
