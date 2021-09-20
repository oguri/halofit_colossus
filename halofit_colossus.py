#!/usr/bin/env python
import numpy as np 
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

from colossus.cosmology import cosmology
from astropy import constants as const
from astropy import units as u

def ks_func(k, z, cosmo):
    return cosmo.sigma(1.0 / k, z, filt = 'gaussian') - 1.0

def ks_calc(z, cosmo):
    z = scipy.optimize.root_scalar(ks_func, args = (z, cosmo), method = 'brentq', bracket = (1.0e-2, 1.0e5))

    return z.root
    
#
# revised halofit (Takahashi+2012)
# delta2k_halofit = (k^3 / 2pi^2) * pk_halofit, k in units of Mpc/h
# 
def delta2k_halofit(k, z, cosmo):
    ks = ks_calc(z, cosmo)
    neff = (-2.0) * cosmo.sigma(1.0 / ks, z, filt = 'gaussian', derivative = True) - 3.0
    #
    # use spline
    #
    #lnr = np.linspace(np.log(0.3 / ks), np.log(3.0 / ks), 100)
    #lns = cosmo.sigma(np.exp(lnr), z, filt = 'gaussian', derivative = True)
    #sig_spline = _spline(lnr, lns, k = 5)
    #dsdr2 = sig_spline.derivatives(np.log(1.0 / ks))[1]
    #
    # get derivative from central difference
    #
    h = 1.0e-4
    sp = cosmo.sigma((1.0 / ks) * (1.0 + h), z, filt = 'gaussian', derivative = True)
    sm = cosmo.sigma((1.0 / ks) * (1.0 - h), z, filt = 'gaussian', derivative = True)
    dsdr2 = (sp - sm) / (np.log((1.0 / ks) * (1.0 + h)) - np.log((1.0 / ks) * (1.0 - h)))
    #
    C = (-2.0) * dsdr2 
    
    om = cosmo.Om(z)
    ow = cosmo.Ode(z)
    w  = cosmo.wz(z)
         
    an = 10.0 ** (1.5222 + 2.8553 * neff + 2.3706 * neff * neff + 0.9903 * neff * neff * neff + 0.2250  * neff * neff * neff * neff - 0.6038 * C + 0.1749 * ow * (1.0 + w))
    bn = 10.0 ** (-0.5642 + 0.5864 * neff + 0.5716 * neff * neff - 1.5474 * C + 0.2279 * ow * (1.0 + w))
    cn = 10.0 ** (0.3698 + 2.0404 * neff + 0.8161 * neff * neff + 0.5869 * C)
    gamman = 0.1971 - 0.0843 * neff + 0.8460 * C
    alphan = np.abs(6.0835 + 1.3373 * neff - 0.1959 * neff * neff - 5.5274 * C)
    betan  = 2.0379 - 0.7354 * neff + 0.3157 * neff * neff + 1.2490 * neff * neff * neff + 0.3980 * neff * neff * neff * neff - 0.1682 * C
    mun    = 0.
    nun    = 10.0 ** (5.2105 + 3.6902 * neff)
    f1 = om ** (-0.0307)
    f2 = om ** (-0.0585)
    f3 = om ** 0.0743

    y  = k / ks
    fy = (y / 4.0) + (y * y / 8.0)

    # colossus use the Eisenstein & Hu transfer function by default
    pk_lin = cosmo.matterPowerSpectrum(k, z)
    deltal = k * k * k * pk_lin / (2.0 * np.pi * np.pi)
    
    deltaq = deltal * (((1.0 + deltal) ** betan) / (1.0 + alphan * deltal)) * np.exp((-1.0) * fy)
    deltad = (an * y ** (3.* f1)) / (1.0 + bn * (y ** f2) + (cn * f3 * y) ** (3.0 - gamman)) 
    deltah = deltad / (1.0 + (mun / y) + (nun / (y * y)))
    
    return deltaq + deltah

def pk_halofit(k, z, cosmo):
    return 2.0 * np.pi * np.pi * delta2k_halofit(k, z, cosmo) / (k * k * k)

#
# show an example
#
if __name__ == '__main__':

    #cosmo = cosmology.setCosmology('planck15')
    my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.81, 'ns': 0.96}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)

    # linear and nonlinear matter power spectrum at z = 0
    k_bin = 10 ** np.arange(-5.0, 2.0, 0.1)
    Pk_lin = cosmo.matterPowerSpectrum(k_bin)
    Pk_nl  = pk_halofit(k_bin, 0.0, cosmo)

    print('# k[Mpc/h] P_lin(k) P_NL(k)')
    for i in range(len(k_bin)):
        print('%e %e %e' % (k_bin[i], Pk_lin[i], Pk_nl[i]))

