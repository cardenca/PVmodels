
# Python packages
# -----------------

import numpy as np
from scipy.optimize import least_squares
from scipy.special import lambertw

from lambertW import lw_roberts

# Custom functions
# -----------------

def photovoltaic_current(Volt: list, Iph:float ,Io:float, A:float) -> list:
    """
    Calculate the photovoltaic current for given parameters.

    Parameters
    ----------
    Volt : list
        List of photovoltaic voltages (V).
    Iph : float
        Photo-generated current (A).
    Io : float
        Saturation current (A).
    A : float
        Equivalent factor of the diode (V).

    Returns
    -------
    list
        Calculated photovoltaic current (A).
    """
    return Iph-Io*(np.exp(Volt/A)-1)

def photovoltaic_voltage(ipv,iph,io,a):
    return a*np.log((iph+io-ipv)/io)

def _maximum_power_equation_system(x,iph,io,a):
    vmp,imp = x
    t4 = np.exp(vmp / a)
    A0 = iph - (t4 - 1) * io - imp
    A1 = io * t4 * vmp - imp * a
    return A0, A1

def _maximum_power_equation_jacobian(x,iph,io,a):
    vmp,imp = x
    t2 = 0.1e1 / a
    t5 = np.exp(t2 * vmp)
    A0 = -io * t5 * t2
    A1 = -1
    A2 = io * t2 * (a + vmp) * t5
    A3 = -a
    return np.array([[A0,A1],[A2,A3]])

def _maximum_power_lambertW(iph,io,a):
    t3 = io + iph
    # t5 = np.e
    # t7 = lambertw(t5 * t3 / io).real
    t7 = lw_roberts(np.e+np.log(t3 / io))
    t8 = t7 - 1
    A0 = t8 * a
    A1 = 0.1e1 / t7 * t3 * t8
    return A0, A1

def maximum_power_point(iph,io,a,method='lm'):

    isc = photovoltaic_current(0,iph,io,a)
    voc = photovoltaic_voltage(0,iph,io,a)

    iph_scaled = iph/isc
    io_scaled  = io/isc
    a_scaled = a/voc

    if method == 'lm':
        # Scaled initial point
        t20  = np.exp(1/a_scaled)-1
        vmp0_scaled = (io_scaled*np.exp(1/a_scaled)-a_scaled)/io_scaled/t20
        imp0_scaled = np.exp(1/a_scaled)*(a_scaled-io_scaled)/a_scaled/t20

        sol = least_squares(
            fun=_maximum_power_equation_system,
            x0=(vmp0_scaled,imp0_scaled),
            jac=_maximum_power_equation_jacobian,
            method='lm',
            args=(iph_scaled,io_scaled,a_scaled)
        )
            
        return sol.x[0]*voc, sol.x[1]*isc, 

    if method=='lw':
        return _maximum_power_lambertW(iph,io,a)

def affine_transformations(a,isc,voc):
    return isc, isc/(np.exp(voc/a)-1)

def scaled_null_resistive_losses_model_voltage(imp):
    t2 = 1 / (imp - 1)
    t3 = t2 * imp
    t4 = np.exp(t3)
    t6 = lambertw(t4 * t3).real
    t7 = 0.1e1 / t6
    t10 = np.log(t2 * t7 * imp)
    t14 = imp ** 2
    t20 = np.log(1 / (t14 - 2 * imp + 1) * t7 * (imp * t6 - t6 - 1) * imp)
    return(0.1e1 / t20 * t10)

def _maximum_power_equation_system_nrl1(x,a):
    vmp, imp = x
    t3 = 0.1e1 / a
    t4 = np.exp(t3)
    t7 = np.exp(t3 * vmp)
    A0 = t4 * (-imp + 1) - t7 + imp
    A1 = t7 * vmp - (t4 - 1) * a * imp
    return A0, A1

def _jacobian_maximum_power_equation_system_nrl1(x,a):
    vmp, imp = x
    t2 = 0.1e1 / a
    t4 = np.exp(t2 * vmp)
    A0 = -t4 * t2
    t6 = np.exp(t2)
    A1 = 1 - t6
    A2 = t2 * (a + vmp) * t4
    A3 = A1 * a
    return np.array([[A0,A1],[A2,A3]])

def nrl_limit(a,vmp0,imp0):
            
    sol = least_squares(
        fun=_maximum_power_equation_system_nrl1,
        x0=(vmp0,imp0),
        jac=_jacobian_maximum_power_equation_system_nrl1,
        method='lm',
        args=[a],
    )
    
    return sol.x[0], sol.x[1]

