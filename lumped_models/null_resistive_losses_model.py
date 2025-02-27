
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
    Volt    : list
        List of photovoltaic voltages (V).
    Iph     : float
        Photo-generated current (A).
    Io      : float
        Saturation current (A).
    A       : float
        Equivalent factor of the diode (V).

    Return
    -------
    list
        Calculated photovoltaic current (A).
    """
    return Iph-Io*(np.exp(Volt/A)-1)

def photovoltaic_voltage(Curr:float,Iph:float,Io:float,A:float) -> list:
    """
    Calculate the photovoltaic current for given parameters.

    Parameters
    ----------
    Curr    : list
        List of photovoltaic current (A)
    Iph     : float
        Photo-generated current (A)
    Io      : float
        Saturation current (A)
    A       : float
        Equivalent factor of the diode (V)

    Return
    -------
    list
        Calculated photovoltaic voltage (V)
    """
    return A*np.log((Iph+Io-Curr)/Io)

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

def maximum_power_point(Iph: float,Io: float,A:float,method: str = 'lm') -> tuple:
    """
    Parameters
    ----------
    Iph : float
        Photogenerated current
    Io  : float
        Dark saturation current of the diode
    A   : float
        Equivalent factor of the diode
    method  : string
        Defines the method used for computing the maximum power point.
        The options can be Levenverg-Marquardt 'lm', or LambertW 'lw'.

    Return
    ---------
    Vmp : float
        Maximum power voltage
    Imp : float
        Maximum power current
    """

    Isc = photovoltaic_current(0,Iph,Io,A)
    Voc = photovoltaic_voltage(0,Iph,Io,A)

    iph, io, a = Iph/Isc, Io/Isc, A/Voc # scaled parameters

    if method == 'lm':
        # Scaled initial point
        t20  = np.exp(1/a)-1
        vmp0 = (io*np.exp(1/a)-a)/io/t20 
        imp0 = np.exp(1/a)*(a-io)/a/t20

        sol = least_squares(
            fun=_maximum_power_equation_system,
            x0=(vmp0,imp0),
            jac=_maximum_power_equation_jacobian,
            method='lm',
            args=(iph,io,a)
        )
            
        return sol.x[0]*Voc, sol.x[1]*Isc, 

    if method=='lw':
        return _maximum_power_lambertW(Iph,Io,A)

def affine_transformation(A: float, Isc: float, Voc:float) -> tuple:
    """
    Parameters
    -----------
    A   : float
        Equivalent factor of the diode
    Isc : float
        Short-circuit current
    Voc : float
        Open-circuit voltage
    
    Return
    -----------
    Iph : float
        Photogenerated current 
    Io  : float
        Dark saturation current of the diode

    """
    return Isc, Isc/(np.exp(Voc/A)-1)



