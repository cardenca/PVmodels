
# Python packages
# -----------------

import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import sys

# Custom functions
# -----------------

from lambertW import lw_roberts

# Computable limits
# -----------------

ainf = 1/np.log(10)/302


# Single diode model photovoltaic current estimation
# -----------------

def _photovoltaic_current_fsd(ipv,vpv,iph,io,a,rs,gsh):
    t5 = np.exp((rs * ipv + vpv) / a)
    return(iph - (t5 - 1) * io - gsh * vpv - (rs * gsh + 1) * ipv)

def _photovoltaic_current_dfsd(ipv,vpv,iph,io,a,rs,gsh):
    t2 = 0.1e1 / a
    t6 = np.exp(t2 * (rs * ipv + vpv))
    return(-io * rs * t2 * t6 - rs * gsh - 1)

def photovoltaic_current(
        Volt: list, Iph: float, Io: float, A: float, Rs: float, Gsh: float, method='lw-roberts'
    ) -> list:
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
    Rs : float
        Series resistance (Ohms).
    Gsh : float
        Shunt conductance (1/Ohms).
    method : str, optional
        Method to compute the photovoltaic current. Options are:
        'lm' (Levenberg-Marquardt least-squares method) or 
        'lw-roberts' (Lambert W approximation by Roberts). Default is 'lm'.

    Returns
    -------
    list
        Calculated photovoltaic current (A).
    """
    # Convert Volt to a NumPy array for faster processing
    Volt = np.asarray(Volt, dtype=float)

    if method == 'lm':
        # Levenberg-Marquardt least-squares method
        Ipv = []

        for Volt_point in Volt:
            # Initial guess for the current
            Ipv_initial_guess = (Iph + Io - Gsh * Volt_point) / (Rs * Gsh + 1)
            Ipv_alternative_guess = A * np.log((Iph + Io - Gsh * Volt_point) / (Io * np.exp(Volt_point / A)))

            # Use the smaller of the two initial guesses
            Ipv_guess = min(Ipv_initial_guess, Ipv_alternative_guess)

            # Solve using least-squares optimization
            solution = least_squares(
                fun=_photovoltaic_current_fsd,
                jac=_photovoltaic_current_dfsd,
                x0=Ipv_guess,
                args=(Volt_point, Iph, Io, A, Rs, Gsh),
                method='lm'  # Levenberg-Marquardt method
            )
            Ipv.append(solution.x[0])  # Append the solution

        return Ipv

    elif method == 'lw-roberts':
        # Lambert W approximation (Roberts' method)
        T8 = 1 / A
        T12 = 1 / (Rs * Gsh + 1)
        T18 = lw_roberts(
            T12 * T8 * ((Io + Iph) * Rs + Volt) + np.log(T12 * T8 * Io * Rs)
        )
        return (
            1 / Rs * T12 * (T18 * (-A * Gsh * Rs - A) + Rs * (-Gsh * Volt + Io + Iph))
        )

    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'lm' or 'lw-roberts'.")

# Single diode model photovoltaic voltage estimation
# -----------------
    
def _photovoltaic_voltage_fsd(vpv,ipv,iph,io,a,rs,gsh):
    t5 = np.exp((rs * ipv + vpv) / a)
    return(iph - (t5 - 1) * io - gsh * vpv - (rs * gsh + 1) * ipv)

def _photovoltaic_voltage_dfsd(vpv,ipv,iph,io,a,rs,gsh):
    t1 = 0.1e1 / a
    t6 = np.exp(t1 * (rs * ipv + vpv))
    return(-io * t1 * t6 - gsh)

def photovoltaic_voltage(
        Curr:list, Iph:float, Io:float, A:float, Rs:float, Gsh:float,method='lw-roberts'
    ):
    """
    Calculate the photovoltaic voltage for given parameters.

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
    Rs : float
        Series resistance (Ohms).
    Gsh : float
        Shunt conductance (1/Ohms).
    method : str, optional
        Method to compute the photovoltaic current. Options are:
        'lm' (Levenberg-Marquardt least-squares method) or 
        'lw-roberts' (Lambert W approximation by Roberts). Default is 'lm'.

    Returns
    -------
    list
        Calculated photovoltaic voltage (V).
    """

    Curr = np.asarray(Curr, dtype=float)

    if method=='lm':

        vpv = []
        for ipv_point in Curr:
            # Initial point estimation
            if Gsh > 0:
                vpv0 = (Iph+Io-(Rs*Gsh+1)*ipv_point)/Gsh
            else: 
                vpv0 = A*np.log(Iph/Io+1)
            vpv1 = A*np.log((Iph+Io-(Rs*Gsh+1)*ipv_point)/Io/np.exp(Rs*ipv_point/A))
            if vpv0 > vpv1:
                vpv0 = vpv1
            # Resolution of the system
            sol = least_squares(
                fun=_photovoltaic_voltage_fsd,
                jac=_photovoltaic_voltage_dfsd,
                x0=vpv0,
                args=(ipv_point,Iph,Io,A,R,Gsh),
                method='lm'
            )
            vpv.append( sol.x[0] )

        return vpv

    if method=='lw-roberts':
        t1 = 0.1e1 / A
        t3 = 0.1e1 / Gsh
        t10 = lw_roberts( t3 * t1 * (Io + Iph - Curr) + np.log(t3 * t1 * Io) )
        return(t3 * (-A * Gsh * t10 - Gsh * Curr * Rs + Io + Iph - Curr))

# Single diode model maximum power point estimation
# -----------------

def _total_derivative_current_voltage(vpv,ipv,iph,io,a,rs,gsh):
    t6 = np.exp((rs * ipv + vpv) / a)
    t8 = a * gsh
    return 0.1e1 / (io * rs * t6 + rs * t8 + a) * (-io * t6 - t8)

def _maximum_power_point_system(x,iph,io,a,rs,gsh):
    vmp,imp = x
    t2 = imp * rs
    t4 = 0.1e1 / a
    t6 = np.exp(t4 * (t2 + vmp))
    t9 = gsh * vmp
    fmp = iph - (t6 - 1) * io - t9 - (rs * gsh + 1) * imp
    dfmp = t4 * (-t6 * (t2 - vmp) * io - (gsh * imp * rs + imp - t9) * a)
    return np.array([fmp, dfmp])

def _maximum_power_point_jacobian(x,iph,io,a,rs,gsh):
    vmp,imp = x
    t2 = imp * rs
    t4 = 0.1e1 / a
    t6 = np.exp(t4 * (t2 + vmp))
    t8 = a * gsh
    A0 = t4 * (-io * t6 - t8)
    t10 = io * rs
    A1 = t4 * (-rs * t8 - t10 * t6 - a)
    t17 = a ** 2
    t20 = 0.1e1 / t17
    A2 = t20 * (t6 * (-t2 + a + vmp) * io + gsh * t17)
    A3 = t20 * (-t6 * (t2 + a - vmp) * t10 - (rs * gsh + 1) * t17)
    return np.array([[A0,A1],[A2,A3]])

def maximum_power_point(Iph: float,Io: float,A: float,Rs: float,Gsh: float,Isc: float,Voc: float) -> tuple:
    '''
    Parameters
    --------
    Iph: float
        Photogenerated current
    Io: float
        Dark saturation current of the diode
    A: float
        Equivalent factor of the diode
    Rs: float
        Series resistance
    Gsh: float
        Shunt conductance
    Isc: float
        Short-circuit current
    Voc:
        Open-circuit voltage
    
    Return
    --------
    Vmp: float
        Maximum power voltage
    Imp: float
        Maximum power current
    '''

    # estimation of the initial point based on derivatives
    msc = _total_derivative_current_voltage(0,Isc,Iph,Io,A,Rs,Gsh)
    moc = _total_derivative_current_voltage(Voc,0,Iph,Io,A,Rs,Gsh)

    vmp0 = (Voc*moc+Isc)/(moc-msc)
    imp0 = msc*vmp0+Isc

    # system resolution
    sol = least_squares(
        fun=_maximum_power_point_system,
        jac=_maximum_power_point_jacobian,
        x0=(vmp0,imp0),
        args=(Iph,Io,A,Rs,Gsh),
        method='lm'
    )

    return sol.x[0], sol.x[1] 

# Reduced representations
# ---------------

computational_limits = pd.read_csv(r'./lumped_models/computational_limits.csv')

vmp_nrl_limit, imp_nrl_limit = computational_limits['vmp_nrl'], computational_limits['imp_nrl']
vmp_nsh_limit, imp_nsh_limit = computational_limits['vmp_nsh'], computational_limits['imp_nsh']
vmp_nsr_limit, imp_nsr_limit = computational_limits['vmp_nsr'], computational_limits['imp_nsr']

vmp_inf = max( min(vmp_nsh_limit), min(vmp_nsr_limit), min(vmp_nrl_limit) ) #infimum computable voltage
vmp_sup = min( max(vmp_nsh_limit), max(vmp_nsr_limit), max(vmp_nrl_limit) ) #supremum computable voltage

# Three parameters representation
#* Don't forget that to ensure positives values of iph and io: \
#* rs \in ]0, voc/isc[
#* gsh \in ]0, isc/(voc-rs*isc) [
# -----------------

def affine_transformation_sd3(A:float,Rs:float,Gsh:float,Isc:float,Voc:float)->tuple:
    '''
    Parameters
    -----------
        A   : float
            Equivalent factor of the diode
        Rs  : float
            Series resistance
        Gsh : float
            Shunt conductance
        Isc : float
            Short-circuit current
        Voc : float
            Open-circuit voltage
        
    Return
    -----------
        Iph : float
            Photo-generated current
        Io  : float
            Dark saturation current of the diode
    '''

    t3 = Isc * Rs
    t4 = 0.1e1 / A
    t6 = np.exp(t4 * t3)
    t12 = np.exp(t4 * Voc)
    t14 = -t3 + Voc
    t18 = 0.1e1 / (-t6 + t12)
    Iph = t18 * (-t6 * Voc * Gsh + t12 * (Gsh * Rs + 1) * Isc + Gsh * t14 - Isc)
    Io = t18 * (-Gsh * t14 + Isc)

    return Iph, Io

# Three parameters representation
# -----------------

def affine_transformation_sd2(A: float,Rs: float,Isc: float,Vmp: float,Imp: float,Voc: float)->tuple:
    
    '''
    Parameters
    -----------
        A   : float
            Equivalent factor of the diode
        Rs  : float
            Series resistance
        Isc : float
            Short-circuit current
        Vmp : float
            Maximum power voltage
        Imp : float
            Maximum power current
        Voc : float
            Open-circuit voltage
        
    Return
    -----------
        Iph : float
            Photo-generated current
        Io  : float
            Dark saturation current of the diode
        Gsh : float
            Shunt conductance
    '''


    t2 = Imp * Rs
    t4 = 0.1e1 / A
    t6 = np.exp(t4 * (t2 + Vmp))
    t7 = Isc * t6
    t9 = Isc * Rs
    t11 = np.exp(t4 * t9)
    t12 = Imp * t11
    t15 = np.exp(t4 * Voc)
    t18 = -Vmp + Voc
    t20 = Imp * Voc
    t29 = 0.1e1 / (t6 * (t9 - Voc) + t11 * (-t2 - Vmp + Voc) + (t2 - t9 + Vmp) * t15)
    Iph = t29 * (Isc * t15 * Vmp + Isc * t18 + t12 * Voc - t7 * Voc - t20)
    Io = t29 * (-Isc * t18 + t20)
    Gsh = t29 * (-t7 + t12 - (Imp - Isc) * t15)

    return Iph, Io, Gsh

def _shunt_conductance_numerator_nsh_rs(rs,u,imp,vmp):
    t3 = u ** (imp * rs + vmp)
    t4 = u ** rs
    return(-t3 + imp * t4 + (-imp + 1) * u)

def _partial_shunt_conductance_numerator_dnsh_drs(rs,u,imp,vmp):
    t3 = u ** (imp * rs + vmp)
    t5 = np.log(u)
    t7 = u ** rs
    return(-imp * t3 * t5 + imp * t5 * t7)

# Maximum power point function as a function of u and rs (partial derivatives included)
# this function is expresed in terms of u=np.exp(1/a)

def _maximum_power_point_function_fmp_rs(rs,u,imp,vmp):
    t2 = imp * rs
    t5 = np.log(u)
    t9 = u ** (t2 + vmp)
    t14 = u ** rs
    return(t9 * (-t5 * (t2 - vmp) * (imp + vmp - 1) + imp - vmp) + t14 * (2 * vmp - 1) * imp - 2 * (imp - 0.1e1 / 0.2e1) * u * vmp)

def _partial_maximum_power_point_function_dfmp_drs(rs,u,imp,vmp):
    t1 = imp + vmp - 1
    t3 = np.log(u)
    t4 = imp * rs
    t6 = u ** (t4 + vmp)
    t19 = u ** rs
    return(-t6 * t3 * imp * t1 + t3 * imp * t6 * (-t3 * (t4 - vmp) * t1 + imp - vmp) + t3 * t19 * imp * (2 * vmp - 1))

def _maximum_power_point_function_zerors_fmp_u(u,imp,vmp):
    t3 = np.log(u)
    t6 = u ** vmp
    return(t6 * (t3 * vmp * (imp + vmp - 1) + imp - vmp) + ((-2 * u + 2) * imp + u) * vmp - imp)

def _partial_maximum_power_point_function_zerors_dfmp_du(u,imp,vmp):
    t3 = np.log(u)
    t5 = 2 * imp
    t8 = u ** (vmp - 1)
    return(vmp * (t8 * (t3 * vmp * (imp + vmp - 1) + t5 - 1) - t5 + 1))

def _equation_system_sd2_region(x,imp,vmp):
    u, rs = x
    t2 = imp * rs
    t4 = u ** (t2 + vmp)
    t5 = u ** rs
    A0 = -t4 + imp * t5 + (-imp + 1) * u
    t12 = np.log(u)
    A1 = t4 * (-t12 * (t2 - vmp) * (imp + vmp - 1) + imp - vmp) + t5 * (2 * vmp - 1) * imp - 2 * u * vmp * (imp - 0.1e1 / 0.2e1)
    return A0, A1

def _jacobian_equation_system_sd2_region(x,imp,vmp):
    u, rs = x
    t2 = imp * rs
    t3 = -t2 - vmp
    t5 = u ** (t2 + vmp - 1)
    t7 = rs - 1
    t8 = u ** t7
    A0 = imp * rs * t8 + t3 * t5 - imp + 1
    t11 = np.log(u)
    t12 = imp * t11
    t13 = u ** rs
    t14 = -t3
    t15 = u ** t14
    A1 = (t13 - t15) * t12
    t19 = (t2 - vmp) * (imp + vmp - 1)
    A2 = t5 * (-t11 * t14 * t19 + imp * (-2 * t7 * vmp + rs) - vmp) + 2 * t8 * rs * imp * (vmp - 0.1e1 / 0.2e1) - 2 * vmp * imp + vmp
    t37 = 2 * vmp
    A3 = -(t15 * (t11 * t19 + t37 - 1) + t13 * (-t37 + 1)) * t12
    return np.array([[A0,A1],[A2,A3]])

# Pre-loaded data regarding the computable limits
# The file used to compute the limits is under the name of "test/computational_limits.py"

def _interpolation_imp(vmp,vmp_region,imp_region):
    ndat = sum(vmp_region<vmp)
    v1, v2 = vmp_region[ndat-1], vmp_region[ndat]
    i1, i2 = imp_region[ndat-1], imp_region[ndat]
    mregion = (i2-i1)/(v2-v1)
    nregion = i1-mregion*v1
    return mregion*vmp+nregion

# Main function to compute the boundaries of the sdm given imp, vmp

def calculate_domain_sd1(Isc:float, Vmp: float, Imp: float, Voc: float) -> tuple:
    '''
    Parameters
    ------
    Isc : float
        Short-circuit current
    Vmp : float
        Maximum power voltage
    Imp : float
        Maximum power current
    Voc : float, optional
        Open-circuit voltage

    Returns
    ------
    Amax : float
        Maximum limit for the equivalent factor of the diode
    Rsmin : float
        Minimum limit for the series resistance 
    region : str
        Indicates the region of the data. The possibles values are 'nsr region' or 'nsh region'.
    '''

    # scaling values
    vmp, imp = Vmp/Voc, Imp/Isc

    # Checking the computable voltages
    if (vmp<=vmp_inf) + (vmp>=vmp_sup):
        raise 'Error: voltage point outside computable limits (Vmp>Voc or Vmp<Voc/2)'

    # Checking the computable currents
    imp_nsh = _interpolation_imp(vmp,vmp_nsh_limit,imp_nsh_limit)
    imp_nsr = _interpolation_imp(vmp,vmp_nsr_limit,imp_nsr_limit)

    if (imp_nsh<imp) + (imp<imp_nsr):
        raise 'Error: current point outside computable limits (Imp>Isc or Imp<Isc/2)'
    
    # Classifying the region
    imp_nrl = _interpolation_imp(vmp,vmp_nrl_limit,imp_nrl_limit)

    try:
    
        if imp<=imp_nrl:
            region = 'nsr region'

            # TODO: Look for a good initial point
            # I realized that a good point is that one such that fmp(rs=0) < 0

            a0 = (vmp-1)/np.log(1-imp)
            u0 = np.exp(1/a0)
            f0 = _maximum_power_point_function_zerors_fmp_u(u0,imp,vmp)

            while f0>0:
                a0 = (ainf+a0)/2
                u0 = np.exp(1/a0)
                f0 = _maximum_power_point_function_zerors_fmp_u(u0,imp,vmp)

            #* Equation solution

            umin = least_squares( 
                fun = _maximum_power_point_function_zerors_fmp_u, 
                x0 = u0, 
                jac = _partial_maximum_power_point_function_zerors_dfmp_du, 
                args = (imp,vmp), 
                method = 'lm'
                ).x[0] 
                
            amax = 1/np.log(umin)
            rsmin = 0

            return Voc*amax, Voc*rsmin/Isc, region
        
        else:
            region = 'nsh region'

            rsmax = (1-vmp)/imp
            amax0 = (vmp-1)/np.log(1-imp)

            # TODO: Look for a good initial point
            # I realized that a good point is that one such that Rs_sh>Rs_mp
            # I'm sure there is a good mathmetical explanation for that
            # The issue with this solution is that is making the whole solution slower

            a0 = (ainf+amax0)/2
            u0 = np.exp(1/a0)
            r0 = rsmax*(1-a0/amax0)

            rsh0 = least_squares( fun = _shunt_conductance_numerator_nsh_rs, 
                                    x0 = r0, 
                                    jac = _partial_shunt_conductance_numerator_dnsh_drs, 
                                    args = (u0,imp,vmp), 
                                    method = 'lm' ).x[0]

            rmp0 = least_squares( fun = _maximum_power_point_function_fmp_rs, 
                                    x0 = r0, 
                                    jac = _partial_maximum_power_point_function_dfmp_drs, 
                                    args = (u0,imp,vmp), 
                                    method = 'lm' ).x[0]

            while rmp0 > rsh0:

                a0  = (ainf+a0)/2
                r0 = rsmax*(1-a0/amax0)
                u0 = np.exp(1/a0)

                rsh0 = least_squares( fun = _shunt_conductance_numerator_nsh_rs, 
                                        x0 = r0, 
                                        jac = _partial_shunt_conductance_numerator_dnsh_drs, 
                                        args = (u0,imp,vmp), 
                                        method = 'lm' ).x[0]

                rmp0 = least_squares( fun = _maximum_power_point_function_fmp_rs, 
                                        x0 = r0, 
                                        jac = _partial_maximum_power_point_function_dfmp_drs, 
                                        args = (u0,imp,vmp), 
                                        method = 'lm' ).x[0]
                
            r0 = rsh0

            sol = least_squares( 
                fun = _equation_system_sd2_region, 
                x0 = (u0,r0), 
                jac = _jacobian_equation_system_sd2_region, 
                args = (imp,vmp), 
                method = 'lm' )
            
            amax  = 1/np.log(sol.x[0])
            rsmin = sol.x[1]

            return Voc*amax, Voc*rsmin/Isc, region
        
    except:
        raise 'Error: non computable value, possible problem with the initial point estimation, '+region

def calculate_parameters_sd1(A: float, Isc:float, Vmp:float, Imp: float, Voc: float) -> tuple:
    """
    Parameters
    --------
    A   : float
        Equivalent factor of the diode
    Isc : float 
        Short-circuit curent
    Vmp : float
        Maximum power voltage
    Imp : float
        Maximum power current
    Voc : float
        Open circuit voltage
    
    Return
    --------
    Iph : float
        Photogenerated current
    Io  : float
        Dark saturation current of the diode
    Rs  : float
        Series resistance
    Gsh : float
        Shunt conductance
    """

    vmp, imp = Vmp/Voc, Imp/Isc
    a = A/Voc

    amp, rmp, region = calculate_domain_sd1(1, vmp, imp, 1)
    rmax = (1-vmp)/imp

    u = np.exp(1/a)
    r0 = (rmp-rmax)*a/amp+rmax

    rs = least_squares(
            fun=_maximum_power_point_function_fmp_rs,
            jac=_partial_maximum_power_point_function_dfmp_drs,
            x0=r0,
            args=(u,imp,vmp),
            method='lm').x[0]
    
    iph, io, gsh = affine_transformation_sd2(a,rs,1,vmp,imp,1)

    return Isc*iph,Isc*io,Voc*rs/Isc,Isc*gsh/Voc
