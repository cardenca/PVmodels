
# Python packages
# -----------------

import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import sys

# Custom functions
# -----------------

from package.src.math_support.lambertW import lw_roberts

# Computable limits input
# -----------------

# from lumped_models.data.computational_constants import ainf as ainf_system
float_supremum_python = sys.float_info.max
ainf_theo  = 1/np.log(float_supremum_python)
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
        Volt: list, Iph: float, Io: float, A: float, Rs: float, Gsh: float, method='lm'
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
        Calculated photovoltaic currents (A).
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
        ipv:list, iph:float, io:float, a:float, rs:float, gsh:float,method='lm'
    ):

    ipv = np.asarray(ipv, dtype=float)

    if method=='lm':

        vpv = []
        for ipv_point in ipv:
            # Initial point estimation
            if gsh > 0:
                vpv0 = (iph+io-(rs*gsh+1)*ipv_point)/gsh
            else: 
                vpv0 = a*np.log(iph/io+1)
            vpv1 = a*np.log((iph+io-(rs*gsh+1)*ipv_point)/io/np.exp(rs*ipv_point/a))
            if vpv0 > vpv1:
                vpv0 = vpv1
            # Resolution of the system
            sol = least_squares(
                fun=_photovoltaic_voltage_fsd,
                jac=_photovoltaic_voltage_dfsd,
                x0=vpv0,
                args=(ipv_point,iph,io,a,rs,gsh),
                method='lm'
            )
            vpv.append( sol.x[0] )

        return vpv

    if method=='lw-roberts':
        t1 = 0.1e1 / a
        t3 = 0.1e1 / gsh
        t10 = lw_roberts( t3 * t1 * (io + iph - ipv) + np.log(t3 * t1 * io) )
        return(t3 * (-a * gsh * t10 - gsh * ipv * rs + io + iph - ipv))

# Single diode model maximum power point estimation
# -----------------

def total_derivative_current_voltage_sdm(vpv,ipv,iph,io,a,rs,gsh):
    t6 = np.exp((rs * ipv + vpv) / a)
    t8 = a * gsh
    return 0.1e1 / (io * rs * t6 + rs * t8 + a) * (-io * t6 - t8)

def maximum_power_point_system_sdm(x,iph,io,a,rs,gsh):
    vmp,imp = x
    t2 = imp * rs
    t4 = 0.1e1 / a
    t6 = np.exp(t4 * (t2 + vmp))
    t9 = gsh * vmp
    fmp = iph - (t6 - 1) * io - t9 - (rs * gsh + 1) * imp
    dfmp = t4 * (-t6 * (t2 - vmp) * io - (gsh * imp * rs + imp - t9) * a)
    return np.array([fmp, dfmp])

def maximum_power_point_jacobian_sdm(x,iph,io,a,rs,gsh):
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

def maximum_power_point_sdm(iph: float,io: float,a: float,rs: float,gsh: float,isc: float,voc: float) -> tuple:
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
    msc = total_derivative_current_voltage_sdm(0,isc,iph,io,a,rs,gsh)
    moc = total_derivative_current_voltage_sdm(voc,0,iph,io,a,rs,gsh)

    vmp0 = (voc*moc+isc)/(moc-msc)
    imp0 = msc*vmp0+isc

    # system resolution
    sol = least_squares(
        fun=maximum_power_point_system_sdm,
        jac=maximum_power_point_jacobian_sdm,
        x0=(vmp0,imp0),
        args=(iph,io,a,rs,gsh),
        method='lm'
    )

    return sol.x[0], sol.x[1] 

# Reduced representations
# ---------------

computational_limits = pd.read_pickle(r'./package/data/computational_limits.pkl')
vmp_nrl_limit, imp_nrl_limit = computational_limits['vmp_nrl'], computational_limits['imp_nrl']
vmp_nsh_limit, imp_nsh_limit = computational_limits['vmp_nsh'], computational_limits['imp_nsh']
vmp_nsr_limit, imp_nsr_limit = computational_limits['vmp_nsr'], computational_limits['imp_nsr']

vmp_inf = max( min(vmp_nsh_limit), min(vmp_nsr_limit), min(vmp_nrl_limit) )
vmp_sup = min( max(vmp_nsh_limit), max(vmp_nsr_limit), max(vmp_nrl_limit) )

# Three parameters representation
#* Don't forget that to ensure positives values of iph and io: \
#* rs \in ]0, voc/isc[
#* gsh \in ]0, isc/(voc-rs*isc) [
# -----------------

def affine_transformation_sd3(a,rs,gsh,isc,voc):
    '''
    Returns:
        iph: dark saturatino current of the diode
        io: 
    '''
    t3 = isc * rs
    t4 = 0.1e1 / a
    t6 = np.exp(t4 * t3)
    t12 = np.exp(t4 * voc)
    t14 = -t3 + voc
    t18 = 0.1e1 / (-t6 + t12)
    iph = t18 * (-t6 * voc * gsh + t12 * (gsh * rs + 1) * isc + gsh * t14 - isc)
    io = t18 * (-gsh * t14 + isc)
    return iph, io

# Three parameters representation
# -----------------

def affine_transformation_sd2(a,rs,isc,voc,imp,vmp):
    '''
    Inputs:
        a:
        rs:
        isc:
        voc:
        imp:
        vmp
    Outputs:
        iph:
        io:
        gsh
    '''
    t2 = imp * rs
    t4 = 0.1e1 / a
    t6 = np.exp(t4 * (t2 + vmp))
    t7 = isc * t6
    t9 = isc * rs
    t11 = np.exp(t4 * t9)
    t12 = imp * t11
    t15 = np.exp(t4 * voc)
    t18 = -vmp + voc
    t20 = imp * voc
    t29 = 0.1e1 / (t6 * (t9 - voc) + t11 * (-t2 - vmp + voc) + (t2 - t9 + vmp) * t15)
    iph = t29 * (isc * t15 * vmp + isc * t18 + t12 * voc - t7 * voc - t20)
    io = t29 * (-isc * t18 + t20)
    gsh = t29 * (-t7 + t12 - (imp - isc) * t15)
    return iph, io, gsh

# nsh as a function of u and rs (partial derivatives included)
# this function is expresed in terms of u=np.exp(1/a)

def shunt_conductance_numerator_nsh_a(a,imp,vmp):
    t1 = 0.1e1 / a
    t3 = np.exp(t1 * vmp)
    t4 = np.exp(t1)
    return(-t3 + imp - (imp - 1) * t4)

def partial_shunt_conductance_numerator_dnsh_da(a,imp,vmp):
    t1 = 0.1e1 / a
    t3 = np.exp(t1 * vmp)
    t5 = np.exp(t1)
    t9 = a ** 2
    return(0.1e1 / t9 * (t3 * vmp + (imp - 1) * t5))

def shunt_conductance_numerator_nsh_rs(rs,u,imp,vmp):
    t3 = u ** (imp * rs + vmp)
    t4 = u ** rs
    return(-t3 + imp * t4 + (-imp + 1) * u)

def partial_shunt_conductance_numerator_dnsh_drs(rs,u,imp,vmp):
    t3 = u ** (imp * rs + vmp)
    t5 = np.log(u)
    t7 = u ** rs
    return(-imp * t3 * t5 + imp * t5 * t7)

def maximum_equivalent_factor_diode_ash_max(imp,vmp):
    # Initial point
    amax0 = (vmp-1)/np.log(1-imp)
    sol = least_squares(
        fun=shunt_conductance_numerator_nsh_a,
        jac=partial_shunt_conductance_numerator_dnsh_da,
        x0=amax0,
        args=(imp,vmp),
        method='lm'
    )
    return sol.x[0]

# Maximum power point function as a function of u and rs (partial derivatives included)
# this function is expresed in terms of u=np.exp(1/a)

def maximum_power_point_function_fmp_rs(rs,u,imp,vmp):
    t2 = imp * rs
    t5 = np.log(u)
    t9 = u ** (t2 + vmp)
    t14 = u ** rs
    return(t9 * (-t5 * (t2 - vmp) * (imp + vmp - 1) + imp - vmp) + t14 * (2 * vmp - 1) * imp - 2 * (imp - 0.1e1 / 0.2e1) * u * vmp)

def partial_maximum_power_point_function_dfmp_drs(rs,u,imp,vmp):
    t1 = imp + vmp - 1
    t3 = np.log(u)
    t4 = imp * rs
    t6 = u ** (t4 + vmp)
    t19 = u ** rs
    return(-t6 * t3 * imp * t1 + t3 * imp * t6 * (-t3 * (t4 - vmp) * t1 + imp - vmp) + t3 * t19 * imp * (2 * vmp - 1))

def maximum_power_point_function_zerors_fmp_u(u,imp,vmp):
    t3 = np.log(u)
    t6 = u ** vmp
    return(t6 * (t3 * vmp * (imp + vmp - 1) + imp - vmp) + ((-2 * u + 2) * imp + u) * vmp - imp)

def partial_maximum_power_point_function_zerors_dfmp_du(u,imp,vmp):
    t3 = np.log(u)
    t5 = 2 * imp
    t8 = u ** (vmp - 1)
    return(vmp * (t8 * (t3 * vmp * (imp + vmp - 1) + t5 - 1) - t5 + 1))

def equation_system_sd2_region(x,imp,vmp):
    u, rs = x
    t2 = imp * rs
    t4 = u ** (t2 + vmp)
    t5 = u ** rs
    A0 = -t4 + imp * t5 + (-imp + 1) * u
    t12 = np.log(u)
    A1 = t4 * (-t12 * (t2 - vmp) * (imp + vmp - 1) + imp - vmp) + t5 * (2 * vmp - 1) * imp - 2 * u * vmp * (imp - 0.1e1 / 0.2e1)
    return A0, A1

def jacobian_equation_system_sd2_region(x,imp,vmp):
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

def interpolation_imp(vmp,vmp_region,imp_region):
    ndat = sum(vmp_region<vmp)
    v1, v2 = vmp_region[ndat-1], vmp_region[ndat]
    i1, i2 = imp_region[ndat-1], imp_region[ndat]
    mregion = (i2-i1)/(v2-v1)
    nregion = i1-mregion*v1
    return mregion*vmp+nregion

# Main function to compute the boundaries of the sdm given imp, vmp

def one_dimensional_sdm_domain(imp: float,vmp: float, ainf: float = ainf) -> tuple:
    '''
    Parameters
    ------
    imp : float
        Scaled maximum power current
    vmp : float
        Scaled maximum power voltage
    ainf : float, optional
        Minimim (infimum) float supported by python such that ``exp(1/a)`` is finite.

    Returns
    ------
    amax : float
        Maximum limit for the scaled equivalent factor of the diode
    rsmin : float
        Minimum limit for the scaled series resistance 
    region : str
        Indicates the region of the data. The possibles values are 'nsr region', 'nsh region', or 'nan'.
    '''

    # Checking the computable voltages
    if (vmp<=vmp_inf) + (vmp>=vmp_sup):
        # print('region nan')
        return np.nan, np.nan, np.nan

    # Checking the computable currents
    imp_nsh = interpolation_imp(vmp,vmp_nsh_limit,imp_nsh_limit)
    imp_nsr = interpolation_imp(vmp,vmp_nsr_limit,imp_nsr_limit)

    if (imp_nsh<imp) + (imp<imp_nsr):
        # print('region nan')
        return np.nan, np.nan, np.nan
    
    # Classifying the region
    imp_nrl = interpolation_imp(vmp,vmp_nrl_limit,imp_nrl_limit)
    # ainf = 1/np.log(10)/302

    try:
    
        if imp<=imp_nrl:
            region = 'nsr region'

            # Looking for a good initial point
            # I realized that a good point is that one such that fmp(rs=0) < 0

            a0 = (vmp-1)/np.log(1-imp)
            u0 = np.exp(1/a0)
            f0 = maximum_power_point_function_zerors_fmp_u(u0,imp,vmp)

            while f0>0:
                a0 = (ainf+a0)/2
                u0 = np.exp(1/a0)
                f0 = maximum_power_point_function_zerors_fmp_u(u0,imp,vmp)

            #* Equation solution

            umin = least_squares( 
                fun = maximum_power_point_function_zerors_fmp_u, 
                x0 = u0, 
                jac = partial_maximum_power_point_function_zerors_dfmp_du, 
                args = (imp,vmp), 
                method = 'lm'
                ).x[0] 
                
            amax = 1/np.log(umin)
            rsmin = 0

            return amax, rsmin, region
        
        else:
            region = 'nsh region'

            # function rsmax*(1-a/amax0)
            rsmax = (1-vmp)/imp
            amax0 = (vmp-1)/np.log(1-imp)

            # Looking for a good initial point
            # I realized that a good point is that one such that Rs_sh>Rs_mp
            # I'm sure there is a good mathmetical explanation for that
            # The issue with this solution is that is making the whole solution slower

            a0 = (ainf+amax0)/2
            u0 = np.exp(1/a0)
            r0 = rsmax*(1-a0/amax0)

            rsh0 = least_squares( fun = shunt_conductance_numerator_nsh_rs, 
                                    x0 = r0, 
                                    jac = partial_shunt_conductance_numerator_dnsh_drs, 
                                    args = (u0,imp,vmp), 
                                    method = 'lm' ).x[0]

            rmp0 = least_squares( fun = maximum_power_point_function_fmp_rs, 
                                    x0 = r0, 
                                    jac = partial_maximum_power_point_function_dfmp_drs, 
                                    args = (u0,imp,vmp), 
                                    method = 'lm' ).x[0]

            while rmp0 > rsh0:

                a0  = (ainf+a0)/2
                r0 = rsmax*(1-a0/amax0)
                u0 = np.exp(1/a0)

                rsh0 = least_squares( fun = shunt_conductance_numerator_nsh_rs, 
                                        x0 = r0, 
                                        jac = partial_shunt_conductance_numerator_dnsh_drs, 
                                        args = (u0,imp,vmp), 
                                        method = 'lm' ).x[0]

                rmp0 = least_squares( fun = maximum_power_point_function_fmp_rs, 
                                        x0 = r0, 
                                        jac = partial_maximum_power_point_function_dfmp_drs, 
                                        args = (u0,imp,vmp), 
                                        method = 'lm' ).x[0]
                
            r0 = rsh0

            sol = least_squares( 
                fun = equation_system_sd2_region, 
                x0 = (u0,r0), 
                jac = jacobian_equation_system_sd2_region, 
                args = (imp,vmp), 
                method = 'lm' )
            
            amax  = 1/np.log(sol.x[0])
            rsmin = sol.x[1]

            return amax, rsmin, region
        
    except:
        return np.nan, np.nan, np.nan

def maximum_power_resistance_function(imp,vmp,asup=0,ainf=ainf,Npoints=250):

    # if ainf == 'auto':
    #     ainf = 1/300/np.log(10)

    amax_mp, rsmin, region = one_dimensional_sdm_domain(imp,vmp)
    ash = maximum_equivalent_factor_diode_ash_max(imp,vmp)

    amax = ash
    if asup > amax:
        amax = asup
     
    a   = np.linspace(amax,ainf,Npoints)
    u = np.exp(1/a)
    rs = []
    rs0 = rsmin 

    for cont in range(Npoints):

        sol = least_squares(
                fun=maximum_power_point_function_fmp_rs,
                jac=partial_maximum_power_point_function_dfmp_drs,
                x0=rs0,
                args=(u[cont],imp,vmp),
                method='lm')

        rs0 = sol.x[0]
        rs.append(rs0)

    rs = np.asarray(rs,dtype=float)

    return a, rs

def shunt_resistance_function(imp,vmp,ainf=ainf,Npoints=10):

    # if ainf == 'auto':
    #     ainf = 1/300/np.log(10)

    amax = maximum_equivalent_factor_diode_ash_max(imp,vmp)

    a = np.linspace(amax,ainf,Npoints)
    u = np.exp(1/a)
    rs0 = 0
    rs = [ ] 

    for cont in range(Npoints):

        sol = least_squares(
                fun= shunt_conductance_numerator_nsh_rs,
                jac=partial_shunt_conductance_numerator_dnsh_drs,
                x0=rs0,
                args=(u[cont],imp,vmp),
                method='lm' )

        rs0 = sol.x[0]
        rs.append(rs0) 
    
    rs = np.asarray(rs,dtype=float)

    return a, rs

def calculate_parameters_sd1(a: float,imp: float,vmp: float) -> tuple:

    amp, rmp, region = one_dimensional_sdm_domain(imp,vmp)
    rmax = (1-vmp)/imp

    u = np.exp(1/a)
    r0 = (rmp-rmax)*a/amp+rmax

    rs = least_squares(
            fun=maximum_power_point_function_fmp_rs,
            jac=partial_maximum_power_point_function_dfmp_drs,
            x0=r0,
            args=(u,imp,vmp),
            method='lm').x[0]
    
    iph, io, gsh = affine_transformation_sd2(a,rs,1,1,imp,vmp)

    return iph,io,a,rs,gsh

def fsd2_rs(a, imp, vmp):

    amp, rmp, region = one_dimensional_sdm_domain(imp,vmp)
    rmax = (1-vmp)/imp

    u = np.exp(1/a)
    r0 = (rmp-rmax)*a/amp+rmax

    rs = least_squares(
            fun=maximum_power_point_function_fmp_rs,
            jac=partial_maximum_power_point_function_dfmp_drs,
            x0=r0,
            args=(u,imp,vmp),
            method='lm').x[0]

    return rs

def fsd2_ipv(a, rs, vpv, imp, vmp):
    t3 = 0.1e1 / a
    t5 = np.exp(t3 * (imp * rs + vmp))
    t6 = -vmp + 1
    t8 = np.exp(t3 * rs)
    t10 = np.exp(t3)
    t12 = t10 * vmp + t6 * t8 - t5
    t16 = vpv - 1
    t20 = imp * t16
    t34 = 0.1e1 / t12
    # t36 = np.exp(t34 * t3 * (t5 * (rs * t16 - vpv) + t8 * (-rs * t20 + t6 * vpv) + t10 * ((vpv * imp + vmp - vpv) * rs + vpv * vmp)))
    # t40 = scipy.special.lambertw(t34 * t3 * t36 * rs * (imp + vmp - 1))

    t40 = lw_roberts(
        t34 * t3 * (t5 * (rs * t16 - vpv) + t8 * (-rs * t20 + t6 * vpv) + t10 * ((vpv * imp + vmp - vpv) * rs + vpv * vmp)) + np.log(t34 * t3 * rs * (imp + vmp - 1))
    )

    return(t34 / rs * (-t40 * t12 * a + rs * (t5 * t16 - t8 * t20 + t10 * (vmp + (imp - 1) * vpv))))
