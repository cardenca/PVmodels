
# Python packages
# -----------------

import numpy as np
from scipy.optimize import least_squares
import pandas as pd

# Computable limits input
# -----------------

from lumped_models.data.computational_constants import ainf as ainf_system

computational_limits = pd.read_pickle(r'./lumped_models/data/computational_limits.pkl')
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

def one_dimensional_sdm_domain(imp: float,vmp: float, ainf: float = ainf_system) -> tuple:
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

def maximum_power_resistance_function(imp,vmp,asup=0,ainf=ainf_system,Npoints=250):

    # if ainf == 'auto':
    #     ainf = 1/300/np.log(10)

    amax_mp, rsmin, region = single_parameter_single_diode_model_sd1_constrains(imp,vmp)
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

def shunt_resistance_function(imp,vmp,ainf=ainf_system,Npoints=10):

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

def five_parameters_sd1(a,imp,vmp):

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

