
# Python packages
# -----------------

import numpy as np
from scipy.optimize import least_squares
from lwfunction.roberts import lw_roberts

def photovoltaic_current(vpv,iph,io,a,gsh):
    t3 = np.exp(vpv / a)
    return(-gsh * vpv - io * t3 + io + iph)

def function_null_series_resistance(vpv,ipv,iph,io,a,gsh):
    return iph-io*(np.exp(vpv/a)-1)-gsh*vpv-ipv

def pdevVpv_function_null_series_resistance(vpv,ipv,iph,io,a,gsh):
    return -io*np.exp(vpv/a)/a-gsh

def photovoltaic_voltage(ipv,iph,io,a,gsh,method='lm'):
    if method == 'lm':
        # Initial point estimation
        vpv0 = gsh*(iph+io-ipv)
        vpv1 = a*np.log((iph+io-ipv)/io)
        if vpv0 > vpv1:
            vpv0 = vpv1
        # Resolution of the system
        sol = least_squares(
            fun=function_null_series_resistance,
            jac=pdevVpv_function_null_series_resistance,
            x0=vpv0,
            args=(ipv,iph,io,a,gsh),
            method='lm'
        )
        return sol.x[0]
    elif method == 'lw':
        t1 = 0.1e1 / a
        t3 = 0.1e1 / gsh
        t6 = t1 * (io + iph - ipv) * t3
        t10 = lw_roberts( t6 + np.log( t3 * t1 * io) )
        return(a * (-t10 + t6))

def maximum_power_point(iph,io,a,gsh):

    isc = photovoltaic_current(0,iph,io,a,gsh)
    voc = photovoltaic_voltage(0,iph,io,a,gsh)

    iph_scaled = iph/isc
    io_scaled  = io/isc
    a_scaled = a/voc
    gsh_scaled = gsh*voc/isc
        
    # Initial point estimated using the slope method  
    vmp0_scaled, imp0_scaled = _scaled_maximum_power_initial_point(
        iph_scaled,io_scaled,a_scaled,gsh_scaled)

    sol_scaled = least_squares(
        fun=_scaled_maximum_power_equation_system,
        x0=(vmp0_scaled,imp0_scaled),
        jac=_scaled_maximum_power_equation_jacobian,
        method='lm',
        args=(iph_scaled,io_scaled,a_scaled,gsh_scaled)
    )
    
    return sol_scaled.x[0]*voc, sol_scaled.x[1]*isc, 
    
def total_derivative_current_voltage(vpv,ipv,iph,io,a,gsh):
    return -io*np.exp(vpv/a)/a-gsh

def _scaled_maximum_power_initial_point(
                iph_scaled,io_scaled,a_scaled,gsh_scaled
                ):
    msc = total_derivative_current_voltage(
        0,1,
        iph_scaled,io_scaled,a_scaled,gsh_scaled)

    moc = total_derivative_current_voltage(
        1,0,
        iph_scaled,io_scaled,a_scaled,gsh_scaled)

    return (moc+1)/(moc-msc), moc*(msc+1)/(moc-msc)

def _scaled_maximum_power_equation_system(x,iph,io,a,gsh):
    vmp,imp=x
    t2 = 0.1e1 / a
    t4 = np.exp(t2 * vmp)
    t7 = gsh * vmp
    A0 = iph - (t4 - 1) * io - t7 - imp
    A1 = t2 * (vmp * io * t4 + (t7 - imp) * a)
    return np.array([A0,A1])

def _scaled_maximum_power_equation_jacobian(x,iph,io,a,gsh):
    vmp,imp=x
    t2 = 0.1e1 / a
    t4 = np.exp(t2 * vmp)
    A0 = t2 * (-a * gsh - io * t4)
    A1 = -1
    t11 = a ** 2
    A2 = 0.1e1 / t11 * (t4 * (a + vmp) * io + gsh * t11)
    A3 = -1
    return np.array([[A0,A1],
                     [A2,A3]])

def affine_transformations(a,gsh,isc,voc):
    iph = isc
    t6  = np.exp(voc / a)
    io  = -0.1e1 / (t6 - 1) * (gsh * voc - isc)
    return iph, io



def _maximum_power_equation_system_nsr2(x,a,gsh):
    vmp, imp = x
    t2 = gsh * vmp
    t4 = 0.1e1 / a
    t5 = np.exp(t4)
    t7 = gsh - 1
    t9 = np.exp(t4 * vmp)
    A0 = t5 * (-t2 - imp + 1) + t9 * t7 + (-1 + vmp) * gsh + imp
    A1 = t4 * (-t9 * t7 * vmp + (t2 - imp) * (t5 - 1) * a)
    return A0, A1

def _jacobian_maximum_power_equation_system_nsr2(x,a,gsh):
    vmp, imp = x
    t2 = gsh - 1
    t3 = 0.1e1 / a
    t5 = np.exp(t3 * vmp)
    t7 = np.exp(t3)
    t8 = t7 - 1
    A0 = t3 * (-a * gsh * t8 + t2 * t5)
    A1 = -t8
    t15 = a ** 2
    A2 = 0.1e1 / t15 * (-t5 * (a + vmp) * t2 + t8 * t15 * gsh)
    A3 = A1
    return np.array([[A0,A1],[A2,A3]])

def maximum_power_point_ainf(a,gsh,vmp0,imp0):
            
    sol = least_squares(
        fun=_maximum_power_equation_system_nsr2,
        x0=(vmp0,imp0),
        jac=_jacobian_maximum_power_equation_system_nsr2,
        method='lm',
        args=(a,gsh),
    )
    
    return sol.x[0], sol.x[1]