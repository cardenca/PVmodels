
# Python packages
# -----------------

import numpy as np
from scipy.optimize import least_squares
# from scipy.special import lambertw

from lambertW import lw_roberts

# Custom functions
# -----------------

def photovoltaic_voltage(ipv,iph,io,a,rs):
    return a*np.log((iph+io-ipv)/io)-rs*ipv

def function_null_shunt_conductance(ipv,vpv,iph,io,a,rs):
    return iph-io*(np.exp((vpv+rs*ipv)/a)-1)-ipv

def pdevIpv_function_null_shunt_conductance(ipv,vpv,iph,io,a,rs):
    return -io*rs*np.exp((vpv+rs*ipv)/a)/a - 1

def photovoltaic_current( vpv, iph,io,a,rs,
                          method='lm'):
    
    if method=='lm':

        # Initial point estimation
        ipv0=iph+io
        ipv1 = (np.log((iph+io)/io)*a-vpv)/rs
        if ipv0 > ipv1:
            ipv0 = ipv1

        # Resolution of the system
        sol = least_squares(
            fun=function_null_shunt_conductance,
            jac=pdevIpv_function_null_shunt_conductance,
            x0=ipv0,
            args=(vpv,iph,io,a,rs),
            method='lm'
        )

        return sol.x[0]

    if method=='lw':
        t2 = 0.1e1 / a
        t4 = rs * (io + iph)
        # t7 = np.exp(t2 * (t4 + vpv))
        # t10 = lambertw(t7 * t2 * io * rs).real
        t10 = lw_roberts( t2 * (t4 + vpv) + np.log(t2 * io * rs) )
        return(0.1e1 / rs * (-a * t10 + t4))

def maximum_power_point(iph,io,a,rs):

    isc = photovoltaic_current(0,iph,io,a,rs)
    voc = photovoltaic_voltage(0,iph,io,a)

    iph_scaled = iph/isc
    io_scaled  = io/isc
    a_scaled = a/voc
    rs_scaled = rs*isc/voc
            
    # TODO: Improve estimation of MPP for limits situations 

    vmp0_scaled, imp0_scaled = _scaled_maximum_power_initial_point(
            iph_scaled,io_scaled,a_scaled,rs_scaled
            )
        
    sol = least_squares(
        fun=_scaled_maximum_power_equation_system,
        x0=(vmp0_scaled,imp0_scaled),
        jac=_scaled_maximum_power_equation_jacobian,
        method='lm',
        args=(iph_scaled,io_scaled,a_scaled,rs_scaled)
    )
    
    return sol.x[0]*voc, sol.x[1]*isc, 

def _scaled_maximum_power_initial_point(iph,io,a,rs):
    t2 = io ** 2
    t3 = rs * t2
    t4 = rs - 1
    t6 = 0.1e1 / a
    t8 = np.exp(t6 * (rs + 1))
    t13 = np.exp(t6 * rs)
    t14 = t13 * rs * io
    t16 = np.exp(t6)
    t24 = 0.1e1 / (t16 - t13)
    vmp0 = t24 / io * t6 * (-t8 * t4 * t3 - (io * t16 * t4 + a + t14) * a)
    t29 = np.exp(t6 * (2 * rs + 1))
    imp0 = 0.1e1 / (t14 + a) * t24 * t6 * (t29 * t4 * t3 + (2 * t8 * io * (rs - 0.1e1 / 0.2e1) + t16 * a) * a)
    return vmp0, imp0

def _scaled_maximum_power_equation_system(x,iph,io,a,rs):
    vmp, imp = x
    t2 = imp * rs
    t6 = np.exp(1 / a * (t2 + vmp))
    A0 = iph - (t6 - 1) * io - imp
    A1 = -t6 * (t2 - vmp) * io - a * imp
    return np.array([A0,A1])

def _scaled_maximum_power_equation_jacobian(x,iph,io,a,rs):
    vmp, imp = x
    t2 = 0.1e1 / a
    t4 = imp * rs
    t7 = np.exp(t2 * (t4 + vmp))
    A0 = -t7 * t2 * io
    t9 = io * rs
    t10 = t7 * t2
    A1 = -t10 * t9 - 1
    t14 = (t4 - vmp) * io
    A2 = io * t7 - t14 * t10
    A3 = -rs * t14 * t2 * t7 - t7 * t9 - a
    return np.array([[A0,A1],
                     [A2,A3]])

def affine_transformations(a,rs,isc,voc):
    t2 = 0.1e1 / a
    t4 = np.exp(t2 * voc)
    t9 = np.exp(t2 * rs * isc)
    t11 = 0.1e1 / (-t4 + t9)
    iph = -t11 * isc * (t4 - 1)
    io = -isc * t11
    return iph, io


def _maximum_power_equation_system_nsh2(x,a,rs):
    vmp, imp = x
    t2 = imp * rs
    t4 = 0.1e1 / a
    t6 = np.exp(t4 * (t2 + vmp))
    t8 = np.exp(t4 * rs)
    t10 = np.exp(t4)
    A0 = -t6 + t8 * imp - (imp - 1) * t10
    A1 = t4 * (t6 * (-t2 + vmp) - (t10 - t8) * imp * a)
    return A0, A1

def _jacobian_maximum_power_equation_system_nsh2(x,a,rs):
    vmp, imp = x
    t2 = 0.1e1 / a
    t3 = imp * rs
    t6 = np.exp(t2 * (t3 + vmp))
    t7 = t6 * t2
    A0 = -t7
    t8 = t2 * rs
    t10 = np.exp(t8)
    t11 = np.exp(t2)
    A1 = -t6 * t8 + t10 - t11
    t12 = -t3 + vmp
    A2 = t2 * (t12 * t2 * t6 + t6)
    A3 = t2 * (-t6 * rs + t7 * rs * t12 - (-t10 + t11) * a)
    return np.array([[A0,A1],[A2,A3]])

def maximum_power_point_ainf(a,rs,vmp0,imp0):
            
    sol = least_squares(
        fun=_maximum_power_equation_system_nsh2,
        x0=(vmp0,imp0),
        jac=_jacobian_maximum_power_equation_system_nsh2,
        method='lm',
        args=(a,rs),
    )
    
    return sol.x[0], sol.x[1]

