#
# * Packages
# * -----------------

# Adding directory
# -----------------

import os, sys
sys.path.append( os.getcwd() )

# Python packages
# -----------------

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Custom functions
# -----------------

# from lumped_models.null_resistive_losses_model import photovoltaic_current
# from lumped_models.null_resistive_losses_model import photovoltaic_voltage
# from lumped_models.null_resistive_losses_model import mpp_nrlm
from lumped_models.null_resistive_losses_model import nrl_limit
from lumped_models.null_shunt_conductance_model import maximum_power_point_ainf as mpp_nsh
from lumped_models.null_series_resistance_model import maximum_power_point_ainf as mpp_nsr

# from lumped_models.data.computational_constants import float_supremum_python
from lumped_models.data.computational_constants import ainf

# Python maximum float
# -----------------

# float_supremum_python = sys.float_info.max
# ainf  = 1/np.log(float_supremum_python)
# # a_supremum_scaled = 1/np.log(1+2e-16)

# ainf_mods = 1/np.log(10)/302
ninterpol = 1000

# print(float_supremum_python)

# print(f'Python sup: {float_supremum_python:.3e}')
# print(f'ainf: {ainf:.3e}')
# print(f'ainf_mods: {ainf_mods:.3e}')


# NRLM
# ------------

anrl_list = 10**np.linspace( np.log10(ainf),7,ninterpol)
vmp_nrl, imp_nrl = [], []
vmp0, imp0 = 1, 1

for anrl in anrl_list:
    vmp_sol, imp_sol = nrl_limit(anrl,vmp0,imp0)
    vmp_nrl.append(vmp_sol)
    imp_nrl.append(imp_sol)
    vmp0, imp0 = vmp_sol, imp_sol

vmp_nrl = np.asarray(vmp_nrl,dtype=float)
imp_nrl = np.asarray(imp_nrl,dtype=float)

nrlm_sort = np.argsort(vmp_nrl)
vmp_nrl_sort = vmp_nrl[nrlm_sort]
imp_nrl_sort = imp_nrl[nrlm_sort]

# NSHM
# ------------

rs_border = np.linspace(0,1,ninterpol)
vmp_nsh, imp_nsh = [], []
vmp0, imp0 = 1, 1

for rs in rs_border:
    vmp_sol, imp_sol = mpp_nsh(ainf,rs,vmp0,imp0)
    vmp_nsh.append(vmp_sol)
    imp_nsh.append(imp_sol)
    vmp0, imp0 = vmp_sol, imp_sol

vmp_nsh = np.asarray(vmp_nsh,dtype=float)
imp_nsh = np.asarray(imp_nsh,dtype=float)

nshm_sort = np.argsort(vmp_nsh)
vmp_nsh_sort = vmp_nsh[nshm_sort]
imp_nsh_sort = imp_nsh[nshm_sort]

# NSRM
# ------------

gsh_border = np.linspace(0,1,ninterpol)
vmp_nsr, imp_nsr = [], []
vmp0, imp0 = 1, 1

for gsh in gsh_border:
    vmp_sol, imp_sol = mpp_nsr(ainf,gsh,vmp0,imp0)
    vmp_nsr.append(vmp_sol)
    imp_nsr.append(imp_sol)
    vmp0, imp0 = vmp_sol, imp_sol  

vmp_nsr = np.asarray(vmp_nsr,dtype=float)
imp_nsr = np.asarray(imp_nsr,dtype=float)

nsrm_sort = np.argsort(vmp_nsr)
vmp_nsr_sort = vmp_nsr[nsrm_sort]
imp_nsr_sort = imp_nsr[nsrm_sort]

vmp_test, imp_test = 0.75, 0.75

vmp_min = max( min(vmp_nsh_sort), min(vmp_nsr_sort), min(vmp_nrl_sort) )
vmp_max = min( max(vmp_nsh_sort), max(vmp_nsr_sort), max(vmp_nrl_sort) )

imp_min = max( min(imp_nsh_sort), min(imp_nsr_sort), min(imp_nrl_sort) )
imp_max = min( max(imp_nsh_sort), max(imp_nsr_sort), max(imp_nrl_sort) )

# print( vmp_min, vmp_max )
# print( imp_min, imp_max )

# print(vmp_nsh_sort)

def interpolation_imp(vmp,vmp_region,imp_region):
    ndat = sum(vmp_region<vmp)
    v1, v2 = vmp_region[ndat-1], vmp_region[ndat]
    i1, i2 = imp_region[ndat-1], imp_region[ndat]
    mregion = (i2-i1)/(v2-v1)
    nregion = i1-mregion*v1
    return mregion*vmp+nregion

df = pd.DataFrame(
    {'vmp_nsh':vmp_nsh_sort,'imp_nsh':imp_nsh_sort,
     'vmp_nrl':vmp_nrl_sort,'imp_nrl':imp_nrl_sort,
     'vmp_nsr':vmp_nsr_sort,'imp_nsr':imp_nsr_sort}
)

df.to_pickle(r'./package/data/computational_limits.pkl')

# insh_sup = interpolation_imp(vmp_test,imp_test,vmp_nsh_sort,imp_nsh_sort)
# insr_sup = interpolation_imp(vmp_test,imp_test,vmp_nsr_sort,imp_nsr_sort)
# inrl_lim = interpolation_imp(vmp_test,imp_test,vmp_nrl_sort,imp_nrl_sort)


plt.figure(1)
plt.title('Computed from program')
plt.scatter(vmp_nrl_sort,imp_nrl_sort,s=1,color='black',alpha=0.5)
plt.scatter(vmp_nsh_sort,imp_nsh_sort,s=1,color='tab:red',alpha=0.5)
plt.scatter(vmp_nsr_sort,imp_nsr_sort,s=1,color='tab:blue',alpha=0.5)

# plt.axvline(vmp_test)
# plt.axhline(imp_test,color='black')

# plt.axhline(insh_sup,color='red')
# plt.axhline(insr_sup,color='blue')
# plt.axhline(inrl_lim,color='black')

plt.show()

