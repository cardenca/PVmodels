
# Python packages
import os, sys
sys.path.append( os.getcwd() )

import numpy as np
from scipy.optimize import least_squares
from scipy.constants import Boltzmann, elementary_charge
import matplotlib.pyplot as plt

from lumped_models.single_diode_model import photovoltaic_current
from lumped_models.single_diode_model import photovoltaic_voltage

# Número de módulos en serie
Nmod = 8
Nserie = 66
Tamb = 25+273.15
Vt = Tamb*Boltzmann/elementary_charge

# string dictionary
moduleParams = { }

# reference parameters SDM
sdParams = {
    'iph': 7.5, 'io':2.436e-10, 'a': Nserie*Vt, 'rs': 1e-2, 'gsh': 1e-4
}

# 3-level nested dictionary with keys: module (level-1), side (level-2), module parameters (level-3)
for nMod in range(Nmod):
    modName = 'module-'+str(nMod)
    moduleParams[ modName ] = sdParams

# reference parameters diode (basic model)
diodeParams = {
    'io':1e-6, 'a': 1.5*Vt
}

bypassParams = { }

# 2-level nested dictionary with keys: module (level-1), diode parameters (level-2)
for nMod in range(Nmod):
    modName = 'module-'+str(nMod)
    bypassParams[ modName ] = diodeParams


ipv = 0
inputParams = [ moduleParams, bypassParams, ipv ]

def _photovoltaic_voltage_residual( x, *args ):

    moduleParams, bypassParams, ipv = args 

    Nmod = len(moduleParams)
    fout = [ 0 ]*2*Nmod

    moduleList = list( moduleParams.keys() )
    modCount = 0

    for module in moduleList:
        
        sstringParameters = moduleParams[module]
        sbypassParameters = bypassParams[module]

        vparal  = x[ modCount ]
        vbypass = x[ modCount+1 ]

        idiode  = sstringParameters['io']*(np.exp(vparal/sstringParameters['a'])-1)
        ishunt  = sstringParameters['gsh']*vparal
        iseries = sstringParameters['iph']-idiode-ishunt
        vseries = sstringParameters['rs']*iseries

        ibypass = sbypassParameters['io']*(np.exp(-vbypass/sbypassParameters['a'])-1)

        fout[ modCount ]   = vparal - vseries - vbypass
        fout[ modCount+1 ] = iseries + ibypass - ipv

        modCount+=2

    return fout

Voc_module_app = 40
xinit = [ 40 ]*2*Nmod

# resolution of equation system
sol = least_squares(
    fun=_photovoltaic_voltage_residual,
    x0=xinit,
    args=[ moduleParams, bypassParams, 0 ],
    method='lm'
)

Voc = 0
for nvj in range(0,len(sol.x),2):
    Voc += sol.x[nvj]


# Isc_distributed = iseries+ibypass 
Isc_lumped = photovoltaic_current( 0, 7.5, 2.436e-10, Nmod*Nserie*Vt, Nmod*1e-2, 1e-4/Nmod, method='lw-roberts')
Voc_lumped = photovoltaic_voltage( 0, 7.5, 2.436e-10, Nmod*Nserie*Vt, Nmod*1e-2, 1e-4/Nmod, method='lw-roberts')

Ipv_input = np.linspace(0,Isc_lumped,1000)
Vpv_computed = [ ]

for curr in Ipv_input:

    # resolution of equation system
    sol = least_squares(
        fun=_photovoltaic_voltage_residual,
        x0=xinit,
        args=[ moduleParams, bypassParams, curr ],
        method='lm'
    )

    volt = 0
    for nvj in range(0,len(sol.x),2):
        volt += sol.x[nvj]

    Vpv_computed.append( volt )

    xinit = sol.x

plt.plot(
    Vpv_computed, Ipv_input, 'o'
)
plt.show()

# print( sol.x )

# print(Voc, voc_lumped)





