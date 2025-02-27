
# Python packages
import os, sys
sys.path.append( os.getcwd() )

import numpy as np
from scipy.optimize import least_squares
from scipy.constants import Boltzmann, elementary_charge
import matplotlib.pyplot as plt

from lumped_models.single_diode_model import photovoltaic_current
from lumped_models.single_diode_model import photovoltaic_voltage

import time

# Número de módulos en serie
Nmod = 8
Nserie = 66
Tamb = 25+273.15
Vt = Tamb*Boltzmann/elementary_charge

# string dictionary
moduleParams = { }

# reference parameters SDM
sdParams = {
    'iph': 7.5, 'io':2.436e-10, 'a': Nserie*Vt, 'gsh': 1e-4, 'rs': 1e-2
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

def _photovoltaic_current_residual( x, *args ):

    moduleParams, bypassParams, volt = args 

    Nmod = len(moduleParams)
    fout = [ 0 ]*2*Nmod

    moduleList = list( moduleParams.keys() )
    modCount = 0
    cumBypass = 0

    for module in moduleList:
        
        sstringParameters = moduleParams[module]
        sbypassParameters = bypassParams[module]

        idiode  = sstringParameters['io']*(np.exp(x[ modCount ]/sstringParameters['a'])-1)
        ishunt  = sstringParameters['gsh']*x[ modCount ]
        iseries = sstringParameters['iph']-idiode-ishunt
        vseries = sstringParameters['rs']*iseries

        ibypass = sbypassParameters['io']*(np.exp(-x[ modCount+1 ]/sbypassParameters['a'])-1)

        fout[ modCount ]   = x[ modCount ] - vseries - x[ modCount+1 ]
        fout[ modCount+1 ] = iseries + ibypass
        fout[ modCount-1 ] = fout[ modCount-1 ]-iseries-ibypass

        cumBypass += x[ modCount+1 ]

        modCount+=2

    fout[ -1 ] = volt - cumBypass

    return fout

# Isc_distributed = iseries+ibypass 
# Isc_lumped = photovoltaic_current( 0, 7.5, 2.436e-10, Nmod*Nserie*Vt, Nmod*1e-2, 1e-4/Nmod )
Voc_lumped = photovoltaic_voltage( 0, 7.5, 2.436e-10, Nmod*Nserie*Vt, Nmod*1e-2, 1e-4/Nmod)
# Voc_lumped = 400

Vpv_input = np.linspace(Voc_lumped,0,500)
Ipv_computed = []

xinit = [ Voc_lumped/Nmod ]*2*Nmod

timeinit = time.process_time()

for volt in Vpv_input:

    # resolution of equation system
    sol = least_squares(
        fun=_photovoltaic_current_residual,
        x0=xinit,
        args=[ moduleParams, bypassParams, volt ],
        xtol=1e-8,
        method='lm'
    )

    vparal = sol.x[ 0 ]
    vbypass = sol.x[ 1 ]

    idiode  = moduleParams['module-0']['io']*(np.exp(vparal/moduleParams['module-0']['a'])-1)
    ishunt  = moduleParams['module-0']['gsh']*vparal
    iseries = moduleParams['module-0']['iph']-idiode-ishunt

    ibypass = bypassParams['module-0']['io']*(np.exp(-vbypass/bypassParams['module-0']['a'])-1)

    Ipv_computed.append( iseries+ibypass )

    xinit = sol.x


timefin = time.process_time()


print(
    'Computation time: ', timefin-timeinit
)

plt.plot(
    Vpv_input, Ipv_computed
)
plt.xlabel('voltage')
plt.ylabel('current')
plt.show()



