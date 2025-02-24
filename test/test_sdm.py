
import os, sys
sys.path.append( os.getcwd() )

import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
import numpy as np

# Custom functions
from single_diode_model import photovoltaic_current
from single_diode_model import photovoltaic_voltage

from single_diode_model import calculate_parameters_sd1
from single_diode_model import one_dimensional_sdm_domain

# Cardinal points
Isc, Voc = 9, 40
Imp, Vmp = 7, 35
imp, vmp =  Imp/Isc, Vmp/Voc

Tc = 25+273.15 # Cell emperature
Ns = 72 # series solar cell
Vt = Boltzmann*Tc/elementary_charge 

# Calcule domain of positive paramters fitting cardinal points
amax, rmin, region = one_dimensional_sdm_domain(imp, vmp)
nmax = amax*Voc/Vt/Ns
Amax = amax*Voc*0.99

# calculate sd1 paramters (scaled)
iph,io,a,rs,gsh = calculate_parameters_sd1(
    Amax/Voc, Imp/Isc, Vmp/Voc
)

# calculate cardinal points from the model
Voc = Voc*photovoltaic_voltage(0, iph, io, a, rs, gsh, method='lw-roberts')
Vpv = np.linspace( 0, Voc, 100 )

Ipv = Isc*photovoltaic_current(Vpv/Voc, iph, io, a, rs, gsh, method='lw-roberts')


plt.figure(1)
plt.plot(
    [0, Vmp, Voc], [Isc, Imp, 0], 'o', mfc='none', label='Input cardinal points'
)
plt.plot(
    Vpv, Ipv, label='Estimated I--V curve'
)

plt.figure(2)
plt.plot(
    [0, Vmp, Voc], [0, Vmp*Imp, 0], 'o', mfc='none', label='Input cardinal points'
)
plt.plot(
    Vpv, Ipv*Vpv
)

plt.show()


