
import os, sys
sys.path.append( os.getcwd() )

import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
import numpy as np
from prettytable import PrettyTable

# Custom functions
from models.single_diode_model import photovoltaic_current
from models.single_diode_model import photovoltaic_voltage
from models.single_diode_model import maximum_power_point

from models.single_diode_model import calculate_parameters_sd1
from models.single_diode_model import calculate_domain_sd1

from models.single_diode_model import affine_transformation_sd2
from models.single_diode_model import affine_transformation_sd3

# Cardinal points
Isc, Voc = 9.01, 52.46
Imp, Vmp = 8.48, 42.99
imp, vmp =  Imp/Isc, Vmp/Voc

Tc = 25+273.15 # Cell temperature
Ns = 72 # series solar cell
Vt = Boltzmann*Tc/elementary_charge 

# Calcule domain of positive paramters fitting cardinal points
Amax, Rsmin, region = calculate_domain_sd1(Isc, Vmp, Imp, Voc)
# nmax = 1.5

# print(
#     Amax / Vt / Ns
# )

A_test = 1*Vt*Ns

# calculate sd1 paramters (scaled)
Iph_test,Io_test,Rs_test,Gsh_test = calculate_parameters_sd1(
    A_test, Isc, Vmp, Imp, Voc
)

print(
    Iph_test,Io_test,Rs_test,Gsh_test
)

# calculate cardinal points from the model
Voc_test= photovoltaic_voltage(0, Iph_test, Io_test, A_test, Rs_test, Gsh_test)
Vpv_test = np.linspace( 0, Voc_test, 150 )

Ipv_test = photovoltaic_current(Vpv_test, Iph_test, Io_test, A_test, Rs_test, Gsh_test)
Isc_test = Ipv_test[0]

Vmp_test, Imp_test = maximum_power_point(Iph_test, Io_test, A_test, Rs_test, Gsh_test, Isc_test, Voc_test)

table = PrettyTable()

table.field_names = ["Variable", "Datasheet", "SDM-1 estimation"]
table.add_row(["Isc", Isc, Isc_test])
table.add_row(["Voc", Voc, Voc_test])
table.add_row(["Imp", Imp, Imp_test])
table.add_row(["Vmp", Vmp, Vmp_test])

print(
    table
)

plt.figure(1)
plt.plot(
    [0, Vmp, Voc], [Isc, Imp, 0], 'o', mfc='none', label='Input cardinal points'
)
plt.plot(
    Vpv_test, Ipv_test, label='Estimated I--V curve'
)
plt.xlabel(r'Photovoltaic voltage, $V_\mathrm{pv}$ (V)')
plt.ylabel(r'Photovoltaic current, $I_\mathrm{pv}$ (A)')

plt.figure(2)
plt.plot(
    [0, Vmp, Voc], [0, Vmp*Imp, 0], 'o', mfc='none', label='Input cardinal points'
)
plt.plot(
    Vpv_test, Ipv_test*Vpv_test
)

plt.xlabel(r'Photovoltaic voltage, $V_\mathrm{pv}$ (V)')
plt.ylabel(r'Photovoltaic power, $P_\mathrm{pv}$ (W)')

plt.show()


