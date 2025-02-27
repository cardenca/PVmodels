
import os, sys
sys.path.append( os.getcwd() )

import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge
import numpy as np
from prettytable import PrettyTable

# Custom functions
from models.null_resistive_losses_model import photovoltaic_current
from models.null_resistive_losses_model import photovoltaic_voltage
from models.null_resistive_losses_model import maximum_power_point
from models.null_resistive_losses_model import affine_transformation

# Example 
Isc, Voc = 1, 1
Tc = 25+273.15 # Cell temperature
Ns = 72 # series solar cell
Vt = Boltzmann*Tc/elementary_charge 
A = Vt*Ns

Iph_test, Io_test = affine_transformation(A, Isc, Voc)

Voc_test = photovoltaic_voltage(0, Iph_test, Io_test, A)
Isc_test = photovoltaic_current(0, Iph_test, Io_test, A)
Vmp_test, Imp_test = maximum_power_point(Iph_test, Io_test, A)

Vpv_test = np.linspace( 0, Voc_test, 150 )
Ipv_test = photovoltaic_current(Vpv_test, Iph_test, Io_test, A)

table = PrettyTable()

table.field_names = ["Variable", "Datasheet", "NRLM-1"]
table.add_row(["Isc", Isc, Isc_test])
table.add_row(["Voc", Voc, Voc_test])
table.add_row(["Imp", '-', Imp_test])
table.add_row(["Vmp", '-', Vmp_test])

print(table)

plt.figure(1)
plt.plot(
    [0, Voc], [Isc, 0], 'o', mfc='none', label='Input cardinal points'
)
plt.plot(
    Vpv_test, Ipv_test, label='Estimated I--V curve'
)
plt.plot(
    Vmp_test, Imp_test, 'o', mfc='none', color='orange'
)
plt.xlabel(r'Photovoltaic voltage, $V_\mathrm{pv}$ (V)')
plt.ylabel(r'Photovoltaic current, $I_\mathrm{pv}$ (A)')

plt.figure(2)
plt.plot(
    [0, Vmp_test, Voc_test], [0, Vmp_test*Imp_test, 0], 'o', mfc='none', color='orange'
)
plt.plot(
    Vpv_test, Ipv_test*Vpv_test, color='tab:orange'
)

plt.xlabel(r'Photovoltaic voltage, $V_\mathrm{pv}$ (V)')
plt.ylabel(r'Photovoltaic power, $P_\mathrm{pv}$ (W)')

plt.show()