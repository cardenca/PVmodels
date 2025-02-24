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
from scipy.special import lambertw

# Custom functions
# -----------------



# LaTeX font - plt
# -----------------

plt.rc('text.latex', preamble=r'\usepackage{nicefrac} \usepackage{mathtools} \usepackage{amssymb} \usepackage{amsmath} \usepackage{xcolor}')
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Computer Modern Roman" })

# Main
# -----------------

wz1 = np.linspace(-4,-1,100) 
wz0 = np.linspace(-1,2,100)
z1 = wz1*np.exp(wz1)
z0 = wz0*np.exp(wz0)

wz = np.linspace(-4,2,100)
z  = wz*np.exp(wz)

# z = 1
# wz = lambertw( z ).real

zmin = -1*np.exp(-1)

# print(f'minimum Z: {zmin:.2f}')

# print(wz*np.exp(wz))

plt.plot(z0,wz0,label='Branch: $W_0$ (principal)')
plt.plot(z1,wz1,label='Branch $W_{-1}$')
plt.axvline(zmin,alpha=0.5,label='$z_\mathrm{min}=-\\nicefrac{1}{e}$',linestyle='dashed')

plt.xlabel('Independent variable z')
plt.ylabel('Function $W$')

plt.xlim([-1,6])
plt.ylim([-4,2])

plt.legend(loc='best',fancybox=True,framealpha=1,shadow=True)
plt.grid(linestyle='dotted',alpha=0.5)

plt.savefig( r"./lwfunction/wlambert_function.pdf", bbox_inches="tight", pad_inches=0 ) 

plt.show()

