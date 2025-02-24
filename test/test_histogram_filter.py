
import os, sys
sys.path.append( os.getcwd() )

import numpy as np
from matplotlib import pyplot as plt

from package.src.math_support.histogram import histogram_filt

ndata = 15
x = np.sort(np.random.random(ndata))
y = np.random.random(ndata)

xfilt, yfilt = histogram_filt(x, y)

plt.figure(1)
plt.plot(
    x, y, 'o', mfc='none', alpha=0.5, label='Synthetic data'
)
plt.plot(
    xfilt, yfilt, '--x', alpha=0.5, label='Filtered data'
)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(linestyle='dotted', alpha=0.5)
plt.legend()

plt.figure(2)
plt.hist(
    x,
    bins=ndata,
    histtype='step'
)
plt.grid(linestyle='dotted', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.show()


