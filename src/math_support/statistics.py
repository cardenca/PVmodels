
import numpy as np

def residual(xmeas : list, xest : list) -> list:
    return xmeas - xest 

def rmsd(xmeas : list, xest : list) -> float:
    return np.sqrt(np.sum(residual(xest,xmeas)**2)/len(xmeas))

def rmsd_scaled(xmeas : list, xest : list) -> float:
    xmax = max(xmeas)
    xmin = min(xmeas)
    return np.sqrt(np.sum( (residual(xest,xmeas)/(xmax-xmin))**2)/len(xmeas))

