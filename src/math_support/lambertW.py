
import numpy as np

def lw_roberts(x,kcont=5):
    # Method extracted from https://arxiv.org/abs/1504.01964
    me = (1+np.e)/2/np.e
    ne = (1-np.e)/2 
    y0 = (x<=-np.e)*x + (-np.e<x)*(x<np.e)*(me*x+ne) + (np.e<=x)*np.log(abs(x))

    for _ in range(kcont):
        y0 = y0 - 2*(y0 + np.exp(y0) - x)*(1+np.exp(y0))/(2*(1+np.exp(y0))**2-(y0 + np.exp(y0) - x)*np.exp(y0))
    return np.exp(y0)

