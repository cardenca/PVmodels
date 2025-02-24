
import numpy as np

def histogram_filt(x: list,y: list) -> tuple:
    """
    This function is intended to eliminate cumulous of data.

    Parameters
    --------
    x : list or array
        Usually voltage vector
    y : list or array
        Usually current vector
    Returns
    --------
    xcomp : array 
        filtered voltage
    ycomp : array
        filtered current
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    hist, bin_edges = np.histogram(x,bins=len(x))

    nidx_hist = [i for i,k in enumerate(hist) if k > 1] # values higher than 1 in the vector x

    xcomp = []
    ycomp = []

    bin0 = bin_edges[0]

    for nidx in nidx_hist:

        bin_min = bin_edges[nidx]
        bin_max = bin_edges[nidx+1]

        # Compute the average value in the bin
        if bin_max == bin_edges[-1]:
            yavg = np.average( y[(bin_min<=x)*(x<=bin_max)] )
            xavg = np.average( x[(bin_min<=x)*(x<=bin_max)] )
        else:
            yavg = np.average( y[(bin_min<=x)*(x<bin_max)] )
            xavg = np.average( x[(bin_min<=x)*(x<bin_max)] )


        # Create the compressed vector
        if bin_min == bin_edges[0]: # If the max-bin corresponds to the first item
            ycomp = [ yavg ]
            xcomp = [ xavg ]
        elif bin_min == bin0: # If there are two consecutives max-bins
            ycomp = np.concatenate( [ ycomp, [ yavg ] ] )
            xcomp = np.concatenate( [ xcomp, [ xavg ] ] )
        else:
            ycomp = np.concatenate( [ ycomp, y[(bin0<=x)*(x<bin_min)], [ yavg ] ] )
            xcomp = np.concatenate( [ xcomp, x[(bin0<=x)*(x<bin_min)], [ xavg ] ] )
        
        bin0 = bin_max

    # In the case there is no concentration in the last bin:
    if bin_max<bin_edges[-1]:
        ycomp = np.concatenate( [ ycomp, y[(bin0<=x)] ] )
        xcomp = np.concatenate( [ xcomp, x[(bin0<=x)] ] )

    return xcomp, ycomp
