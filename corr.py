# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:45:02 2017

@author: wxiao
"""
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from utility import get_index, spearmanr_by_matrix, kendalltau_by_matrix
    
def corr(x, y, method="pearsonr"):
    """ 
    Compute pearson correlation on time series x and y
    
    Parameters
    ----------
    x: time series
    y: time series
    
    Returns
    -------
    corr: correlation between x and y
    """
    if method == "pearsonr":
        corr = pearsonr(x, y)[0]
    elif method == "spearmanr":
        corr = spearmanr(x, y)[0]
    elif method == "kendalltau":
        corr = kendalltau(x, y)[0]
    return corr
    
def batch_corr(x, y, method="pearsonr", ngap=1):
    """ 
    Compute correlation on streaming sequence x and y (batch method)
 
    Parameters
    ----------   
    x: time series
    y: time series
    method: determin which type of correlation is computed. Accept method is 
        "pearsonr", "spearmanr" or "kendalltau"
    ngap: output correlation every ngap observations
    
    Returns
    -------
    corrs: correlations computed at selected time indexes. The selected time indexes are ngap-1, 2*ngap-1, ...
    t: selected time indexes
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    corrs = np.empty(n1//ngap)
    if method == "pearsonr":
        for i in range(ngap-1, n1, ngap):
            corrs[(i+1)/ngap - 1] = pearsonr(x[:(i+1)], y[:(i+1)])[0]
    elif method == "spearmanr":
        for i in range(ngap-1, n1, ngap):
            corrs[(i+1)/ngap - 1] = spearmanr(x[:(i+1)], y[:(i+1)])[0] 
    elif method == "kendalltau":
        for i in range(ngap-1, n1, ngap):
            corrs[(i+1)/ngap -1] = kendalltau(x[:(i+1)], y[:(i+1)])[0] 
    else:
        raise ValueError(('The method "%s" is not supported. Please specify one of ' 
        'the following options: "pearsonr", "spearmanr" or "kendalltau"') % method)
    t = range(ngap-1, n1, ngap)
    return corrs, t
    
def batch_mv_corr(x, y, nwin, method="pearsonr", ngap=1):
    """ 
    Compute correlation on streaming sequence x and y (batch method) with moving windows
 
    Parameters
    ----------   
    x: time series
    y: time series
    nwin: window size
    method: determin which type of correlation is computed. Accept method is 
        "pearsonr", "spearmanr" or "kendalltau"
    ngap: output correlation every ngap observations
    
    Returns
    -------
    corrs: correlations computed at selected time indexes. The selected time indexes are nwin-1, nwin-1+ngap, ...
    t: selected time indexes    
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    corrs = np.empty((n1 - nwin)//ngap + 1)
    if method == "pearsonr":
        for i in range(nwin-1, n1, ngap):
            corrs[(i-nwin+1)/ngap] = pearsonr(x[(i+1-nwin):(i+1)], y[(i+1-nwin):(i+1)])[0]
    elif method == "spearmanr":
        for i in range(nwin-1, n1, ngap):
            corrs[(i-nwin+1)/ngap] = spearmanr(x[(i+1-nwin):(i+1)], y[(i+1-nwin):(i+1)])[0] 
    elif method == "kendalltau":
        for i in range(nwin-1, n1, ngap):
            corrs[(i-nwin+1)/ngap] = kendalltau(x[(i+1-nwin):(i+1)], y[(i+1-nwin):(i+1)])[0] 
    else:
        raise ValueError(('The method "%s" is not supported. Please specify one of ' 
        'the following options: "pearsonr", "spearmanr" or "kendalltau"') % method)
    t = range(nwin-1, n1, ngap)
    return corrs, t
    
def online_corr(x, y, method="pearsonr", cutpoints_x=[], cutpoints_y=[], ngap=1):
    """ 
    Compute correlation on streaming sequence x and y (online method)
    
    The definition of Kendall’s tau that is used is:
    tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
    where P is the number of concordant pairs, Q the number of discordant pairs, 
    T the number of ties only in x, and U the number of ties only in y. 
    If a tie occurs for the same pair in both x and y, it is not added to either T or U.

    Parameters
    ----------   
    x: time series
    y: time series
    method: determin which type of correlation is computed. Accept method is 
        "pearsonr", "spearmanr" or "kendalltau"
    cutpoints_x: cutpoints of time series x in increasing order
    cutpoints_y: cutpoints of time series y in increasing order
    ngap: output correlation every ngap observations
    
    Returns
    -------
    corrs: correlations
    t: time indexes
    
    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 13, 1000)
    
    >>> corrs1, t1 = batch_corr(x, y, method="spearmanr")
    >>> corrs2, t2 = online_corr(x, y, method="spearmanr", cutpoints_x=range(1,10), cutpoints_y=range(1,13))
    >>> max(abs(corrs1 -corrs2)[2:]) < 1e-8
    True
    
    >>> corrs1, t1 = batch_corr(x, y, method="kendalltau")
    >>> corrs2, t2 = online_corr(x, y, method="kendalltau", cutpoints_x=range(1,10), cutpoints_y=range(1,13))
    >>> max(abs(corrs1 -corrs2)[2:]) < 1e-8
    True

    >>> corrs1, t1 = batch_corr(x, y, method="spearmanr", ngap=10)
    >>> corrs2, t2 = online_corr(x, y, method="spearmanr", cutpoints_x=range(1,10), cutpoints_y=range(1,13), ngap=10)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    
    >>> corrs1, t1 = batch_corr(x, y, method="kendalltau", ngap=10)
    >>> corrs2, t2 = online_corr(x, y, method="kendalltau", cutpoints_x=range(1,10), cutpoints_y=range(1,13), ngap=10)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    
    >>> x = np.random.randn(10000)
    >>> y = np.random.randn(10000)
    
    >>> corrs1, t1 = batch_corr(x, y, method="pearsonr", ngap=100)
    >>> corrs2, t2 = online_corr(x, y, method="pearsonr", ngap=100)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    
    >>> corrs1, t1 = batch_corr(x, y, method="pearsonr", ngap=1)
    >>> corrs2, t2 = online_corr(x, y, method="pearsonr", ngap=1)
    >>> max(abs(corrs1 -corrs2)[2:]) < 1e-8
    True
    """
    return online_corr_with_ngap(x, y, method, cutpoints_x, cutpoints_y, ngap)


def online_corr_with_ngap(x, y, method="pearsonr", cutpoints_x=[], cutpoints_y=[], ngap=10):
    """ 
    Compute correlation on streaming sequence x and y with ngap > 1
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    corrs = np.empty(n1//ngap)
    if method == "pearsonr":
        Sxy, Sxx, Syy, Sx, Sy = 0, 0, 0, 0, 0
        for i in range(n1):
            Sxy += x[i]*y[i]
            Sxx += x[i]**2
            Syy += y[i]**2
            Sx += x[i]
            Sy += y[i]
            if (i+1)%ngap == 0:
                Vx = Sxx - (Sx**2)/(i+1)
                Vy = Syy - (Sy**2)/(i+1)
                corrs[(i+1)/ngap - 1] = (Sxy - Sx*Sy/(i+1))/np.sqrt(Vx)/np.sqrt(Vy)
    elif method == "spearmanr":
        if cutpoints_x == [] or cutpoints_y == []:
            raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
        # construct M matrix
        m1 = len(cutpoints_x) + 1
        m2 = len(cutpoints_y) + 1
        M = np.zeros(shape=(m1, m2)) 
        miplus = np.zeros(m1)
        mplusj = np.zeros(m2)
        for i in range(n1):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y) 
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1
            if (i+1)%ngap == 0:
                corrs[(i+1)/ngap - 1] = spearmanr_by_matrix(M, miplus, mplusj, i+1)
    elif method == "kendalltau":
        if cutpoints_x == [] or cutpoints_y == []:
            raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
        # construct M matrix
        m1 = len(cutpoints_x) + 1
        m2 = len(cutpoints_y) + 1
        M = np.zeros(shape=(m1, m2)) 
        miplus = np.zeros(m1)
        mplusj = np.zeros(m2) 
        for i in range(n1):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y)
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1
            if (i+1)%ngap == 0:
                corrs[(i+1)/ngap - 1] =  kendalltau_by_matrix(M, miplus, mplusj, i+1)
    t = range(ngap-1, n1, ngap)
    return corrs, t

def online_mv_corr(x, y, nwin, method="pearsonr", cutpoints_x=[], cutpoints_y=[], ngap=1):
    """ 
    Compute correlation on streaming sequence x and y (online method)
    
    The definition of Kendall’s tau that is used is:
    tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
    where P is the number of concordant pairs, Q the number of discordant pairs, 
    T the number of ties only in x, and U the number of ties only in y. 
    If a tie occurs for the same pair in both x and y, it is not added to either T or U.

    Parameters
    ----------   
    x: time series
    y: time series
    nwin: window size
    method: determin which type of correlation is computed. Accept method is 
        "pearsonr", "spearmanr" or "kendalltau"
    cutpoints_x: cutpoints of time series x in increasing order
    cutpoints_y: cutpoints of time series y in increasing order
    ngap: output correlation every ngap observations
    
    Returns
    -------
    corrs: correlations computed at selected time indexes. The selected time indexes are nwin-1, nwin-1+ngap, ...
    t: selected time indexes  
    
    Examples
    --------
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 13, 1000)
    
    >>> corrs1, t1 = batch_mv_corr(x, y, nwin=100, method="spearmanr")
    >>> corrs2, t2 = online_mv_corr(x, y, nwin=100, method="spearmanr", cutpoints_x=range(1,10), cutpoints_y=range(1,13))
    >>> max(abs(corrs1 -corrs2)[2:]) < 1e-8
    True
    
    >>> corrs1, t1 = batch_mv_corr(x, y, nwin=100, method="kendalltau")
    >>> corrs2, t2 = online_mv_corr(x, y, nwin=100, method="kendalltau", cutpoints_x=range(1,10), cutpoints_y=range(1,13))
    >>> max(abs(corrs1 -corrs2)[2:]) < 1e-8
    True

    >>> corrs1, t1 = batch_mv_corr(x, y, nwin=100, method="spearmanr", ngap=3)
    >>> corrs2, t2 = online_mv_corr(x, y, nwin=100, method="spearmanr", cutpoints_x=range(1,10), cutpoints_y=range(1,13), ngap=3)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    
    >>> corrs1, t1 = batch_mv_corr(x, y, nwin=100, method="kendalltau", ngap=3)
    >>> corrs2, t2 = online_mv_corr(x, y, nwin=100, method="kendalltau", cutpoints_x=range(1,10), cutpoints_y=range(1,13), ngap=3)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    
    >>> x = np.random.randn(10000)
    >>> y = np.random.randn(10000)
    
    >>> corrs1, t1 = batch_mv_corr(x, y, nwin=1000, method="pearsonr", ngap=33)
    >>> corrs2, t2 = online_mv_corr(x, y, nwin=1000, method="pearsonr", ngap=33)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    
    >>> corrs1, t1 = batch_mv_corr(x, y, nwin=1000, method="pearsonr", ngap=1)
    >>> corrs2, t2 = online_mv_corr(x, y, nwin=1000, method="pearsonr", ngap=1)
    >>> max(abs(corrs1 -corrs2)) < 1e-8
    True
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    corrs = np.empty((n1 - nwin)//ngap + 1)
    
    # add first nwin observations and compute first correlation
    if method == "pearsonr":
        Sxy, Sxx, Syy, Sx, Sy = 0, 0, 0, 0, 0
        for i in range(nwin):
            Sxy += x[i]*y[i]
            Sxx += x[i]**2
            Syy += y[i]**2
            Sx += x[i]
            Sy += y[i]
        Vx = Sxx - (Sx**2)/nwin
        Vy = Syy - (Sy**2)/nwin
        corrs[0] = (Sxy - Sx*Sy/nwin)/np.sqrt(Vx)/np.sqrt(Vy)
    elif method == "spearmanr":
        if cutpoints_x == [] or cutpoints_y == []:
            raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
        # construct M matrix
        m1 = len(cutpoints_x) + 1
        m2 = len(cutpoints_y) + 1
        M = np.zeros(shape=(m1, m2)) 
        miplus = np.zeros(m1)
        mplusj = np.zeros(m2)
        for i in range(nwin):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y) 
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1
        corrs[0] = spearmanr_by_matrix(M, miplus, mplusj, nwin)
    elif method == "kendalltau":
        if cutpoints_x == [] or cutpoints_y == []:
            raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
        # construct M matrix
        m1 = len(cutpoints_x) + 1
        m2 = len(cutpoints_y) + 1
        M = np.zeros(shape=(m1, m2)) 
        miplus = np.zeros(m1)
        mplusj = np.zeros(m2) 
        for i in range(nwin):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y)
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1 
        corrs[0] = kendalltau_by_matrix(M, miplus, mplusj, nwin)    
            
    # compute correlation based on moving window 
    # (add the most recent observation and delete the oldest observation)        
    if method == "pearsonr":
        for i in range(nwin, n1):
            Sxy += x[i]*y[i] - x[i-nwin]*y[i-nwin]
            Sxx += x[i]**2 - x[i-nwin]**2
            Syy += y[i]**2 - y[i-nwin]**2
            Sx += x[i] - x[i-nwin]
            Sy += y[i] - y[i-nwin]            
            if (i - nwin + 1) % ngap == 0:
                Vx = Sxx - (Sx**2)/nwin
                Vy = Syy - (Sy**2)/nwin
                corrs[(i - nwin + 1)//ngap] = (Sxy - Sx*Sy/nwin)/np.sqrt(Vx)/np.sqrt(Vy)
    elif method == "spearmanr":
        for i in range(nwin, n1):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y) 
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1
            index3 = get_index(x[i-nwin], cutpoints_x)
            index4 = get_index(y[i-nwin], cutpoints_y)
            M[index3, index4] -= 1
            miplus[index3] -= 1
            mplusj[index4] -= 1
            if (i - nwin + 1) % ngap == 0:
                corrs[(i - nwin + 1)//ngap] = spearmanr_by_matrix(M, miplus, mplusj, nwin)
    elif method == "kendalltau":
        for i in range(nwin, n1):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y)
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1
            index3 = get_index(x[i-nwin], cutpoints_x)
            index4 = get_index(y[i-nwin], cutpoints_y)
            M[index3, index4] -= 1
            miplus[index3] -= 1
            mplusj[index4] -= 1
            if (i - nwin + 1) % ngap == 0:
                corrs[(i - nwin + 1)//ngap] = kendalltau_by_matrix(M, miplus, mplusj, nwin)
    t = range(nwin-1, n1, ngap)
    return corrs, t
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()