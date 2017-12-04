# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:54:32 2017

@author: wxiao
"""
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr, kendalltau

def window(size):
    return np.ones(size)/float(size)
    
def corr(x, y, method="pearsonr"):
    """ 
    Compute pearson correlation on time series x and y
    """
    if method == "pearsonr":
        return pearsonr(x, y)[0]
    elif method == "spearmanr":
        return spearmanr(x, y)[0]
    elif method == "kendalltau":
        return kendalltau(x, y)[0]
    
def batch_corr(x, y, method="pearsonr", ngap=1):
    """ 
    Compute correlation on streaming sequence x and y (batch method)
    
    x: time series
    y: time series
    ngap: output correlation every ngap observations.
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    corrs = np.empty(n1)
    if method == "pearsonr":
        for i in range(0, len(x), ngap):
            corrs[i] = pearsonr(x[:(i+1)], y[:(i+1)])[0]
    elif method == "spearmanr":
        for i in range(0, len(x), ngap):
            corrs[i] = spearmanr(x[:(i+1)], y[:(i+1)])[0] 
    elif method == "kendalltau":
        for i in range(0, len(x), ngap):
            corrs[i] = kendalltau(x[:(i+1)], y[:(i+1)])[0] 
    else:
        raise ValueError(('The method "%s" is not supported. Please specify one of ' 
        'the following options: "pearsonr", "spearmanr" or "kendalltau"') % method)
    return corrs
    
def online_corr(x, y, method="pearsonr", cutpoints_x=[], cutpoints_y=[], ngap=1):
    """ 
    Compute correlation on streaming sequence x and y (online method)
    
    The definition of Kendall’s tau that is used is:
    tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
    where P is the number of concordant pairs, Q the number of discordant pairs, 
    T the number of ties only in x, and U the number of ties only in y. 
    If a tie occurs for the same pair in both x and y, it is not added to either T or U.
    """
    if ngap == 1:
        n1 = len(x)
        n2 = len(y)
        assert (n1 == n2),"The length of time series x and time series y must be equal!"
        corrs = np.empty(n1)
        if method == "pearsonr":
            Sxy, Sxx, Syy, Sx, Sy = 0, 0, 0, 0, 0
            for i in range(n1):
                Sxy += x[i]*y[i]
                Sxx += x[i]**2
                Syy += y[i]**2
                Sx += x[i]
                Sy += y[i]
                Vx = Sxx - (Sx**2)/(i+1)
                Vy = Syy - (Sy**2)/(i+1)
                corrs[i] = (Sxy - Sx*Sy/(i+1))/np.sqrt(Vx)/np.sqrt(Vy)
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
                corrs[i] = spearmanr_by_matrix(M, miplus, mplusj, i)
        elif method == "kendalltau":
            if cutpoints_x == [] or cutpoints_y == []:
                raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
            # construct M matrix
            m1 = len(cutpoints_x) + 1
            m2 = len(cutpoints_y) + 1
            M = np.zeros(shape=(m1, m2)) 
            miplus = np.zeros(m1)
            mplusj = np.zeros(m2) 
            P, Q, T, U, NC = 0, 0, 0, 0, 0
            for i in range(n1):
                index1 = get_index(x[i], cutpoints_x)
                index2 = get_index(y[i], cutpoints_y)
                P += M[:index1, :index2].sum() + M[(index1+1):, (index2+1):].sum()
                T += miplus[index1] - M[index1, index2]
                U += mplusj[index2] - M[index1, index2]
                NC += M[index1, index2]
                Q = (i+1)*i/2 - P - T - U - NC
                M[index1, index2] += 1
                miplus[index1] += 1
                mplusj[index2] += 1 
                corrs[i] = (P - Q) / np.sqrt((P + Q + T) * (P + Q + U))
        return corrs
    else:
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
        for i in range(0, n1, ngap):
            Sxy += x[i]*y[i]
            Sxx += x[i]**2
            Syy += y[i]**2
            Sx += x[i]
            Sy += y[i]
            if (i+1)%ngap == 0:
                Vx = Sxx - (Sx**2)/(i+1)
                Vy = Syy - (Sy**2)/(i+1)
                corrs[(i+1)/ngap] = (Sxy - Sx*Sy/(i+1))/np.sqrt(Vx)/np.sqrt(Vy)
    elif method == "spearmanr":
        if cutpoints_x == [] or cutpoints_y == []:
            raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
        # construct M matrix
        m1 = len(cutpoints_x) + 1
        m2 = len(cutpoints_y) + 1
        M = np.zeros(shape=(m1, m2)) 
        miplus = np.zeros(m1)
        mplusj = np.zeros(m2)
        for i in range(0, n1, ngap):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y) 
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1
            corrs[(i+1)/ngap] = spearmanr_by_matrix(M, miplus, mplusj, i)
    elif method == "kendalltau":
        if cutpoints_x == [] or cutpoints_y == []:
            raise ValueError("Both parameters cutpoints_x and cutpoints_y need to be specified")
        # construct M matrix
        m1 = len(cutpoints_x) + 1
        m2 = len(cutpoints_y) + 1
        M = np.zeros(shape=(m1, m2)) 
        miplus = np.zeros(m1)
        mplusj = np.zeros(m2) 
        P, Q, T, U, NC = 0, 0, 0, 0, 0
        for i in range(n1):
            index1 = get_index(x[i], cutpoints_x)
            index2 = get_index(y[i], cutpoints_y)
            M[index1, index2] += 1
            miplus[index1] += 1
            mplusj[index2] += 1 
            
            P += M[:index1, :index2].sum() + M[(index1+1):, (index2+1):].sum()
            T += miplus[index1] - M[index1, index2]
            U += mplusj[index2] - M[index1, index2]
            NC += M[index1, index2]
            Q = (i+1)*i/2 - P - T - U - NC
            corrs[i] = (P - Q) / np.sqrt((P + Q + T) * (P + Q + U))
    return corrs
        
def ccf(x, y, min_lag=-10, max_lag=10):
    """ 
    Compute cross correlation of time series x and y from min_lag to max_lag 
    (based on pearson correlation). r[lag] = corr(x[t-lag], y[t]), where lag = 
    min_lag, min_lag + 1, ..., max_lag.
    
    Parameters
    ----------
    x: time series
    y: time series
    min_lag : int, default -10
    max_lag : int, default 10

    Returns
    ----------
    a dictionary with four keys. "corrs": correlation coefficient corresponding to the lags);
    "lags": corresponding lags; "lb": lower bound; "ub": upper bound.
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    assert (min_lag <= max_lag),"min_lag must less than or equal to max_lag!" 
    nlags = max_lag - min_lag + 1       
    corrs = np.empty(nlags)
    for k, lag in enumerate(range(min_lag, (max_lag+1))):
        if lag == 0:
            corrs[k] = pearsonr(x, y)[0]
        if lag < 0:
            corrs[k] = pearsonr(x[(-lag):], y[:lag])[0]
        if lag > 0:
            corrs[k] = pearsonr(x[:(-lag)], y[lag:])[0]            
    return {"corrs": corrs, "lags": range(min_lag, (max_lag+1)), 
                "lb": np.repeat(-1/np.sqrt(n1), nlags), 
                "ub": np.repeat(1/np.sqrt(n1), nlags)} 

def npccf(x, y, method="spearmanr", min_lag=-10, max_lag=10):
    """ Compute cross correlation of time series x and y from min_lag to max_lag 
    (based on nonparametric correlation). r(lag) = corr(x[t-lag], y[t]).
    
    Parameters
    ----------
    x: time series
    y: time series
    method: "spearmanr" or "kendalltau"
    min_lag : int, default -10
    max_lag : int, default 10

    Returns
    ----------
    a dictionary with keys "corrs" (correlation coefficient corresponding to the lags),
    "lags" (corresponding lags), "lb" (lower bound) and "ub" (upper bound).
    """
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    assert (min_lag <= max_lag),"min_lag must less than or equal to max_lag!" 
    nlags = max_lag - min_lag + 1       
    corrs = np.empty(nlags)
    if method == "spearmanr":
        for k, lag in enumerate(range(min_lag, (max_lag+1))):
            if lag == 0:
                corrs[k] = spearmanr(x, y)[0]
            if lag < 0:
                corrs[k] = spearmanr(x[(-lag):], y[:lag])[0]
            if lag > 0:
                corrs[k] = spearmanr(x[:(-lag)], y[lag:])[0]
    elif method == "kendalltau":
        for k, lag in enumerate(range(min_lag, (max_lag+1))):
            if lag == 0:
                corrs[k] = kendalltau(x, y)[0]
            if lag < 0:
                corrs[k] = kendalltau(x[(-lag):], y[:lag])[0]
            if lag > 0:
                corrs[k] = kendalltau(x[:(-lag)], y[lag:])[0] 
    else:
        raise ValueError("The method %s is not supported." % method)
    return {"corrs": corrs, "lags": range(min_lag, (max_lag+1)), 
                "lb": np.repeat(-1/np.sqrt(n1), nlags), 
                "ub": np.repeat(1/np.sqrt(n1), nlags)}

def batch_npccf(x, y, method="spearmanr", min_lag=-10, max_lag=10):
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    assert (min_lag <= max_lag),"min_lag must less than or equal to max_lag!" 
    nlags = max_lag - min_lag + 1
    for t in range(n1):
        if method == "spearmanr":
            for k, lag in enumerate(range(min_lag, (max_lag+1))):
                if lag == 0:
                    corrs[k] = spearmanr(x[:(t+1)], y[:(t+1)])[0]
                if lag < 0:
                    corrs[k] = spearmanr(x[(-lag):(t+1)], y[:(t+1+lag)])[0]
                if lag > 0:
                    corrs[k] = spearmanr(x[:(t+1-lag)], y[lag:(t+1)])[0]        
        
    
def get_index(x, cutpoints_x):
    """
    Apply binary search to find index for x with respect to cutpoints_x.
    Assume cutpoints_x = [q_1, q_2, ..., q_k], where q_1<q_2<...<q_k. We 
    furthur define q_0 = -Inf and q_(k+1) = Inf.If x >= qi and x < q(i+1), 
    return i. 
    
    Parameters
    ----------
    x: a numerical value
    
    cutpoints_x: an array of cutpoints in increasing order
        
    Examples
    --------
    >>> get_index(5, range(10))
    6
    >>> get_index(4, [-2,1,7, 10])
    2
    """
    
    length = len(cutpoints_x)
    first = 0
    last = length -1
    if x <  cutpoints_x[first]:
        return 0
    if x >= cutpoints_x[last]:
        return length
    while last - first >= 2:        
        midpoint = (first + last)/2
        if x >= cutpoints_x[midpoint]:
            first = midpoint
        else:
            last = midpoint
    return first + 1
    
    
    
def spearmanr_by_matrix(M, miplus, mplusj, n):
    """
    Calculate spearman rank correlation based on working matrix M, row sums miplus,
    col sums mplusj and number of observations n
        
    Examples
    --------
    >>> import scipy
    >>> M = np.array([[4,3], [1,5]])
    >>> x = [2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5]
    >>> y = [-3, -3, -3 , -3 , 2, 2, 2, -3, 2, 2, 2, 2, 2]
    >>> xr = scipy.stats.rankdata(x)
    >>> yr = scipy.stats.rankdata(y)
    >>> cor1 = spearmanr_by_matrix(M, M.sum(axis=1), M.sum(axis=0), M.sum())
    >>> cor2 = spearmanr(x, y)[0]
    >>> cor3 = np.sum((xr - np.mean(xr))*(yr- np.mean(yr)))/np.sqrt(np.sum((xr - np.mean(xr))**2)*np.sum((yr- np.mean(yr))**2))
    >>> round(cor1,6) == round(cor2, 6)
    True
    >>> round(cor1,6) == round(cor3, 6)
    True
    """
    # iteratively calculate the average rank of each row
    rank_row = []
    i = 0
    for r in miplus:
        if r == 0:
            rank_row.append(i)
        else:
            rank_row.append((i + 1 + i + r)/2.0)
            i += r
    rank_row = np.array(rank_row) 
    # iteratively calculate the average rank of each column       
    rank_col = []
    i = 0
    for r in mplusj:
        if r == 0:
            rank_col.append(i)
        else:
            rank_col.append((i + 1 + i + r)/2.0)
            i += r
    rank_col = np.array(rank_col)
    # calculate spearmanr
    rbar = (1 + n)/2.0
    rank_row_demean = rank_row - rbar
    rank_col_demean = rank_col - rbar
    d1 = np.sqrt(np.sum(rank_row_demean**2*miplus))
    d2 = np.sqrt(np.sum(rank_col_demean**2*mplusj))
    return np.sum(np.dot(np.dot(np.diag(rank_row_demean/d1), M), np.diag(rank_col_demean/d2))) 

def kendalltau_by_matrix(M, miplus, mplusj, n):
    """
    Calculate kendalltau correlation based on working matrix M, row sums miplus,
    col sums mplusj and number of observations n
        
    Examples
    --------
    >>> import scipy
    >>> M = np.array([[4,3], [1,5]])
    >>> x = [2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5]
    >>> y = [-3, -3, -3 , -3 , 2, 2, 2, -3, 2, 2, 2, 2, 2]
    >>> cor1 = kendalltau_by_matrix(M, M.sum(axis=1), M.sum(axis=0), M.sum())
    >>> cor2 = kendalltau(x, y)[0]
    >>> round(cor1,6) == round(cor2, 6)
    True
    """
    m1, m2 = M.shape[0], M.shape[0]
    # P the number of concordant pairs
    N = np.zeros((m1, m2))
    for i in range(1, m1):
        N[i,1:] = M[(i-1),:-1].cumsum()
    for i in range(1, m1):
        N[i,] += N[(i-1),]
    P = np.sum(M*N)
    # T the number of ties only in x
    T = 0
    for i in range(m1):
        T += (miplus[i]**2 - np.sum(M[i,:]**2))/2
    # U the number of ties only in y
    U = 0
    for j in range(m2):
        U += (mplusj[j]**2 - np.sum(M[:,j]**2))/2
    # NC the number of ties in both x and y
    NC = np.sum(M * (M-1))/2
    # Q the number of discordant pairs
    Q = (n+1)*n/2 - P - T - U - NC
    return (P - Q) / np.sqrt((P + Q + T) * (P + Q + U))   
     
def fast_npccf(x, y, cutpoints_x, cutpoints_y, method="spearmanr", 
                   min_lag=-10, max_lag=10):
    """
    Compute nonparametric ccf with fast algorithm (approximation) 
    
    Parameters
    ----------
    x: a numerical value
    
    cutpoints_x: an array of cutpoints in increasing order
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randint(0, 10, 1000)
    >>> y = np.random.randint(0, 10, 1000)
    >>> cor1 = npccf(x, y, method="spearmanr", min_lag=0, max_lag=10)
    >>> cor2 = fast_npccf(x, y, range(1,10), range(1,10), method="spearmanr",  min_lag=0, max_lag=10)
    >>> cor3 = np.sum((xr - np.mean(xr))*(yr- np.mean(yr)))/np.sqrt(np.sum((xr - np.mean(xr))**2)*np.sum((yr- np.mean(yr))**2))
    >>> round(cor1,6) == round(cor2, 6)
    True
    """ 
    n1 = len(x)
    n2 = len(y)
    assert (n1 == n2),"The length of time series x and time series y must be equal!"
    assert (min_lag <= max_lag),"Parameter min_lag must be less than or equal to parameter max_lag!" 
    nlags = max_lag - min_lag + 1       
    corrs = np.empty(nlags)
    cutpoints_x = np.sort(cutpoints_x)
    cutpoints_y = np.sort(cutpoints_y)
    # construct M matrix
    m1 = len(cutpoints_x) + 1
    m2 = len(cutpoints_y) + 1
    Mvec = np.zeros(shape=(m1, m2, nlags))
    for k, lag in enumerate(range(min_lag, (max_lag+1))):    
        if lag == 0:
            xypairs = zip(x, y)
        if lag < 0:
            xypairs = zip(x[(-lag):], y[:lag])
        if lag > 0:
            xypairs = zip(x[:(-lag)], y[lag:])
        for xi, yi in xypairs:
            index1 = get_index(xi, cutpoints_x)
            index2 = get_index(yi, cutpoints_y) 
            Mvec[:,:,k][index1, index2] += 1
        miplus = np.sum(Mvec[:,:,k], axis=1)
        mplusj = np.sum(Mvec[:,:,k], axis=0)
        n = len(xypairs)
        corrs[k] = spearmanr_by_matrix(Mvec[:,:,k], miplus, mplusj, n)                               
    return {"corrs": corrs, "lags": range(min_lag, (max_lag+1)), 
                "lb": np.repeat(-1/np.sqrt(n1), nlags), 
                "ub": np.repeat(1/np.sqrt(n1), nlags)}    
    

                
## generate x, y time series
#nsamples = 10000000
#win_size = 3
#nlag = 5
#sigma = 0.1
#
#x = np.random.normal(0, 1, nsamples + nlag) # generate independent x
#z = np.random.normal(0, 1, nsamples + nlag + ( win_size -1)) 
#y = np.convolve(z, window(win_size), "valid") # generate independent y
#y[nlag:] = y[nlag:] + x[:(-nlag)]
#x = x[nlag:]
#y = y[nlag:]
#
#out1 = ccf(x, y, min_lag=0, max_lag=10)
#out2 = npccf(x, y, method = "spearmanr", min_lag=0, max_lag=10)
#out3 = npccf(x, y, method = "kendalltau", min_lag=0, max_lag=10)

if __name__ == "__main__":
    import doctest
    doctest.testmod() 