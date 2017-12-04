# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:42:52 2017

@author: wxiao
"""
import numpy as np

def window(size):
    return np.ones(size)/float(size)
    
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
    >>> from scipy.stats.stats import spearmanr
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
    # calculate spearmanr
    rank_col = np.array(rank_col)
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
    >>> from scipy.stats.stats import kendalltau
    >>> M = np.array([[4,3], [1,5]])
    >>> x = [2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5]
    >>> y = [-3, -3, -3 , -3 , 2, 2, 2, -3, 2, 2, 2, 2, 2]
    >>> cor1 = kendalltau_by_matrix(M, M.sum(axis=1), M.sum(axis=0), M.sum())
    >>> cor2 = kendalltau(x, y)[0]
    >>> round(cor1,6) == round(cor2, 6)
    True
    """
    m1, m2 = M.shape[0], M.shape[1]
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
    Q = n*(n-1)/2 - P - T - U - NC
    return (P - Q) / np.sqrt((P + Q + T) * (P + Q + U)) 
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()