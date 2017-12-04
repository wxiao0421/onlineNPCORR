# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 00:35:15 2017

@author: wxiao
"""
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import cPickle as pickle

import os, sys
sys.path.insert(0, os.path.abspath('..'))
from corr import batch_mv_corr, online_mv_corr

# read in PHM data of plant 1
df = pd.read_csv("./PHMtrain/plant1.csv")
df = df.fillna(method = "backfill")

print("m1_R1")
print("unique values:")
print(sorted(df["m1_R1"].unique()))
print("length=%d" % len(df["m1_R1"].unique()))

print("m1_S1")
print("unique values:")
print(sorted(df["m1_S1"].unique()))
print("length=%d" % len(df["m1_S1"].unique()))

print("m1_R3")
print("unique values:")
print(sorted(df["m1_R3"].unique()))
print("length=%d" % len(df["m1_R3"].unique()))

print("m1_S3")
print("unique values:")
print(sorted(df["m1_S3"].unique()))
print("length=%d" % len(df["m1_S3"].unique()))

#sampling rate: one observation per 15 min
#window size one year
nwin = 4*24*365

## np corr of R1 vs S1 (keep all cutpoints for S1)
run_time = {}
result = {}

x = df["m1_R1"]
y = df["m1_S1"]
cutpoints_x = [670,675,685,695,705,715,725,760,775]
cutpoints_y = sorted(df["m1_S1"].unique())[1:]

# spearmanr
method = "spearmanr"
start = timer()
corrs1, t1 = batch_mv_corr(x, y, nwin=nwin, method=method, ngap=1)
end = timer()
run_time["batch " + method] = end - start
result["batch " + method] = corrs1
start = timer()
corrs2, t2 = online_mv_corr(x, y, nwin=nwin, method=method, cutpoints_x=cutpoints_x, cutpoints_y=cutpoints_y, ngap=1)
end = timer()
run_time["online " + method] = end - start
result["online " + method] = corrs2 

# kendalltau
method = "kendalltau"
start = timer()
corrs1, t1 = batch_mv_corr(x, y, nwin=nwin, method=method, ngap=1)
end = timer()
run_time["batch " + method] = end - start
result["batch " + method] = corrs1
start = timer()
corrs2, t2 = online_mv_corr(x, y, nwin=nwin, method=method, cutpoints_x=cutpoints_x, cutpoints_y=cutpoints_y, ngap=1)
end = timer()
run_time["online " + method] = end - start
result["online " + method] = corrs2 


with open(r'./result/result_R1_S1.pickle', 'wb') as handle:
    pickle.dump((result, run_time), handle)
    
## np corr of R1 vs S1 (keep 19 cutpoints for S1)
run_time = {}
result = {}

x = df["m1_R1"]
y = df["m1_S1"]
cutpoints_x = [670,675,685,695,705,715,725,760,775]
#cutpoints_y = sorted(df["m1_S1"].unique())
cutpoints_y = list(df["m1_S1"].quantile(np.linspace(0, 1, num=20, endpoint=False)[1:]))

# spearmanr
method = "spearmanr"
start = timer()
corrs1, t1 = batch_mv_corr(x, y, nwin=nwin, method=method, ngap=1)
end = timer()
run_time["batch " + method] = end - start
result["batch " + method] = corrs1
start = timer()
corrs2, t2 = online_mv_corr(x, y, nwin=nwin, method=method, cutpoints_x=cutpoints_x, cutpoints_y=cutpoints_y, ngap=1)
end = timer()
run_time["online " + method] = end - start
result["online " + method] = corrs2 

# kendalltau
method = "kendalltau"
start = timer()
corrs1, t1 = batch_mv_corr(x, y, nwin=nwin, method=method, ngap=1)
end = timer()
run_time["batch " + method] = end - start
result["batch " + method] = corrs1
start = timer()
corrs2, t2 = online_mv_corr(x, y, nwin=nwin, method=method, cutpoints_x=cutpoints_x, cutpoints_y=cutpoints_y, ngap=1)
end = timer()
run_time["online " + method] = end - start
result["online " + method] = corrs2 


with open(r'./result/result_R1_S1_fast.pickle', 'wb') as handle:
    pickle.dump((result, run_time), handle)
    
    
## np corr of R3 vs S3
run_time = {}
result = {}

x = df["m1_R3"]
y = df["m1_S3"]
cutpoints_x = [25,30,35,45,70,95,110]
cutpoints_y = [0.5,2,4.5,10,20.5,25,28.5,40,68.5,80,100.5]

# spearmanr
method = "spearmanr"
start = timer()
corrs1, t1 = batch_mv_corr(x, y, nwin=nwin, method=method, ngap=1)
end = timer()
run_time["batch " + method] = end - start
result["batch " + method] = corrs1
start = timer()
corrs2, t2 = online_mv_corr(x, y, nwin=nwin, method=method, cutpoints_x=cutpoints_x, cutpoints_y=cutpoints_y, ngap=1)
end = timer()
run_time["online " + method] = end - start
result["online " + method] = corrs2 

# kendalltau
method = "kendalltau"
start = timer()
corrs1, t1 = batch_mv_corr(x, y, nwin=nwin, method=method, ngap=1)
end = timer()
run_time["batch " + method] = end - start
result["batch " + method] = corrs1
start = timer()
corrs2, t2 = online_mv_corr(x, y, nwin=nwin, method=method, cutpoints_x=cutpoints_x, cutpoints_y=cutpoints_y, ngap=1)
end = timer()
run_time["online " + method] = end - start
result["online " + method] = corrs2 


with open('./result/result_R3_S3.pickle', 'wb') as handle:
    pickle.dump((result, run_time), handle)
    