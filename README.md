# Online Algorithm for Nonparametric Correlations

Nonparametric correlations such as Spearman's rank correlation and Kendall's tau correlation are widely applied in scientific and engineering fields. Standard batch algorithms are generally too slow to handle real-world big data applications. They also require too much memory because all the data need to be stored in the memory before processing. We develop a novel online algorithm for computing nonparametric correlations in this package. The algorithm has O(1) time complexity and O(1) memory cost and is quite suitable for edge devices, where only limited memory and processing power are available. You can seek a balance between speed and accuracy by changing the number of cutpoints specified in the algorithm. The online algorithm can compute the nonparametric correlations 10 to 1,000 times faster than the corresponding batch algorithm, and it can compute them based either on all past observations or on fixed-size sliding windows.

Please see the [paper](https://arxiv.org/abs/1712.01521) for details.

### What is inside
File **corr.py** contains the following four main functions:
*batch_corr*: Compute correlation on streaming sequence x and y (batch method) 
*batch_mv_corr*: Compute correlation on streaming sequence x and y with moving windows (batch method)
*online_corr*: Compute correlation on streaming sequence x and y (online method)
*online_mv_corr*: Compute correlation on streaming sequence x and y with moving windows (online method) 

Folder simulate_online_npcorr contains simulation studies which compare the batch and the online nonparametric correlation algorithms.

Folder **Example_PHM** applies the batch and online nonparametric correlation algorithms to analyze the sensor data that were generated in industrial plant. The dataset comes from 2015 Prognostics and Health Management Society Competition.

### Citation
If you use this package in any way, please cite the following preprint.
```
@INPROCEEDINGS{9006483,
  author={W. {Xiao}},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)}, 
  title={Novel Online Algorithms for Nonparametric Correlations with Application to Analyze Sensor Data}, 
  year={2019},
  volume={},
  number={},
  pages={404-412},
  doi={10.1109/BigData47090.2019.9006483}}
```
### Contacts
Wei Xiao <wxiao@ncsu.edu>    
