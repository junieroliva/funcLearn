funcLearn
=========
funcLearn is a matlab package for performing machine learning tasks when inputs,
and possibly outputs, are functions or distributions or sets. 

I've tried to write the code to balance both performance and readability.
The code is still very beta but hopefully will illustrate the methods in:
> - *Fast Function to Function Regression.*
Oliva, J., Neiswanger, W., Póczos, B., Schneider, J., & Xing, E.
International Conference on AI and Statistics (AISTATS), JMLR Workshop and Conference Proceedings, 2015.
> - *FuSSO: Functional Shrinkage and Selection Operator.*
Oliva, J., Póczos, B., Verstynen, T., Singh, A., Schneider, J., Yeh, F., Tseng, W.
International Conference on AI and Statistics (AISTATS), JMLR Workshop and Conference Proceedings, 2014.
> - *Fast Distribution to Real Regression.*
Oliva, J., Neiswanger, W., Póczos, B., Schneider, J., & Xing, E.
International Conference on AI and Statistics (AISTATS), JMLR Workshop and Conference Proceedings, 2014.
> - *Distribution to Distribution Regression.*
Oliva, J., Póczos, B., & Schneider, J.
International Conference on Machine Learning (ICML), JMLR Workshop and Conference Proceedings, 2013.

Installation
-------------
In order to use funcLearn you have to have [mtimesx](http://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support/content/mtimesx.m) installed and functioning correctly.
On unix machines you may need to use the following command to compile:
``` 
mex -largeArrayDims -DDEFINEUNIX mtimesx.c -lmwblas
```
Make sure that funcLearn is in you matlab path, either run:
``` 
fl_setup
```
or
``` 
addpath(genpath('/path/to/funcLearn/'))
```

Demos
--------
Please see the following scripts to illustrate how to do various functional based ML tasks:
>- `demos/getpcs_demo.m`
Shows how to use osfe to get projection coefficients to represent functions 
>- `demos/dist2real_demo.m`
Perform distribution to real regression task on synthetic data using double basis 
estimate as in *Fast Distribution to Real Regression.*
>- `demos/dist2dist_demo.m`
Perform distribution to distribution regression task on synthetic data using triple basis 
estimate as in *Fast Function to Function Regression.*
>- `demos/fusso_syndata_demo.m` and `demos/elastic_fusso_syndata_demo.m`
Perform many function to real regression task on synthetic data using FuSSO
estimate as in *FuSSO: Functional Shrinkage and Selection Operator.*