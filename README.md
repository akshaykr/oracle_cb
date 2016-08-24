# oracle_cb
Experimentation for oracle based contextual bandit algorithms. 


******************************
** Installation
******************************
1. Clone repository
2. Instally python3, scipy, numpy, scikit-learn.
3. Fill in settings.py with your information. I recommend using full paths.
   BASE_DIR should point to the base of this repository.
   DATA_DIR should point to root/data/ directory.
   REMOTE_PATH_TO_PYTHON is only used if you want to run things on a cluster. 
   REMOTE_BASE_DIR is only used if you want to run things on a cluster. 
3. Download and prepare datasets (MSLR, Yahoo, MQ2007, MQ2008). This is somewhat optional.
   a. For MSLR: 
      i. Visit https://www.microsoft.com/en-us/research/project/mslr/
      ii. Download MSLR-WEB30K dataset
      iii. unpack it into settings.DATA_DIR/mslr/ you should have 5 files named mslr30k_train<*>.txt where <*> is 1 through 5.
      iv. python3 PreloadMSLR.py
          This will produce a file settings.DATADIR/mslr/mslr30k_train.npz which is required for experiments.
   b. For Yahoo: @akshaykr fill me in
      i. You need to get the dataset, this is somewhat involved. The dataset is C14B here: https://webscope.sandbox.yahoo.com/catalog.php?datatype=c
      ii. unpack it into settings.DATA_DIR/yahoo/ you should have 6 files named set<1>.<2>.txt where <1> is either 1 or 2 and <2> is train, valid, or test.
      iii. python3 PreloadYahoo.py
          This will produce a file settings.DATADIR/yahoo/yahoo_big.npz which is required for experiments.

******************************
** Locally running an algorithm
******************************
1. Use Semibandits.py. 
   It can be run as a script with a few parameters.
   $ python3 Semibandits.py --T 1000 --dataset mslr30k --L 3 --I 0 --alg lin --param 0.1

   This will generate some output and then create a folder in
   root/results/.  That folder will have three files in it containing:
   the reward reported every 10 rounds, validation results on a held
   out dataset (which we currently ignore), and the total running time
   of the execution.


******************************
** Running on a cluster
******************************
1. Clone repository on the cluster. Locally update REMOTE_PATH_TO_PYTHON and REMOTE_BASE_DIR in settings.py
2. On the cluster, make sure that the globals in settings.py point to the right places. 
      BASE_DIR=<location of code directory>
      DATA_DIR=<location of the datasets>
3. Make sure you have the right .npz files in the DATA_DIR. See ContextIterators.py for the naming. For mslr you want to use the MSLR30k iterator, so you need to have DATA_DIR/mslr/mslr30k_train.npz. For yahoo you want to use the YahooContextIterator object, so you need to have DATA_DIR/yahoo/yahoo_big.npz. Put both mslr30k and yahoo on the cluster
4. Locally:
   cd <repository location>
   python3 parallel.py | parallel -S 4/<your login>@msrnyc-##.corp.microsoft.com
   Use all of the servers if you can and wait like 4 days.
   If you want to change the parameters, edit the parallel.py file. 
5. The results will be in <repository location>/results/mslr_T=36000_L=3_e=0.1/ and <repository location>/code/results/yahoo_T=40000_L=2_e=0.5/

******************************
** Plotting results
******************************
1. Move the above to results directories locally.
2. python3 plotting_script.py --save

