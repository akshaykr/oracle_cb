# oracle_cb
Experimentation for oracle based contextual bandit algorithms. 


******************************
** Installation
******************************
1. Clone repository
2. Instally python3, scipy, numpy, scikit-learn.
3. Fill in settings.py with your information. I recommend using full paths.
   BASE_DIR should point to the base of this repository
   DATA_DIR should point to root/data/ directory
   REMOTE_PATH_TO_PYTHON is only used if you want to run things on a cluster. 
3. Download and prepare datasets (MSLR, Yahoo, MQ2007, MQ2008). This is somewhat optional.
   a. For MSLR: @akshaykr fill me in
   b. For Yahoo: @akshaykr fill me in

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


HOW TO RUN AN EXPERIMENT
1. Clone repository on the cluster.
2. Set up datasets
   Create a python file called settings.py with two globals
   BASE_DIR=<location of code directory>
   DATA_DIR=<location of the datasets>
   Make sure you have the right .npz files in the DATA_DIR.
   See ContextIterators.py for the naming.
   For mslr you want to use the MSLR30k iterator.
   So you need to have DATA_DIR/mslr/mslr30k_train.npz
   For yahoo you want to use the YahooContextIterator object.
   So you need to have DATA_DIR/yahoo/yahoo_big.npz
   Put both mslr30k and yahoo on the cluster
3. Create results directory on the cluster.
   mkdir semibandits/code/results/
4. Locally:
   cd semibandits/code
   python3 parallel.py | parallel -S 4/<your login>@msrnyc-##.corp.microsoft.com
   Use all of the servers if you can and wait like 4 days.
5. The results will be in semibandits/code/results/mslr_T=36000_L=3_e=0.1/ and semibandits/code/results/yahoo_T=40000_L=2_e=0.5/

HOW TO PLOT RESULTS.
1. Move the above to results directories locally.
2. python3 plotting_script.py --save

