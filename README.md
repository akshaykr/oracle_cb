# oracle_cb
Experimentation for oracle based contextual bandit algorithms. 


******************************
## Installation

1. Clone repository
2. Instally python3, scipy, numpy, scikit-learn.
3. Fill in settings.py with your information. I recommend using full paths.
   * BASE\_DIR should point to the base of this repository.
   * DATA\_DIR should point to root/data/ directory.
   * REMOTE\_PATH\_TO\_PYTHON is only used if you want to run things on a cluster. 
   * REMOTE\_BASE\_DIR is only used if you want to run things on a cluster. 
3. Download and prepare datasets (MSLR, Yahoo, MQ2007, MQ2008). This is somewhat optional.
   * For MSLR: 
      * Visit https://www.microsoft.com/en-us/research/project/mslr/
      * Download MSLR-WEB30K dataset
      * Unpack it into settings.DATA\_DIR/mslr/ you should have 5 files named mslr30k_train<#>.txt where <#> is 1 through 5. This is different from the default directory structure of the dataset, so you will have to rename the files. 
      * ```$ python3 PreloadMSLR.py``` -- This will produce a file settings.DATA\_DIR/mslr/mslr30k_train.npz which is required for experiments.
   * For Yahoo:
      * You need to get the dataset, this is somewhat involved. The dataset is C14B here: https://webscope.sandbox.yahoo.com/catalog.php?datatype=c
      * Unpack it into settings.DATA\_DIR/yahoo/ you should have 6 files named set<#>.<$>.txt where <#> is either 1 or 2 and <$> is train, valid, or test.
      * ``` $ python3 PreloadYahoo.py``` -- This will produce a file settings.DATA\_DIR/yahoo/yahoo_big.npz which is required for experiments.

******************************
## Locally running an algorithm

1. Use Semibandits.py. 

   For contextual semibandits experiments (experiments for
   https://arxiv.org/abs/1502.05890), the entrypoint is
   Semibandits.py. This script can be executed with a few parameters.

   ```
   $ python3 Semibandits.py --T 1000 --dataset mslr30k --L 3 --I 0 --alg lin --param 0.1
   ```

   This will generate some output and then create a folder in
   root/results/.  That folder will have three files in it containing:
   the reward reported every 10 rounds, validation results on a held
   out dataset (which we currently ignore), and the total running time
   of the execution.

   Please see ./semibandits/make_parallel_script.py for configurations
   used in the experiments for the paper.

2. Use Bose.py

   For semiparametric contextual bandit experiments (experiments for
   https://arxiv.org/abs/1803.04204), the entrypoint is Bose.py. This
   script can be executed with a similar configuration

   ```
   $ python3 Bose.py --T 1000 --dataset semiparametric --d 10 --K 2 --iters 10 --param 0.1 --alg bose --feat pos --noise 0.5
   ```

   This will create a folder in ./results/ and write two files to the
   folder. One will contain the total reward reported every 10 rounds
   and the other will contain the total regret again reported every 10
   rounds.

   Please see ./semiparametric_cb/make_parallel_script.py for
   configurations used in the experiments for the paper.

3. Use LimeCB.py

   For CB model selection experiments (experiments for
   https://arxiv.org/abs/1906.00531), the entrypoint is
   LimeCB.py. This script can be executed with a few parameters as
   well

   ```
   $ python3 LimeCB.py --T 4000 --d 1000 --s 10 --K 2 --iters 10 --param 0.1 --alg limecb --noise 1.0 --base minimonster
   ```

   This will create a folder in ./results/ and write two files to the
   folder. One will contain the total reward reported every 10 rounds
   and the other will contain the total regret again reported every 10
   rounds.

   Please see ./model_selection_cb/make_scripts.py for configurations
   used in the experiments for the paper.

******************************
## Running on a cluster

1. Clone repository on the cluster. Locally update REMOTE_PATH_TO_PYTHON and REMOTE_BASE_DIR in settings.py
2. On the cluster, make sure that the globals in settings.py point to the right places. 
   * BASE_DIR=<location of code directory>
   * DATA_DIR=<location of the datasets>
3. Make sure you have the right .npz files in the DATA_DIR. See ContextIterators.py for the naming. For mslr you want to use the MSLR30k iterator, so you need to have DATA_DIR/mslr/mslr30k_train.npz. For yahoo you want to use the YahooContextIterator object, so you need to have DATA_DIR/yahoo/yahoo_big.npz. Put both mslr30k and yahoo on the cluster
4. Locally:

   ```
   cd <repository location>
   python3 parallel.py | parallel -S <number of threads>/<your login>@<your server>
   ```

   Use as many servers as you can but note that the process is memory intensive so parallel doesn't do a great job of allocating threads. 
   I was doing at most 4 jobs per machine. 
   If you want to change the parameters, edit the parallel.py file. 
5. The results will be in <repository location>/results/mslr_T=36000_L=3_e=0.1/ and <repository location>/code/results/yahoo_T=40000_L=2_e=0.5/


******************************
## Plotting results

1. Move the above to results directories locally.
2. 
```
python3 plotting_script.py --save
```
