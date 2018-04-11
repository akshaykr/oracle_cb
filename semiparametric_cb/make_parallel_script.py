import numpy as np

if __name__=='__main__':
    T = 2000
    ds = [10] ## , 15, 20, 50, 100]
    K = 5
    delta_vals = np.logspace(-3,1,20)
    eps_vals = np.logspace(-3,0,20)
    iters = 10
    start = 0

    for d in ds:
        for i in range(len(delta_vals)):
            for dataset in ['linear', 'semiparametric']:
                job_name = '%s_d=%d_K=%d_delta=%0.3f' % (dataset, d, K, delta_vals[i])
                jobfile = '../jobs/%s' % (job_name)
                f = open(jobfile, 'w')
                f.write("""#!/bin/bash

#SBATCH --job-name=%s
#SBATCH --output=../logs/%s.out  # output file
#SBATCH -e ../logs/%s.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=20:00:00         # Runtime in D-HH:MM:SS
#SBATCH --mem-per-cpu=60000    # Memory in MB per cpu allocated
cd ../
""" % (job_name, job_name, job_name))
                for feat in ['sphere', 'pos']:
                    for alg in ['minimonster', 'epsgreedy', 'linucb', 'bose']:
                        for noise in [0.1, 0.5]:
                            if alg == 'linucb' or alg=='bose':
                                f.write("/mnt/nfs/work1/akshay/akshay/anaconda3/bin/python3 -W ignore Bose.py --dataset %s --T %d --d %d --K %d --iters %d --param %0.3f --alg %s --feat %s --noise %0.1f" % (dataset, T, d, K, iters, delta_vals[i], alg, feat, noise))
                            else:
                                f.write("/mnt/nfs/work1/akshay/akshay/anaconda3/bin/python3 -W ignore Bose.py --dataset %s --T %d --d %d --K %d --iters %d --param %0.3f --alg %s --feat %s --noise %0.1f" % (dataset, T, d, K, iters, eps_vals[i], alg, feat, noise))
                            f.write("\n")
                f.write("""sleep 1
exit""")
                f.close()
