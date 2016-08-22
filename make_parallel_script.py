import numpy as np

if __name__=='__main__':
    T = 36000
    mu_vals = np.logspace(-3,1,10)
    eps_vals = np.logspace(-3,0,10)
    delta_vals = np.logspace(-3,1,10)
    iters = 10
    start = 0

    L = 3
    dataset = "mslr30k"
    noise = 0.1
    learning_algs = ["gb2", "gb5", "lin"]

    for i in range(start, start+iters):
        alg = "mini"
        for l in learning_algs:
            for mu in mu_vals:
                print("cd ~/projects/semibandits/code/; /home/REDMOND/akshaykr/anaconda3/bin/python3 Semibandits.py --dataset %s --T %d --L %d --I %d --noise %f --alg %s --learning_alg %s --param %0.3f > logs/%s_%s_%s_%0.3f_%d" % (dataset, T, L, i, noise, alg, l, mu, dataset, alg, l, mu, i))
    
        alg = "eps"
        for l in learning_algs:
            for eps in eps_vals:
                print("cd ~/projects/semibandits/code/; /home/REDMOND/akshaykr/anaconda3/bin/python3 Semibandits.py --dataset %s --T %d --L %d --I %d --noise %f --alg %s --learning_alg %s --param %0.3f > logs/%s_%s_%s_%0.3f_%d" % (dataset, T, L, i, noise, alg, l, eps, dataset, alg, l, eps, i))

        alg = "lin"
        for delta in delta_vals:
            print("cd ~/projects/semibandits/code/; /home/REDMOND/akshaykr/anaconda3/bin/python3 Semibandits.py --dataset %s --T %d --L %d --I %d --noise %f --alg %s --param %0.3f > logs/%s_%s_%0.5f_%d" % (dataset, T, L, i, noise, alg, delta, dataset, alg, delta, i))
