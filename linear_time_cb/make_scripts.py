import numpy as np

if __name__=='__main__':
    T = 3000
    d = 10
    K = 10

    delta_vals = np.logspace(-2,1.0,13)
#     lr_vals = [0.1]
#     lr2_vals = [0.1]
    lr_vals = np.logspace(-1,1,3)
#     lr2_vals = np.logspace(-4,0,5)

    iters = 20

    print("cd ../")
    for i in range(len(delta_vals)):
        ## Run LinUCB
        print("python3 -W ignore LinearTimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], 'linucb'))
        ## Run Greedy+OLS
        print("python3 -W ignore LineartimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --lr 0 --lr2 0 --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], 'linlin'))
        ## Run RND+OLS
        print("python3 -W ignore LineartimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --lr 0 --lr2 0.1 --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], 'linlin'))
        for j in range(len(lr_vals)):
             ## Run Greedy+SGD
             print("python3 -W ignore LinearTimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --lr %s --lr2 %s --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], lr_vals[j], 0, 'linlin'))
             ## Run RND+SGD
             print("python3 -W ignore LinearTimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --lr %s --lr2 %s --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], lr_vals[j], lr_vals[j], 'linlin'))
