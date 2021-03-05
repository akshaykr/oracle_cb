import numpy as np

if __name__=='__main__':
    T = 10000
    d = 10
    K = 4

    delta_vals = np.logspace(-3,1,5)
    lr_vals = [0.1]
    lr2_vals = [0.1]
#     lr_vals = np.logspace(-4,0,5)
#     lr2_vals = np.logspace(-4,0,5)

    iters = 20

    print("cd ../")
    for i in range(len(delta_vals)):
         print("python3 -W ignore LinearTimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], 'linucb'))
         for j in range(len(lr_vals)):
             print("python3 -W ignore LinearTimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --lr %s --lr2 %s --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], lr_vals[j], 0, 'linlin'))
             for k in range(len(lr2_vals)):
                 print("python3 -W ignore LinearTimeUCB.py --T %d --d %d --K %d --iters %d --param %0.3f --lr %s --lr2 %s --alg %s --noise 1.0" % (T, d, K, iters, delta_vals[i], lr_vals[j], lr2_vals[k], 'linlin'))
