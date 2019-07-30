import numpy as np

if __name__=='__main__':
    T = 4000
    d = 1000
    s = 10
    K = 2

    delta_vals = np.logspace(-3,1,10)
    eps_vals = np.logspace(-3,1,10)

    iters = 20

    print("cd ../")
    for i in range(len(delta_vals)):
         print("python3 -W ignore LimeCB.py --T %d --d %d --s %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0 --base linucb" % (T, d, s, K, iters, eps_vals[i], 'limecb'))
         print("python3 -W ignore LimeCB.py --T %d --d %d --s %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0 --base linucb" % (T, d, s, K, iters, eps_vals[i], 'oracle'))
         print("python3 -W ignore LimeCB.py --T %d --d %d --s %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0 --base minimonster" % (T, d, s, K, iters, eps_vals[i], 'limecb'))
         print("python3 -W ignore LimeCB.py --T %d --d %d --s %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0 --base minimonster" % (T, d, s, K, iters, eps_vals[i], 'oracle'))
         print("python3 -W ignore LimeCB.py --T %d --d %d --s %d --K %d --iters %d --param %0.3f --alg %s --noise 1.0" % (T, d, s, K, iters, delta_vals[i], 'linucb'))

