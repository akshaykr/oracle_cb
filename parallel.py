import numpy as np
import settings

eps_vals = np.logspace(-3, 0, 10) ## np.arange(0.01,0.21,0.01)
mu_vals = np.logspace(-3, 1, 10) ## np.arange(0.1, 2.1, 0.1)
delta_vals = np.logspace(-3, 1, 10) ## np.arange(0.1, 2.1, 0.1)

iters = 10
algs = ['gb2', 'gb5', 'lin']

### MSLR30k ###
noise = 0.1
T = 36000
L = 3
for i in range(iters):
    for mu in mu_vals: 
        for alg in algs:
            print("""cd %s; %s Semibandits.py --dataset mslr30k --T %d --L %d --I %d --noise %0.3f --alg mini --learning_alg %s --param %0.3f > logs/mslr30k_mini_%s_%0.3f_%d""" % (settings.REMOTE_BASE_DIR, settings.REMOTE_PATH_TO_PYTHON, T, L, i, noise, alg, mu, alg, mu, i))
    for eps in eps_vals:
        for alg in algs:
            print("""cd %s; %s Semibandits.py --dataset mslr30k --T %d --L %d --I %d --noise %0.3f --alg eps --learning_alg %s --param %0.3f > logs/mslr30k_eps_%s_%0.3f_%d""" % (settings.REMOTE_BASE_DIR, settings.REMOTE_PATH_TO_PYTHON, T, L, i, noise, alg, mu, alg, mu, i))
    for delta in delta_vals:
        print("""cd %s; %s Semibandits.py --dataset mslr30k --T %d --L %d --I %d --noise %0.3f --alg lin --param %0.5f > logs/mslr30k_lin_%0.5f_%d""" % (settings.REMOTE_BASE_DIR, settings.REMOTE_PATH_TO_PYTHON, T, L, i, noise, delta, delta, i))


### Yahoo ###
noise = 0.5
T = 40000
L = 2
for i in range(iters):
    for mu in mu_vals: 
        for alg in algs:
            print("""cd %s; %s Semibandits.py --dataset yahoo --T %d --L %d --I %d --noise %0.3f --alg mini --learning_alg %s --param %0.3f > logs/yahoo_mini_%s_%0.3f_%d""" % (settings.REMOTE_BASE_DIR, settings.REMOTE_PATH_TO_PYTHON, T, L, i, noise, alg, mu, alg, mu, i))
    for eps in eps_vals:
        for alg in algs:
            print("""cd %s; %s Semibandits.py --dataset yahoo --T %d --L %d --I %d --noise %0.3f --alg eps --learning_alg %s --param %0.3f > logs/yahoo_eps_%s_%0.3f_%d""" % (settings.REMOTE_BASE_DIR, settings.REMOTE_PATH_TO_PYTHON, T, L, i, noise, alg, mu, alg, mu, i))
    for delta in delta_vals:
        print("""cd %s; %s Semibandits.py --dataset yahoo --T %d --L %d --I %d --noise %0.3f --alg lin --param %0.5f > logs/yahoo_lin_%0.5f_%d""" % (settings.REMOTE_BASE_DIR, settings.REMOTE_PATH_TO_PYTHON, T, L, i, noise, delta, delta, i))
