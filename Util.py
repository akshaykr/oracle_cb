import numpy as np
import scipy.misc
import scipy.optimize
import itertools

def mixture_decomp(w,l):
    k = w.shape[0]
    M = np.zeros((np.round(scipy.misc.comb(k, l)), k))
    i = 0
    for it in itertools.combinations(range(k), l):
        M[i,it] = 1
        i += 1
    
    (x,res) = scipy.optimize.nnls(M.T, l*w)
    return (M, x)
    
def slate_eq_ind(A,B):
    if (np.array(A) == np.array(B)).all():
        return 1
    return 0
    

def training_points(T, schedule='exp'):
    out = []
    i = 4
    while True:
        out.append(int(np.sqrt(2)**i))
        i += 1
        if np.sqrt(2)**i > T:
            break
    if schedule == 'lin':
        ## Switch to linear training schedule with step 100
        out = [50]
        i=1
        while True:
            out.append(int(i*100))
            i+=1
            if i*100>T:
                break
    return (out)
