import numpy as np
import scipy.linalg
from Semibandits import Semibandit

class LinUCB(Semibandit):
    """
    Implementation of Semibandit Linucb.
    This algorithm of course only works if features are available 
    in the SemibanditSim object. 

    It must also have a fixed dimension featurization and expose
    B.d as an instance variable.
    """
    def __init__(self, B):
        """
        Initialize a linUCB object.
        """
        self.B = B

    def init(self, T, params={}):
        """
        Initialize the regression target and the 
        feature covariance. 
        """
        self.T = T
        self.d = self.B.d
        self.weights = np.matrix(np.random.normal(0,1,(self.d,1)))
        self.aux_weights = np.matrix(np.random.normal(0,1,(self.d,1)))
        self.b_vec = np.matrix(np.zeros((self.d,1)))
        self.aux_vec = np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.aux_weights = self.Cinv*self.b_vec
        self.t = 1

        if "delta" in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 0.05
        if "lr" in params.keys():
            self.lr = params['lr']
        else:
            self.lr = 0.1
        if "lr2" in params.keys():
            self.lr2 = params['lr2']
        else:
            self.lr2 = 0.1
        if "schedule" in params.keys():
            self.schedule = params['schedule']
        else:
            self.schedule = 100
        self.reward = []
        self.opt_reward = []

        if self.lr2 == 0:
            self.aux_weights = np.matrix(np.zeros((self.d,1)))

    def update(self, x, A, y_vec, r):
        """
        Update the regression target and feature cov. 
        """
        features = np.matrix(x.get_ld_features())
        for i in range(len(A)):
            rand_noise = np.random.normal(0,1)
#             self.weights -= self.lr/np.sqrt(self.t)*features[A[i],:].T*(features[A[i],:]*self.weights - y_vec[i])
#             self.aux_weights -= self.lr2/np.sqrt(self.t)*features[A[i],:].T*(features[A[i],:]*self.aux_weights - rand_noise)
        
            self.cov += features[A[i],:].T*features[A[i],:]
            self.b_vec += y_vec[i]*features[A[i],:].T
            self.aux_vec += rand_noise*features[A[i],:].T
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        if self.lr2 != 0:
            self.aux_weights = self.Cinv*self.aux_vec
        self.t += 1
#         import pdb
#         pdb.set_trace()
#         if self.t % self.schedule == 0:
#         print("True weight norm: %0.2f" % (np.linalg.norm(self.weights)), flush=True)
#         print("Aux weight norm: %0.2f" % (np.linalg.norm(self.aux_weights)), flush=True)
#             self.Cinv = scipy.linalg.inv(self.cov)
#             self.weights = self.Cinv*self.b_vec

    def get_action(self, x):
        """
        Find the UCBs for the predicted reward for each base action
        and play the composite action that maximizes the UCBs
        subject to whatever constraints. 
        """
        features = np.matrix(x.get_ld_features())
        K = x.get_K()
        # Cinv = scipy.linalg.inv(self.cov)
        # self.weights = Cinv*self.b_vec

#         import pdb
#         pdb.set_trace()
        alpha = np.sqrt(self.d)*self.delta ## *np.log((1+self.t*K)/self.delta)) + 1
        ucbs = [features[k,:]*self.weights + alpha*np.abs(features[k,:]*self.aux_weights) for k in range(K)]
        # ucbs = [features[k,:]*self.weights + alpha*np.sqrt(features[k,:]*self.Cinv*features[k,:].T) for k in range(K)]

        ucbs = [a[0,0] for a in ucbs]
        ranks = np.argsort(ucbs)
        return ranks[K-self.B.L:K]

if __name__=='__main__':
    import sys, os, argparse, time
    import Simulators, Semibandits

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=50000, help='number of rounds', type=int)
    parser.add_argument('--iters', action='store', default=1, type=int)
    parser.add_argument('--d', action='store', default=20, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)

    parser.add_argument('--alg', action='store', default='all', choices=['linucb','linlin'])
    parser.add_argument('--param', action='store', default=None)
    parser.add_argument('--lr', action='store', type=float, default=0.1)
    parser.add_argument('--lr2', action='store', type=float, default=0.1) 
    parser.add_argument('--noise', action='store', default=None)


    Args = parser.parse_args(sys.argv[1:])
    print(Args,flush=True)
    if Args.noise is not None:
        Args.noise = float(Args.noise)

    outdir = './results/T=%d_d=%d_K=%d_sig=%0.1f/' % (Args.T, Args.d, Args.K, Args.noise)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if Args.param is not None:
        Args.param = float(Args.param)

    rewards = []
    regrets = []
    times = []
    for i in range(Args.iters):
        S = Simulators.LinearBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i, pos=True)
        if Args.alg == 'linucb':
            Alg = Semibandits.LinUCB(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'schedule': 1})
                stop = time.time()
        if Args.alg == 'linlin':
            Alg = LinUCB(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'lr': Args.lr, 'lr2': Args.lr2, 'schedule': 1})
                stop = time.time()
        times.append(stop-start)
        rewards.append(r)
        regrets.append(reg)

    if Args.lr2 == 0:
        Args.alg = 'greedy'
    np.savetxt(outdir+"%s_%0.5f_%0.5f_%0.5f_rewards.out" % (Args.alg, Args.param, Args.lr, Args.lr2), rewards)
    np.savetxt(outdir+"%s_%0.5f_%0.5f_%0.5f_regrets.out" % (Args.alg, Args.param, Args.lr, Args.lr2), regrets)
    np.savetxt(outdir+"%s_%0.5f_%0.5f_%0.5f_times.out" % (Args.alg, Args.param, Args.lr, Args.lr2), times)
    print("---- DONE ----")
