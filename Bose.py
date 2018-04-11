import numpy as np
import Simulators
from Util import *
import scipy.linalg
import Semibandits



class BOSE(Semibandits.Semibandit):
    """
    Implementation of BOSE (bandit semiparametric orthogonalized
    estimator) algorithm. This algorithm only works if features are
    available in the SemibanditSim object

    The SemibanditSim must also expose B.D as an instance variable
    """
    def __init__(self,B):
        self.B = B

    def init(self, T, params={}):
        """
        Initialize the regression target and the feature covariance
        """
        self.T = T
        self.d = self.B.d
        self.b_vec= np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        if 'lambda' in params:
            self.cov = params['lambda']*self.cov
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.t=1

        if 'delta' in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 0.05

        self.gamma = np.sqrt(self.d)*self.delta
        
        self.w = None

        self.reward = []
        self.opt_reward = []


    def update(self, x, A, y_vec, r):
        self.t += 1
        if np.linalg.norm(self.w) >=  1.0:
            pass
        else:
            features = np.matrix(x.get_ld_features())
            mu = np.zeros((1, self.B.d))
            for i in range(len(self.w)):
                mu += self.w[i]*features[i,:]
            for i in range(len(A)):
                self.cov += (features[A[i],:] - mu).T*(features[A[i],:]-mu)
                self.b_vec += (y_vec[i]*(features[A[i],:]- mu)).T
                ## self.b_vec = self.b_vec + (y_vec[i]*(features[A[i],:] - mu)).T
                ## print(mu)
                ## print(self.cov)
                ## print(self.b_vec)
        ## if self.t % 100 == 0:
        if True:
            self.Cinv = scipy.linalg.inv(self.cov)
            self.weights = self.Cinv*self.b_vec

    def get_action(self, x):
        features = np.matrix(x.get_ld_features())

        K = features.shape[0]
        survivors = []
        for i in range(K):
            ### Check if action survives
            fail = False
            for j in range(K):
                vec = features[j,:] - features[i,:]
                tmp = self.weights.T*vec.T - self.gamma*np.sqrt(vec*self.Cinv*vec.T)
                if tmp > 0:
                    fail=True
                    break
            if not fail:
                survivors.append(i)
        
        ### Find distribution over survivors (for now uniform)
        w = [1.0/len(survivors) if i in survivors else 0 for i in range(K)]
        ## w = [1.0/K for i in range(K)]
        self.w = w
        
        samp = np.random.multinomial(1, self.w)
        a = np.where(samp==1)[0][0]
        return [a]


if __name__=='__main__':
    import sys, os, argparse, time
    import sklearn.linear_model

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=50000, help='number of rounds', type=int)
    parser.add_argument('--dataset', action='store', choices=['linear', 'semiparametric'])
    parser.add_argument('--feat', action='store', choices=['sphere','pos'])
    ## parser.add_argument('--start', action='store', default=0, type=int)
    parser.add_argument('--iters', action='store', default=1, type=int)
    parser.add_argument('--d', action='store', default=20, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)

    parser.add_argument('--alg', action='store', default='all', choices=['linucb','bose','minimonster','epsgreedy', 'thompson'])
    parser.add_argument('--param', action='store', default=None)
    parser.add_argument('--noise', action='store', default=None)
                        

    Args = parser.parse_args(sys.argv[1:])
    print(Args,flush=True)
    if Args.noise is not None:
        Args.noise = float(Args.noise)

    outdir = './results/%s_%s_T=%d_d=%d_K=%d_sig=%0.1f/' % (Args.dataset, Args.feat, Args.T, Args.d, Args.K, Args.noise)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if Args.param is not None:
        Args.param = float(Args.param)

    
    rewards = []
    regrets = []
    for i in range(Args.iters):
        if Args.dataset =='linear' and Args.feat=='pos':
            S = Simulators.LinearBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i, pos=True)
        if Args.dataset =='linear' and Args.feat=='sphere':
            S = Simulators.LinearBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i, pos=False)
        if Args.dataset =='semiparametric' and Args.feat=='pos':
            S = Simulators.SemiparametricBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i, pos=True)
        if Args.dataset =='semiparametric' and Args.feat=='sphere':
            S = Simulators.SemiparametricBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i, pos=False)

        if Args.alg == 'linucb':
            Alg = Semibandits.LinUCB(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'schedule': 1})
                stop = time.time()
        if Args.alg == 'bose':
            Alg = BOSE(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'schedule': 1})
                stop = time.time()
        if Args.alg == 'minimonster':
            learning_alg = lambda: sklearn.linear_model.LinearRegression()
            Alg = Semibandits.MiniMonster(S,learning_alg=learning_alg,classification=False)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp)=Alg.play(Args.T, verbose=True, params={'mu':Args.param,'schedule':'lin'})
                stop=time.time()
        if Args.alg == 'epsgreedy':
            learning_alg = lambda: sklearn.linear_model.LinearRegression()
            Alg = Semibandits.EpsGreedy(S,learning_alg=learning_alg,classification=False)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp)=Alg.play(Args.T, verbose=True, params={'eps':Args.param,'train_all':True,'schedule':'lin'})
                stop=time.time()
        if Args.alg == 'thompson':
            import Thompson
            Alg = Thompson.Thompson(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'lambda':Args.param})
                stop = time.time()
        rewards.append(r)
        regrets.append(reg)

    np.savetxt(outdir+"%s_%0.5f_rewards.out" % (Args.alg, Args.param), rewards)
    np.savetxt(outdir+"%s_%0.5f_regrets.out" % (Args.alg, Args.param), regrets)

    print("---- DONE ----")
