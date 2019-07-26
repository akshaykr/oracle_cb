import numpy as np
import Simulators
from Util import *
import scipy.linalg
import Semibandits


class LimeCB(Semibandits.Semibandit):
    def __init__(self,B):
        self.B = B

    def init(self, T, params={}):

        self.random_state = np.random.RandomState(params['seed'])

        self.T = T
        self.max_d = self.B.d
        self.dimensions = [2**i for i in range(1, int(np.log2(self.max_d)))]
        self.dimensions.append(self.max_d)
        
    
        self.d = self.dimensions[0]
    
        self.b_vec = np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.t = 1

        self.global_b = np.matrix(np.zeros((self.max_d,1)))
        self.global_cov = np.matrix(np.zeros((self.max_d,self.max_d)))
        self.random_samples = 0

        if "delta" in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 0.05
        if "schedule" in params.keys():
            self.schedule = params['schedule']
        else:
            self.schedule = 100
        if 'mu' in params.keys():
            self.mu = params['mu']
        else:
            self.mu = 1
            
        self.random = False
        self.reward = []
        self.opt_reward = []

    def _get_mu(self):
        """
        Return the current value of mu_t
        """
        ## a = 1.0/(2*self.B.K)
        ## b = np.sqrt(np.log(16.0*(self.t**2)*self.B.N/self.delta)/float(self.B.K*self.B.L*self.t))
        a = self.mu
        b = self.mu*np.sqrt(self.B.K)/np.sqrt(self.B.L*self.t)
        c = np.min([a,b])
        return np.min([1,c])

    def update(self, x, A, y_vec, r):
        """
        Update the regression target and feature cov. 
        """
        features = np.matrix(x.get_ld_features())
        features = features[:,0:self.d]
        for i in range(len(A)):
            self.cov += features[A[i],:].T*features[A[i],:]
            self.b_vec += y_vec[i]*features[A[i],:].T

        features = np.matrix(x.get_ld_features())
        for i in range(features.shape[0]):
            self.global_cov += features[i,:].T*features[i,:]
        if self.random:
            for i in range(len(A)):
                self.global_b += y_vec[i]*features[A[i],:].T
                self.random_samples += 1

        self.t += 1
        if self.t % self.schedule == 0:
            self.Cinv = scipy.linalg.inv(self.cov)
            self.weights = self.Cinv*self.b_vec

        if self.t % 10 == 0:
            self.estimate_residual()

    def get_action(self, x):
        """
        Find the UCBs for the predicted reward for each base action
        and play the composite action that maximizes the UCBs
        subject to whatever constraints. 
        """
        ber = self.random_state.binomial(1, self._get_mu())
        if ber == 1:
            self.random = True
            K = x.get_K()
            act = self.random_state.choice(K)
            return [act]
        else:
            self.random = False
            features = np.matrix(x.get_ld_features())
            features = features[:,0:self.d]
            K = x.get_K()
            # Cinv = scipy.linalg.inv(self.cov)
            # self.weights = Cinv*self.b_vec
            
            alpha = np.sqrt(self.d)*self.delta ## *np.log((1+self.t*K)/self.delta)) + 1
            ucbs = [features[k,:]*self.weights + alpha*np.sqrt(features[k,:]*self.Cinv*features[k,:].T) for k in range(K)]
            
            ucbs = [a[0,0] for a in ucbs]
            ranks = np.argsort(ucbs)
            return ranks[K-self.B.L:K]

    def estimate_residual(self):
        to_move = None
        done = False
        for d in self.dimensions:
            if d <= self.d or done:
                continue
            ### Construct R matrix
            tmp = np.matrix(np.zeros((d,d)))
            tmp[0:self.d,0:self.d] = self.global_cov[0:self.d,0:self.d]
            Sigma = self.global_cov[0:d,0:d]
            R = np.linalg.pinv(tmp) - np.linalg.pinv(Sigma)
            score = self.global_b[0:d].T*R*Sigma*R*self.global_b[0:d]/self.random_samples**2
#             print(score)
#             print("[LimeCB] curr_d=%d, test_d=%d, score=%0.3f, thres=%0.2f" % (self.d, d, score, np.sqrt(d)/self.random_samples), flush=True)
            if score > 0.001*np.sqrt(d)/self.random_samples:
                ### Then we switch!
                print("[LimeCB] Switching to d=%d" % (d), flush=True)
                self.d = d
                self.b_vec = np.matrix(np.zeros((self.d,1)))
                self.cov = np.matrix(np.eye(self.d))
                self.Cinv = scipy.linalg.inv(self.cov)
                self.weights = self.Cinv*self.b_vec
                done = True
            
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

    parser.add_argument('--alg', action='store', default='all', choices=['linucb','bose','minimonster','epsgreedy', 'thompson', 'bag', 'langevin', 'limecb'])
    parser.add_argument('--param', action='store', default=None)
    parser.add_argument('--noise', action='store', default=None)
    parser.add_argument('--loss', action='store', default=False)
                        

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
        if Args.dataset =='linear' and Args.feat=='sphere':
            S = Simulators.LinearBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i, pos=False)

        if Args.alg == 'linucb':
            Alg = Semibandits.LinUCB(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'schedule': 1})
                stop = time.time()
        if Args.alg == 'limecb':
            Alg = LimeCB(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'schedule': 1, 'seed': i})
                stop = time.time()
        if Args.alg == 'epsgreedy':
            learning_alg = lambda: sklearn.linear_model.LinearRegression()
            Alg = Semibandits.EpsGreedy(S,learning_alg=learning_alg,classification=False)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp)=Alg.play(Args.T, verbose=True, params={'eps':Args.param,'train_all':True,'schedule':'lin'})
                stop=time.time()
        rewards.append(r)
        regrets.append(reg)

    np.savetxt(outdir+"%s_%0.5f_rewards.out" % (Args.alg, Args.param), rewards)
    np.savetxt(outdir+"%s_%0.5f_regrets.out" % (Args.alg, Args.param), regrets)
    print("---- DONE ----")
