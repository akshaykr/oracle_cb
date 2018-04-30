import numpy as np
import Semibandits, Simulators
import NNModels
from Util import *


class LangevinCB(Semibandits.Semibandit):

    def __init__(self, B, model=NNModels.LinearModel):
        self.B = B
        self.model = model(B.d, B.K)
        
    def init(self, T, params={}):
        if 'schedule' in params.keys():
            self.training_points = training_points(T, schedule=params['schedule'])
        else:
            self.training_points = training_points(T)

        self.reward = []
        self.opt_reward = []
        self.d = self.B.d
        self.K = self.B.K

        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        
        self.X = np.zeros((self.T, 1, self.B.d*self.B.K))
        self.R = np.zeros((self.T, 1, self.B.K))

        ## Exploration probability
        if 'mu' in params.keys():
            self.mu = params['mu']
        else:
            self.mu = 1.0
        ## Learning rate
        self.lr = 0.1

        ## Burn in
        self.burn_in = 100
        self.step = 10
        
    def update(self, x, A, y_vec, r):
        full_rvec = np.zeros(x.get_K())
        full_rvec[A] = y_vec/self.imp_weights[A]
        self.R[self.t,0,:] = full_rvec
        feat = x.get_ld_features()
        self.X[self.t,0,:] = feat.reshape(1,self.d*self.K)
        
    def _sample(self, x, p_vec):
        unif = np.random.binomial(1, self._get_mu())
        if unif:
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
        else:
            draw = np.random.multinomial(1, p_vec)
            act = np.where(draw)[0][0]
        return act

    def get_action(self,x):
        ## Run Langevin Gradient
        NNModels.langevin_step(self.model, self.X[0:self.t,:,:], self.R[0:self.t,:,:], self.lr, iters=self.burn_in, noise=True)
        p_vec = NNModels.model_to_action(self.model, x)
        self.action = self._sample(x, p_vec)
        ## Compute importance weight for this action
        M = 1
        while True:
            NNModels.langevin_step(self.model, self.X[0:self.t,:,:], self.R[0:self.t,:,:], self.lr, iters=self.step, noise=True)
            p_vec = NNModels.model_to_action(self.model, x)
            act = self._sample(x, p_vec)
            if act == self.action:
                break
            else:
                M += 1
            
        self.imp_weights = np.zeros(x.get_K())
        self.imp_weights[self.action] = M
        return [self.action]

    def _get_mu(self):
        a = self.mu
        b = self.mu*np.sqrt(self.B.K)/np.sqrt(self.t)
        c = np.min([a,b])
        return (np.min([1,c]))


class BagCB(Semibandits.Semibandit):

    def __init__(self, B, model=NNModels.LinearModel):
        self.B = B
        self.model = model
        
    def init(self, T, params={}):
        if 'schedule' in params.keys():
            self.training_points = training_points(T, schedule=params['schedule'])
        else:
            self.training_points = training_points(T)

        self.reward = []
        self.opt_reward = []
        self.d = self.B.d
        self.K = self.B.K

        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        
        ## Exploration probability
        if 'mu' in params.keys():
            self.mu = float(params['mu'])
        else:
            self.mu = 0.00

        ## Learning rate
        if 'lr' in params.keys():
            self.lr = float(params['lr'])
        else:
            self.lr = 0.1

        if 'M' in params.keys():
            self.M = int(params['M'])
        else:
            self.M = 20
        self.models = [self.model(self.B.d, self.B.K) for i in range(self.M)]


    def update(self, x, A, y_vec, r):
        full_rvec = np.zeros(x.get_K())
        full_rvec[A] = y_vec/self.imp_weights[A]
        feat = x.get_ld_features()
        X = np.zeros((1,1,self.d*self.K))
        X[0,0,:] = feat.reshape(1,self.d*self.K)
        full_rvec = full_rvec.reshape(1,1,self.K)
        for model in self.models:
            NNModels.langevin_step(model, X, full_rvec, self.lr, iters=int(np.random.poisson(lam=1.0,size=1)), noise=True)
        
    def _sample(self, x, p_vec):
        unif = np.random.binomial(1, self._get_mu())
        if unif:
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
        else:
            draw = np.random.multinomial(1, p_vec)
            act = np.where(draw)[0][0]
        return act

    def get_action(self,x):
        ## Run Langevin Gradient
        p_vecs = [NNModels.model_to_action(self.models[i], x) for i in range(self.M)]
        p_vec = np.mean(p_vecs,axis=0)
        self.action = self._sample(x, p_vec)

        self.imp_weights = np.zeros(x.get_K())
        self.imp_weights[self.action] = (1-self._get_mu())*p_vec[self.action] + self._get_mu()/x.get_K()
        return [self.action]

    def _get_mu(self):
        a = self.mu
        b = self.mu*np.sqrt(self.B.K)/np.sqrt(self.t)
        c = np.min([a,b])
        return (np.min([1,c]))

if __name__=='__main__':
    S = Simulators.LinearBandit(10, 1, 2, noise=True, pos=False)
    
    # L = LangevinCB(S)
    L = BagCB(S)
    L.play(5000, params={'M': 10})
