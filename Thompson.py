import numpy as np
import Simulators
from Util import *
import scipy.linalg
import Semibandits


class Thompson(Semibandits.Semibandit):

    def __init__(self,B):
        self.B = B

    def init(self, T, params={}):
        self.T = T

        self.d = self.B.d
        self.b_vec= np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        self.lam = 0.1
        if 'lambda' in params:
            self.lam = params['lambda']
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.t=1

        self.reward = []
        self.opt_reward = []

    def update(self, x, A, y_vec, r):
        self.t += 1

        features = np.matrix(x.get_ld_features())
        for i in range(len(A)):
            self.cov += features[A[i],:].T*features[A[i],:]
            self.b_vec += (y_vec[i]*features[A[i],:]).T
        if True:
            self.Cinv = scipy.linalg.inv(self.cov)
            self.weights = self.Cinv*self.b_vec

    def get_action(self,x):
        features = np.matrix(x.get_ld_features())
        K = x.get_K()
        w = np.array(self.weights.T)[0]
        mut = np.matrix(np.random.multivariate_normal(w, self.lam*self.Cinv))
        scores = features*mut.T
        scores = [scores[0,0] for a in scores]
        
        ranks = np.argsort(scores)
        return ranks[K-self.B.L:K]
