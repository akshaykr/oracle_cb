import numpy as np

"""
Module of Experts algorithms
"""
class Expert(object):
    def __init__(self, B):
        self.B = B
    def init(self, T):
        self.reward = 0.0
        self.dist = [1.0/self.B.N for i in range(self.B.N)]
    def play(self,T):
        self.init(T)
        for t in range(T):
            if np.log2(t+1) == int(np.log2(t+1)):
                print("t = %d, r = %0.3f" % (t, self.reward))
            x = self.B.get_new_context()
            p = self.get_action(x) 
            self.reward += self.B.get_reward(p)
            self.update(x, self.B.get_all_rewards())
        return self.reward
    def update(self,x,r_vec):
        pass
    def get_action(self, x):
        dist = [1.0/self.B.K for i in range(self.B.K)]
        p = np.random.multinomial(1, dist) ## Distribution over ACTIONS
        p = int(np.nonzero(p)[0])
        return p

class FTPL(Expert):
    def __init__(self,B):
        self.B = B
    def init(self, T):
        self.reward = 0.0
        self.T = T
        self.eta = np.sqrt(T)
        self.scores = [0 for i in range(self.B.N)]
    def update(self, x, r_vec):
        for i in range(self.B.N):
            self.scores[i] += r_vec[self.B.Pi[i,x]]
    def get_action(self, x):
        noise = np.random.normal(0, 1, self.B.N)
        pi = np.argmax(self.scores + self.eta*noise)
        return self.B.Pi[pi,x]

class FTPL2(Expert):
    def __init__(self,B):
        self.B = B
    def init(self, T):
        self.reward = 0.0
        self.T = T
        self.eta = np.sqrt(1/T)
        self.scores = [0 for i in range(self.B.N)]
    def update(self, x, r_vec):
        noise = np.random.normal(0, 1, self.B.K)
        for i in range(self.B.N):
            self.scores[i] += r_vec[self.B.Pi[i,x]] + self.eta*noise[self.B.Pi[i,x]]
    def get_action(self, x):
        pi = np.argmax(self.scores)
        return self.B.Pi[pi,x]

class Exp3(Expert):
    def __init__(self, B):
        self.B = B
    def init(self,T):
        self.reward = 0.0
        self.dist = [1.0/self.B.N for i in range(self.B.N)]
        self.T = T
        self.weight = np.sqrt(2.0*np.log(self.B.N)/self.T)
    def update(self, x, r_vec):
        for i in range(self.B.N):
            self.dist[i] = self.dist[i]*np.exp(self.weight*r_vec[self.B.Pi[i, x]])
        self.dist = self.dist/np.sum(self.dist)
    def get_action(self,x):
        out = []
        for i in range(self.B.K):
            out.append(np.sum([self.dist[j] for j in range(self.B.N) if self.B.Pi[j,x] == i]))
        p = np.random.multinomial(1, out) ## Distribution over ACTIONS
        p = int(np.nonzero(p)[0])
        return p
