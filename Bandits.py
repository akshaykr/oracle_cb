import numpy as np
import sys

"""
Module of bandit algorithms

Bandit algorithms should implement the following methods:
1. __init__(B): constructor that takes a Bandit Simulator object.
2. init(T): prepare to run for T rounds, wipe state, etc.
3. updated(x,a,r): update any state using the current interaction
4. get_action(x): propose an action for this context.
"""
class Bandit(object):
    """
    Bandit Algorithm interface. 

    This is a valid bandit algorithm that just plays randomly every
    round.
    """
    def __init__(self, B):
        self.B = B

    def init(self,T):
        self.reward = 0.0
        self.opt_reward = 0.0
        self.dist = [1.0/self.B.N for i in range(self.B.N)]

    def play(self,T):
        self.init(T)
        scores = []
        for t in range(T):
            if np.log2(t+1) == int(np.log2(t+1)):
                print("t = %d, r = %0.3f, ave_regret = %0.3f" % (t, self.reward, (self.opt_reward - self.reward)/(t+1)))
            x = self.B.get_new_context()
            p = self.get_action(x) 
            self.reward += self.B.get_reward(p)
            self.opt_reward += self.B.get_reward(self.B.Pi[self.B.Pistar,x])
            self.update(x, p, self.B.get_reward(p))
            scores.append(self.opt_reward - self.reward)
        return scores

    def update(self, x, a, r):
        pass

    def get_action(self, x):
        dist = [1.0/self.B.K for i in range(self.B.K)]
        p = np.random.multinomial(1, dist) ## Distribution over ACTIONS
        p = int(np.nonzero(p)[0])
        return p

class BFTPL(Bandit):
    """
    Follow the perturbed leader style bandit algorithm. 

    @deprecated: This needs to be tuned properly, so should not be
    used.
    """
    def init(self,T):
        self.reward = 0.0
        self.weights = np.array([0 for i in range(self.B.N)])
        self.noise = np.random.normal(0, 1, [1,self.B.N])
        self.eta = np.sqrt(T)

    def update(self, x, a, r):
        ## Estimate probability of playing action a
        ## so that we can importance weight
        counts = [0.0 for i in range(self.B.K)]
        for n in range(1000):
            noise = np.random.normal(0, 1, [1,self.B.N])
            pi = self.argmax(noise)
            counts[self.B.Pi[pi, x]] += 1
        counts = [x/1000 for x in counts]
        
        print("Updating policies: action %d, reward %d, IPS %0.2f" % (a, r, counts[a]))
        for i in range(self.B.N):
            if self.B.Pi[i,x] == a:
                self.weights[i] += r/counts[a]

    def get_action(self,x):
        pi = self.argmax(self.noise)
        return self.B.Pi[pi,x]

    def argmax(self, noise):
         return np.argmax(self.weights + self.eta*noise)
