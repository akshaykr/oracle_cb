import numpy as np
import Simulators
from Util import *
import scipy.linalg
import sklearn.linear_model
from Policy import *
import Argmax
import pickle

"""
Module of Semibandit algorithms
"""

class Semibandit(object):
    """
    Default semibandit learning algorithm.  Provides interface for
    semibandit learning but also implements the random policy that
    picks random slates on each iteration and does no updating.
    """
    def __init__(self, B):
        """
        Args: 
        B -- A SemibanditSim instance.
        """
        self.B = B
        
    def init(self, T, params={}):
        """
        Initialize the semibandit algorithm for a run.
        Args:
        T -- maximum number of rounds.
        """
        self.reward = []
        self.opt_reward = []
        self.dist = [1.0/self.B.N for i in range(self.B.N)]

    def play(self, T, params={}, verbose=True, validate=None):
        """
        Execute this algorithm on the semibandit simulator for T rounds.

        Returns: 
        cumulative reward (np.array),
        cumulative optimal reward (np.array)
        cumulative regret (np.array)
        """
        self.verbose=verbose
        self.init(T, params=params)
        self.val_scores = []
        for t in range(T):
            if t != 0 and np.log2(t+1) == int(np.log2(t+1)) and verbose:
                print("t = %d, r = %0.3f, ave_regret = %0.3f" % (t, np.cumsum(self.reward)[len(self.reward)-1], (np.cumsum(self.opt_reward) - np.cumsum(self.reward))[len(self.reward)-1]/(t+1)), flush=True)
            if validate != None and t != 0 and t % 500 == 0:
                val = validate.offline_evaluate(self, train=False)
                if verbose:
                    print("t=%d: val = %0.3f" % (t, val))
                self.val_scores.append(val)
            x = self.B.get_new_context()
            if x == None:
                break
            p = self.get_action(x)
            r = self.B.get_slate_reward(p)
            o = self.B.get_best_reward()
            # if verbose:
            #     print('context: ', x.get_name())
            #     print('action: ', " ".join([str(x) for x in p]))
            #     print('reward: ', " ".join([str(x) for x in self.B.get_base_rewards(p)]))
            self.reward.append(r)
            self.opt_reward.append(o)
            self.update(x, p, self.B.get_base_rewards(p), self.B.get_slate_reward(p))
        l1 = np.cumsum(self.reward)
        l2 = np.cumsum(self.opt_reward)
        return (l1[9::10], (l2-l1)[9::10], self.val_scores)

    def update(self, x, a, y_vec, r):
        """
        Update the state of the semibandit algorithm with the most recent
        context, composite action, and reward.

        The default semibandit doesn't do any updating.

        Args:
        x -- context (should be hashable)
        a -- composite action (np.array of length L)
        r -- reward vector (np.array of length K)
        """
        pass

    def get_action(self, x):
        """
        Pick a composite action to play for this context. 

        Args:
        x -- context

        Returns:
        Composite action (np.array of length L)
        """
        act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
        self.action = act
        return self.action
        # dist = [1.0/self.B.K for i in range(self.B.K)]
        # p = np.random.multinomial(1, dist) ## Distribution over ACTIONS
        # p = int(np.nonzero(p)[0])
        # return p


class EpsGreedy(Semibandit):
    """
    Epsilon Greedy algorithm for semibandit learning.
    Can use scikit_learn LearningAlg as oracle

    Current implementation uses a constant value for epsilon. 
    """
    def __init__(self, B, learning_alg="enumerate", classification=False):
        """
        Initialize the epsilon greedy algorithm.
        
        Args:
        B -- Semibandit Simulator
        learning_alg -- scikit learn regression algorithm.
                        Should support fit() and predict() methods.
        """
        self.B = B
        self.link = "linear"
        self.learning_alg = learning_alg
        if learning_alg == "enumerate":
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

    def init(self, T, params={}):
        """
        Initialize the current run. 
        The EpsGreedy algorithm maintains a lot more state.

        Args:
        T -- Number of rounds of interaction.
        """
        if "eps" in params.keys():
            self.eps = params['eps']
        else:
            self.eps = 0.1
        if "reward" in params.keys() and params['reward'] == True:
            self.use_reward_features = False
        else:
            self.use_reward_features = True
            if 'weight' in params.keys():
                self.weights = params['weight']
            else:
                self.weights = self.B.weight

        if 'train_all' in params.keys() and params['train_all']:
            self.train_all = True
        else:
            self.train_all = False

        if 'learning_alg' in params.keys():
            self.learning_alg = params['learning_alg']
        if self.learning_alg == "enumerate":
            assert 'Pi' in dir(self.B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
        elif 'classification' in params.keys():
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

        if "link" in params.keys():
            self.link = params['link']

        self.training_points = []
        i = 4
        while True:
            self.training_points.append(int(np.sqrt(2)**i))
            i+=1
            if np.sqrt(2)**i > T:
                break
        print(self.training_points)

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None
        self.leader = None
        self.history = []
        self.ber = 0
        self.num_unif = 0

    def get_action(self, x):
        """
        Select a composite action to play for this context.
        Also updates the importance weights.

        Args:
        x -- context.
        """
        self.imp_weights = (x.get_L()/x.get_K())*np.ones(x.get_K())
        if self.leader is not None and self.train_all:
            self.imp_weights = (self._get_eps())*self.imp_weights
            self.imp_weights[self.leader.get_action(x)] += (1-self._get_eps())
        self.ber = np.random.binomial(1,np.min([1, self._get_eps()]))
        if self.leader is None or self.ber:
            A = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.action = A
            self.num_unif += 1
        elif self.use_reward_features:
            A = self.leader.get_weighted_action(x,self.weights)
        else:
            A = self.leader.get_action(x)
        self.action = A
        return A
            
    def update(self, x, act, y_vec, r):
        """
        Update the state for the algorithm.
        We currently call the AMO whenever t is a perfect square.

        Args:
        x -- context.
        act -- composite action (np.array of length L)
        r_vec -- reward vector (np.array of length K)
        """
        if self.use_reward_features:
            full_rvec = np.zeros(self.B.K)
            full_rvec[act] = y_vec ##/self.imp_weights[act]
            if self.train_all or self.ber:
                self.history.append((x, act, full_rvec, 1.0/self.imp_weights))
        elif self.ber:
            self.history.append((x,act,r))
        ##if self.t >= 10 and np.log2(self.t) == int(np.log2(self.t)):
        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("----Training----", flush=True)
            if self.use_reward_features:
                ## self.leader = Argmax.weighted_argmax(self.B, self.history, self.weights, link=self.link, policy_type = self.policy_type, learning_alg = self.learning_alg)
                self.leader = Argmax.argmax2(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
            else:
                ## self.leader = Argmax.argmax(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
                self.leader = Argmax.reward_argmax(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
        self.t += 1

    def _get_eps(self):
        """
        Return the current value of epsilon.
        """
        return np.max([1.0/np.sqrt(self.t), self.eps])

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
        self.b_vec = np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.t = 1

        if "delta" in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 0.05

        self.reward = []
        self.opt_reward = []

    def update(self, x, A, y_vec, r):
        """
        Update the regression target and feature cov. 
        """
        features = np.matrix(x.get_ld_features())
        for i in range(len(A)):
            self.cov += features[A[i],:].T*features[A[i],:]
            self.b_vec += y_vec[i]*features[A[i],:].T

        self.t += 1
        if self.t % 100 == 0:
            self.Cinv = scipy.linalg.inv(self.cov)
            self.weights = self.Cinv*self.b_vec

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

        alpha = np.sqrt(self.d)*self.delta ## *np.log((1+self.t*K)/self.delta)) + 1
        ucbs = [features[k,:]*self.weights + alpha*np.sqrt(features[k,:]*self.Cinv*features[k,:].T) for k in range(K)]

        ucbs = [a[0,0] for a in ucbs]
        ranks = np.argsort(ucbs)
        return ranks[K-self.B.L:K]


class EELS(Semibandit):
    """
    Implementation of GLM-Semibandit with scikit_learn learning algorithm as the AMO.
    """
    def __init__(self, B):
        self.B = B
        self.learning_alg = "enumerate"
        self.link = "linear"

    def init(self, T, params=None):
        if 'learning_alg' in params.keys():
            self.learning_alg = params['learning_alg']
        if 'link' in params.keys():
            self.link = params['link']
        if self.learning_alg == 'enumerate':
            assert 'Pi' in dir(self.B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
        else:
            self.policy_type = RegressionPolicy

        self.lambda_star = np.sqrt(self.B.K*T)                                     ## TODO: maybe needs to change
        self.t_star = (T*self.B.K/self.B.L)**(2.0/3)*(1.0/self.B.L)**(1.0/3)       ## TODO: maybe needs to change
        self.reward = []
        self.opt_reward = []
        self.weights = None
        self.cov = np.matrix(np.zeros((self.B.L, self.B.L)))
        self.reg_data = []
        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        self.history = []
        self.emp_best = None

        self.reward = []
        self.opt_reward = []

    def get_action(self, x):
        if self.weights is None:
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.action = act
        else:
            self.action = self.emp_best.get_weighted_action(x,self.weights)
        return self.action

    def update(self, x, act, y_vec, r):
        if self.emp_best != None:
            return
        ## Otherwise we have to update
        vec = y_vec
        full_rvec = np.zeros(x.get_K())
        full_rvec[act] = y_vec*(self.B.L/self.B.K)*np.ones(len(act))
        self.history.append((x,full_rvec))
        self.cov += np.matrix(vec).T*np.matrix(vec)
        self.reg_data.append((np.matrix(vec).T, r))
        [u,v] = np.linalg.eig(self.cov)
        if np.min(np.real(u)) > self.lambda_star and self.t > self.t_star:
            if self.verbose:
                print("---- Training ----", flush=True)
            if self.link == "linear":
                model = sklearn.linear_model.LinearRegression(fit_intercept=True)
                X = np.matrix(np.hstack([self.reg_data[i][0] for i in range(len(self.reg_data))])).T
                y = np.hstack([self.reg_data[i][1] for i in range(len(self.reg_data))])
                model.fit(X,y)
                self.weights = np.array(model.coef_)[0,:]
            if self.link == "logistic":
                model = sklearn.linear_model.LogisticRegression(C=1000.0,fit_intercept=True)
                X = np.matrix(np.hstack([self.reg_data[i][0] for i in range(len(self.reg_data))])).T
                y = np.hstack([self.reg_data[i][1] for i in range(len(self.reg_data))])
                model.fit(X,y)
                self.weights = np.array(model.coef_)[0,:]
            self.emp_best = Argmax.weighted_argmax(self.B,self.history,self.weights, link=self.link, policy_type=self.policy_type, learning_alg=self.learning_alg)
        self.t += 1

class EELS2(Semibandit):
    """
    Implementation of GLM-Semibandit with scikit_learn learning algorithm as the AMO.
    """
    def __init__(self, B):
        self.B = B
        self.learning_alg = "enumerate"
        self.link = "linear"

    def init(self, T, params=None):
        if "eps" in params.keys():
            self.eps = params['eps']
        else:
            self.eps = 0.1
        if 'learning_alg' in params.keys():
            self.learning_alg = params['learning_alg']
        if 'link' in params.keys():
            self.link = params['link']
        if self.learning_alg == 'enumerate':
            assert 'Pi' in dir(self.B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
        else:
            self.policy_type = RegressionPolicy

        self.reward = []
        self.opt_reward = []
        self.weights = None
        self.cov = np.matrix(np.zeros((self.B.L, self.B.L)))
        self.reg_data = []
        self.T = T
        self.action = None
        self.t = 1
        self.history = []
        self.emp_best = None

        self.reward = []
        self.opt_reward = []

    def get_action(self, x):
        self.ber = np.random.binomial(1,np.min([1,self._get_eps()]))
        if self.weights is None or self.ber:
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.action = act
        else:
            self.action = self.emp_best.get_weighted_action(x,self.weights)
        return self.action

    def update(self, x, act, y_vec, r):
        ## Otherwise we have to update
        vec = y_vec
        full_rvec = np.zeros(x.get_K())
        full_rvec[act] = y_vec*(self.B.L/self.B.K)*np.ones(len(act))
        if self.ber:
            self.history.append((x,full_rvec))
        self.cov += np.matrix(vec).T*np.matrix(vec)
        self.reg_data.append((np.matrix(vec).T, r))
        if self.t >= 10 and np.log2(self.t) == int(np.log2(self.t)):
            if self.verbose:
                print("---- Training ----", flush=True)
            if self.link == "linear":
                model = sklearn.linear_model.LinearRegression(fit_intercept=True)
                X = np.matrix(np.hstack([self.reg_data[i][0] for i in range(len(self.reg_data))])).T
                y = np.hstack([self.reg_data[i][1] for i in range(len(self.reg_data))])
                model.fit(X,y)
                ## print(model.coef_)
                self.weights = np.array(model.coef_)
                self.intercept = model.intercept_
            if self.link == "logistic":
                model = sklearn.linear_model.LogisticRegression(C=1000.0,fit_intercept=True)
                X = np.matrix(np.hstack([self.reg_data[i][0] for i in range(len(self.reg_data))])).T
                y = np.hstack([self.reg_data[i][1] for i in range(len(self.reg_data))])
                model.fit(X,y)
                self.weights = np.array(model.coef_)[0,:]
                self.intercept = model.intercept_
                print(self.weights)
            self.emp_best = Argmax.weighted_argmax(self.B,self.history,self.weights, link=self.link, policy_type=self.policy_type, learning_alg=self.learning_alg, intercept=self.intercept)
        self.t += 1

    def _get_eps(self):
        """
        Return the current value of epsilon.
        """
        return np.max([1.0/np.sqrt(self.t), self.eps])

class MiniMonster(Semibandit):
    """
    Implementation of MiniMonster with a scikit_learn learning algorithm as the AMO.
    """
    def __init__(self, B, learning_alg = None, classification=True):
        self.B = B
        self.learning_alg = learning_alg
        if learning_alg == None:
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy


    def init(self, T, params={}):
        self.training_points = []
        i = 4
        while True:
            ## self.training_points.append(int(np.sqrt(2)**i))
            self.training_points.append(int(np.sqrt(2)**i))
            i+=1
            if np.sqrt(2)**i > T:
                break
        print(self.training_points)

        self.reward = []
        self.opt_reward = []
        self.weights = []
        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        self.leader = None
        self.history = []
        self.num_amo_calls = 0
        if 'mu' in params.keys():
            self.mu = params['mu']
        else:
            self.mu = 1.0
        self.num_unif = 0
        self.num_leader = 0

    def update(self, x, A, y_vec, r):
        full_rvec = np.zeros(x.get_K())
        full_rvec[A] = y_vec
        self.history.append((x, A, full_rvec, 1.0/self.imp_weights))
        ##if np.log2(self.t) == int(np.log2(self.t)):
        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("---- Training ----", flush=True)
            pi = Argmax.argmax2(self.B, self.history, policy_type=self.policy_type, learning_alg = self.learning_alg)                
            self.num_amo_calls += 1
            self.leader = pi
            self.weights = self._solve_op()
            if self.verbose:
                print("t: ", self.t, " Support: ", len(self.weights), flush=True)
            # print("----Evaluating policy distribution on training set----")
            # print("leader weight: %0.2f, score = %0.2f" % (1 - np.sum([z[1] for z in self.weights]), self.B.offline_evaluate(self.leader, train=True)))
            # for item in self.weights:
            #     pi = item[0]
            #     w = item[1]
            #     print("weight %0.2f, score = %0.2f" % (w, self.B.offline_evaluate(pi, train=True)))
            
        self.action = None
        self.imp_weights = None
        self.t += 1

    def get_action(self, x):
        """
        Choose a composite action for context x.
        
        Implements projection, smoothing, mixing etc. 
        Computes importance weights and stores in self.imp_weights.
        """
        ## Compute slate and base action distributions
        p = {}
        p2 = np.zeros(x.get_K())
        for item in self.weights:
            pi = item[0]
            w = item[1]
            (p, p2) = self._update_dists(p, p2, pi.get_action(x), w)

        ## Mix in leader
        if self.leader != None:
            (p, p2) = self._update_dists(p, p2, self.leader.get_action(x), 1 - np.sum([z[1] for z in self.weights]))
        ## Compute importance weights by mixing in uniform
        p2 = (1-self._get_mu())*p2 + (self._get_mu())*x.get_L()/x.get_K()
        self.imp_weights = p2

        ## Decide what action to play
        unif = np.random.binomial(1, self._get_mu())
        ## print("Exploration probability %0.3f" % (self._get_mu()))
        if unif or self.leader is None:
            ## Pick a slate uniformly at random
            ## print("Random action!")
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.num_unif += 1
        else:
            ## Pick a slate from the policy distribution. 
            p = [(k,v) for (k,v) in p.items()]
            draw = np.random.multinomial(1, [x[1] for x in p])
            ## print("Sampled action: ", p[np.where(draw)[0][0]][0], " %0.2f " % (p[np.where(draw)[0][0]][1]))
            ## print("Leader action: ", self.leader.get_action(x))
            act = self._unhash_list(p[np.where(draw)[0][0]][0])
            if p[np.where(draw)[0][0]][1] > 0.5:
                self.num_leader += 1
        self.action = act
        return act

    def _update_dists(self, slate_dist, base_dist, slate, weight):
        """
        This is a subroutine for the projection step. 
        Update the slate_dist and base_dist distributions by 
        incorporating slate played with weight.
        """
        key = self._hash_list(slate)
        if key in slate_dist.keys():
            slate_dist[key] = slate_dist[key] + weight
        else:
            slate_dist[key] = weight
        for a in slate:
            base_dist[a] += weight
        return (slate_dist, base_dist)
        
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

    def _hash_list(self, lst):
        """
        There is a hashing trick we are using to make a dictionary of
        composite actions. _hash_list and _unhash_list implement
        the hash and inverse hash functions. 

        The hash is to write the composite action as a number in base K.
        """
        return np.sum([lst[i]*self.B.K**i for i in range(len(lst))])

    def _unhash_list(self, num):
        lst = []
        if num == 0:
            return np.array([0 for i in range(self.B.L)])
        for i in range(self.B.L):
            rem = num % self.B.K
            lst.append(rem)
            num = (num-rem)/self.B.K
        return lst

    def _solve_op(self):
        """
        Main optimization logic for MiniMonster.
        """
        H = self.history
        mu = self._get_mu()
        Q = [] ## self.weights ## Warm-starting
        psi = 1

        ## MEMOIZATION
        ## 1. Unnormalized historical reward for each policy in supp(Q)
        ## 2. Feature matrix on historical contexts
        ## 3. Recommendations for each policy in supp(Q) for each context in history.
        predictions = {}
        ## vstack the features once here. This is a caching optimization
        if self.policy_type == ClassificationPolicy:
            features = np.zeros((1, H[0][0].get_dim()))
            for x in H:
                features = np.vstack((features, x[0].get_features()))
            features = features[1:,:]
        elif self.policy_type == RegressionPolicy:
            features = np.zeros((1, H[0][0].get_ld_dim()))
            for x in H:
                features = np.vstack((features, x[0].get_ld_features()))
            features = features[1:,:]
        else:
            features = None

        ## Invariant is that leader is non-Null
        (leader_reward, predictions) = self._get_reward(H, self.leader, predictions, features=features)
        leader_reward = leader_reward

        q_rewards = {}
        for item in Q:
            pi = item[0]
            (tmp,predictions) = self._get_reward(H, pi, predictions, features=features)
            q_rewards[pi] = tmp

        updated = True
        iterations = 0
        while updated and iterations < 20:
            print("OP Iteration")
            iterations += 1
            updated = False
            ## First IF statement
            score = np.sum([x[1]*(2*self.B.K*self.B.L/self.B.L + self.B.K*(leader_reward - q_rewards[x[0]])/(psi*self.t*self.B.L*mu)) for x in Q])
            if score > 2*self.B.K*self.B.L/self.B.L:
                # if self.verbose:
                #     print("Shrinking", flush=True)
                c = (2*self.B.K*self.B.L/self.B.L)/score
                Q = [(x[0], c*x[1]) for x in Q]
                updated = True

            ## argmax call and coordinate descent update. 
            ## Prepare dataset
            Rpi_dataset = []
            Vpi_dataset = []
            Spi_dataset = []
            for i in range(self.t):
                context = H[i][0]
                q = self._marginalize(Q, context, predictions)
                act = np.arange(context.get_K())
                r1 = 1.0/(self.t*q)
                r2 = self.B.K/(self.t*psi*mu*self.B.L)*H[i][2]/H[i][3]
                r3 = 1.0/(self.t*q**2)
                weight = np.ones(context.get_K())

                Vpi_dataset.append((context, act, r1, weight))
                Rpi_dataset.append((context, act, r2, weight))
                Spi_dataset.append((context, act, r3, weight))
            dataset = Rpi_dataset
            dataset.extend(Vpi_dataset)

            ## AMO call
            pi = Argmax.argmax2(self.B, dataset, policy_type=self.policy_type, learning_alg = self.learning_alg)
            self.num_amo_calls += 1

            ## This is mostly to make sure we have the predictions cached for this new policy
            if pi not in q_rewards.keys():
                (tmp,predictions) = self._get_reward(H, pi, predictions, features=features)
                q_rewards[pi] = tmp
                if q_rewards[pi] > leader_reward:
                    # if self.verbose:
                    #     print("Changing leader", flush=True)
                    self.leader = pi
                    leader_reward = q_rewards[pi]

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"
            ## Test if we need to update
            (Dpi,predictions) = self._get_reward(dataset, pi, predictions)
            target = 2*self.B.K*self.B.L/self.B.L + self.B.K*leader_reward/(psi*self.t*mu*self.B.L)
            if Dpi > target:
                ## Update
                updated = True
                Dpi = Dpi - (2*self.B.K*self.B.L/self.B.L + self.B.K*leader_reward/(psi*self.t*mu*self.B.L))
                (Vpi,ptwo) = self._get_reward(Vpi_dataset, pi, predictions)
                (Spi,ptwo) = self._get_reward(Spi_dataset, pi, predictions)
                toadd = (Vpi + Dpi)/(2*(1-mu)*Spi)
                Q.append((pi, toadd))
        return Q

    def _get_reward(self, dataset, pi, predictions, features=None):
        """
        For a policy pi whose predictions are cached in predictions dict,
        compute the cumulative reward on dataset.
        """
        ## assert pi in predictions.keys() or features is not None, "Something went wrong with caching"
        if pi not in predictions.keys():
            ## This is going to go horribly wrong if dataset is not the right size
            assert len(dataset) == self.t, "If predictions not yet cached, dataset should have len = self.t"
            predictions[pi] = dict(zip([y[0].get_name() for y in dataset], pi.get_all_actions([y[0] for y in dataset], features=features)))
        score = 0.0
        for item in dataset:
            x = item[0].get_name()
            r = item[2]
            w = item[3]
            score += np.sum(r[predictions[pi][x]]*w[predictions[pi][x]])

        return (score, predictions)
            
    def _marginalize(self, Q, x, predictions):
        """
        Marginalize a set of weights Q for context x
        using the predictions cache. 
        """
        p = np.zeros(self.B.K, dtype=np.longfloat)
        for item in Q:
            pi = item[0]
            w = item[1]
            p[predictions[pi][x.get_name()]] += w
        p = (1.0-self._get_mu())*p + (self._get_mu())*float(self.B.L)/float(self.B.K)
        return p


class SemiExp4(Semibandit):
    """
    Semibandit EXP4 as in Kale, Reyzin, Schapire.
    Uses parameters prescribed in the paper.

    This only works for Semibandit Simulators where there
    is a finite enumerable policy class. To get this to
    work on the Learning-to-Rank datasets, you need to construct
    such a policy class before hand and make it accessible to the 
    SemibanditSim object. 

    Specifically the algorithm uses B.Pi, which is a multidimensional
    array for which B.Pi[policy, context, :] is the prescribed composite
    action. The SemibanditSim object needs to have this structure.
    """
    
    def __init__(self, B):
        assert "Pi" in dir(B), "Cannot run EXP4 without explicit policy class"
        self.B = B
        self.link = "linear"

    def init(self, T, params={}):
        self.reward = []
        self.opt_reward = []
        self.dist = [1.0/self.B.N for i in range(self.B.N)]
        self.T = T
        ## Default parameter settings from the paper
        ## self.gamma = np.sqrt((self.B.K/self.B.L)*np.log(self.B.N)/self.T)
        if 'gamma' in params.keys():
            self.gamma = params['gamma']
        else:
            self.gamma = np.sqrt((self.B.K/self.B.L)*np.log(self.B.N)/self.T)
        if 'eta' in params.keys():
            self.eta = params['eta']
        else:
            self.eta = np.sqrt((1-self.gamma)*self.B.L*np.log(self.B.N)/(self.B.K*self.T))
        self.act_dist = []
        self.action = None
        self.weight = np.ones(self.B.L)
        if 'weight' in params.keys():
            self.weight = params['weight']
        if 'link' in params.keys():
            self.link = params['link']

    def update(self, x, act, y_vec, r):
        """
        This step enumerates the policy class. 
        """
        reward = np.zeros(self.B.K)
        reward[act] = y_vec/(self.B.L*self.act_dist[act])
        
        for n in range(self.B.N):
            if self.link == "linear":
                policy_reward = np.dot(self.weight,reward[self.B.Pi[n].get_action(x)])
            if self.link == "logistic":
                policy_reward = 1.0/(1+np.exp(-np.dot(self.weight, reward[self.B.Pi[n].get_action(x)])))
            self.dist[n] = self.dist[n]*np.exp(self.eta*policy_reward)
        self.dist = self.dist/np.sum(self.dist)

    def get_action(self,x):
        """
        This also enumerates the policy class. 
        """
        p = np.zeros(self.B.K)
        for n in range(self.B.N):
            p[self.B.Pi[n].get_action(x)] += self.dist[n]/self.B.L
        p = (1-self.gamma)*p + self.gamma/self.B.K*np.ones(self.B.K)
        self.act_dist = p
        (M, z) = mixture_decomp(p, self.B.L)
        samp = np.random.multinomial(1, z)
        row = M[np.where(samp)[0][0],:]
        self.action = np.where(row)[0]
        return(self.action)


if __name__=='__main__':
    import sklearn.ensemble
    import sklearn.tree
    import sys, os
    import argparse
    import settings
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store',
                        default=1000,
                        help='number of rounds', type=int)
    parser.add_argument('--dataset', action='store', choices=['synth','mq2007','mq2008', 'yahoo', 'mslr', 'mslrsmall', 'mslr30k', 'xor'])
    parser.add_argument('--L', action='store', default=5, type=int)
    parser.add_argument('--I', action='store', default=0, type=int)
    parser.add_argument('--noise', action='store', default=None)
    parser.add_argument('--alg', action='store' ,default='all', choices=['mini', 'eps', 'lin'])
    parser.add_argument('--learning_alg', action='store', default=None, choices=[None, 'gb2', 'gb5', 'tree', 'lin'])
    parser.add_argument('--param', action='store', default=None)
    
    Args = parser.parse_args(sys.argv[1:])
    print(Args, flush=True)
    if Args.noise is not None:
        Args.noise = float(Args.noise)
        outdir = './results/%s_T=%d_L=%d_e=%0.1f/' % (Args.dataset, Args.T, Args.L, Args.noise)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    else:
        outdir = './results/%s_T=%d_L=%d_e=0.0/' % (Args.dataset, Args.T, Args.L)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

    if Args.param is not None:
        Args.param = float(Args.param)

    loop = True
    if Args.dataset=='mslr' or Args.dataset=='mslrsmall' or Args.dataset=='mslr30k' or Args.dataset=='yahoo':
        loop = False

    B = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=loop, metric=None, noise=Args.noise)
    Bval = None
    if Args.dataset != 'yahoo':
        Bval = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=False, metric=None, noise=Args.noise)

    if Args.dataset == 'mslr30k' and Args.I < 20:
        order = np.load(settings.DATA_DIR+"mslr/mslr30k_train_%d.npz" % (Args.I))
        print("Setting order for Iteration %d" % (Args.I), flush=True)
        o = order['order']
        B.contexts.order = o[np.where(o < B.contexts.X)[0]]
        B.contexts.curr_idx = 0
    if Args.dataset == 'yahoo' and Args.I < 20:
        order = np.load(settings.DATA_DIR+"yahoo/yahoo_train_%d.npz" % (Args.I))
        print("Setting order for Iteration %d" % (Args.I), flush=True)
        o = order['order']
        B.contexts.order = o[np.where(o < B.contexts.X)[0]]
        B.contexts.curr_idx = 0

    print("Setting seed for Iteration %d" % (Args.I), flush=True)
    B.set_seed(Args.I)

    learning_alg = None
    if Args.learning_alg == "gb2":
        learning_alg = lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=2)
    elif Args.learning_alg == "gb5":
        learning_alg = lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=5)
    elif Args.learning_alg == "tree":
        learning_alg = lambda: sklearn.tree.DecisionTreeRegressor(max_depth=2)
    elif Args.learning_alg == "lin":
        learning_alg = lambda: sklearn.linear_model.LinearRegression()

    if Args.alg == "lin":
        L = LinUCB(B)
        if Args.param is not None:
            if os.path.isfile(outdir+"lin_%0.5f_rewards_%d.out" % (Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = L.play(Args.T, verbose=True, validate=Bval, params={'delta': Args.param})
            stop = time.time()
            np.savetxt(outdir+"lin_%0.5f_rewards_%d.out" % (Args.param,Args.I), r)
            np.savetxt(outdir+"lin_%0.5f_validation_%d.out" % (Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"lin_%0.5f_time_%d.out" % (Args.param, Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"lin_default_rewards_%d.out" % (Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = L.play(Args.T, verbose=True, validate=Bval)
            stop = time.time()
            np.savetxt(outdir+"lin_default_rewards_%d.out" % (Args.I), r)
            np.savetxt(outdir+"lin_default_validation_%d.out" % (Args.I), val_tmp)
            np.savetxt(outdir+"lin_default_time_%d.out" % (Args.I), np.array([stop-start]))
    if Args.alg == "mini":
        if learning_alg is None:
            print("Cannot run MiniMonster without learning algorithm")
            sys.exit(1)
        M = MiniMonster(B, learning_alg = learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir+"mini_%s_%0.3f_rewards_%d.out" % (Args.learning_alg,Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = M.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L), 'mu': Args.param})
            stop = time.time()
            np.savetxt(outdir+"mini_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param,Args.I), r)
            np.savetxt(outdir+"mini_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"mini_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param, Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"mini_%s_default_rewards_%d.out" % (Args.learning_alg,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = M.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L)})
            stop = time.time()
            np.savetxt(outdir+"mini_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir+"mini_%s_default_validation_%d.out" % (Args.learning_alg,Args.I), val_tmp)
            np.savetxt(outdir+"mini_%s_default_time_%d.out" % (Args.learning_alg,Args.I), np.array([stop-start]))
        
    if Args.alg == "eps":
        if learning_alg is None:
            print("Cannot run EpsGreedy without learning algorithm")
            sys.exit(1)
        E = EpsGreedy(B, learning_alg = learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir+"epsall_%s_%0.3f_rewards_%d.out" % (Args.learning_alg,Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = E.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L), 'eps': Args.param, 'train_all': True})
            stop = time.time()
            np.savetxt(outdir+"epsall_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param,Args.I), r)
            np.savetxt(outdir+"epsall_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"epsall_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param,Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"epsall_%s_default_rewards_%d.out" % (Args.learning_alg,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = E.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L), 'train_all': True})
            stop = time.time()
            np.savetxt(outdir+"epsall_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir+"epsall_%s_default_validation_%d.out" % (Args.learning_alg,Args.I), val_tmp)
            np.savetxt(outdir+"epsall_%s_default_time_%d.out" % (Args.learning_alg,Args.I), np.array([stop-start]))
    print("---- DONE ----")
