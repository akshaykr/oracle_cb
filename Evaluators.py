import numpy as np
import Logger, Simulators, Util

class Evaluator(object):
    name="Trivial"
    def __init__(self, Sim):
        self.Sim = Sim
        self.needs_training = False
    def train(self, dataset):
        pass
    def estimate(self, dataset, pi):
        return 0.0

class IPSEvaluator(object):
    name = "IPS"
    def __init__(self, Sim):
        self.needs_training = False
        self.Sim = Sim
    def train(self, dataset):
        pass
    def estimate(self, dataset, pi):
        return np.mean([d.slate_feedback/d.slate_prob*Util.slate_eq_ind(d.action, pi.get_action(d.context)) for d in dataset])

class CounterfactualEvaluator(object):
    name = "CF"
    def __init__(self, Sim):
        self.Sim = Sim
        self.needs_training = False
        self.weight = Sim.weight
        self.link = Sim.link
    def train(self, dataset):
        pass
    def estimate(self, dataset, pi):
        return np.mean([self.link(np.dot(self.weight,d.all_feedback[pi.get_action(d.context)])) for d in dataset])

class SkyRegressionEvaluator(object):
    name = "SkyRegress"
    def __init__(self, Sim):
        self.Sim = Sim
        self.K = Sim.K
        self.L = Sim.L
        self.needs_training = False
        self.weight = Sim.weight
        self.link = Sim.link
    def train(self, dataset):
        pass
    def estimate(self, dataset, pi):
        iw_vecs = []
        for d in dataset:
            iw_vec = np.zeros(self.K)
            iw_vec[d.action] = d.action_feedback/d.action_probs
            iw_vecs.append(np.matrix(iw_vec).T)
        return np.mean([self.link(self.weight.T*iw_vecs[i][pi.get_action(dataset[i].context)]) for i in range(len(dataset))])
        
class SemibanditEvaluator(object):
    name = "Semibandit"
    def __init__(self, Sim):
        self.Sim = Sim
        self.K = Sim.K
        self.L = Sim.L
        self.needs_training = False
        self.weight = np.ones(self.L)
        self.link = Sim.link
    def train(self, dataset):
        pass
    def estimate(self, dataset, pi):
        iw_vecs = []
        for d in dataset:
            iw_vec = np.zeros(self.K)
            iw_vec[d.action] = d.action_feedback/d.action_probs
            iw_vecs.append(np.matrix(iw_vec).T)
        return np.mean([self.link(self.weight.T*iw_vecs[i][pi.get_action(dataset[i].context)]) for i in range(len(dataset))])


class RegressionEvaluator(object):
    name = "Regress"
    def __init__(self, Sim):
        self.Sim = Sim
        self.K = Sim.K
        self.L = Sim.L
        self.link = Sim.link
        self.w_hat = None
        self.needs_training = False
    def train(self, dataset):
        pass
    def estimate(self, dataset, pi):
        ## First estimate regression weights
        Sigma = np.matrix(np.zeros((self.L,self.L)))
        vec = np.matrix(np.zeros((self.L,1)))
        for d in dataset:
            Sigma += np.matrix(d.action_feedback).T*np.matrix(d.action_feedback)
            vec += np.matrix(d.action_feedback).T*d.slate_feedback
        self.w_hat = np.linalg.pinv(Sigma)*vec
        self.w_hat
        iw_vecs = []
        for d in dataset:
            iw_vec = np.zeros(self.K)
            iw_vec[d.action] = d.action_feedback/d.action_probs
            iw_vecs.append(np.matrix(iw_vec).T)
        return np.mean([self.link(self.w_hat.T*iw_vecs[i][pi.get_action(dataset[i].context)]) for i in range(len(dataset))])


def EvaluateExperiment(n,K,L):
    print("Deprecated")
    return(0)
    eps = 0.25
    w_vec = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    link = None
    S = Simulators.OrderedSBSim(100, 100, K, L, eps, w_vec=w_vec, link=link, one_pass=False, reward_noise=10.0)
    pi = S.Pi[0]
    pistar = S.Pi[S.Pistar]
    ## first compute true expectation,
    score = 0.0
    for x in range(S.X):
        A = pi.get_action(S.contexts[x])
        Astar = pistar.get_action(S.contexts[x])
        mean_reward = [0.5+eps if A[i] in Astar else 0.5-eps for i in range(len(A))]
        score += np.dot(S.weight, np.array(mean_reward))
    truth = score/S.X
    print("Starting experiment, ground truth = %0.3f" % (truth))
    I = IPSEvaluator()
    C = CounterfactualEvaluator(w_vec=w_vec, link=link)
    Log = Logger.Logger(S)
    Log.collect_uniform_log(n)
    dataset = Log.data
    i_tmp = I.estimate(dataset, pi)
    c_tmp = C.estimate(dataset, pi)    
    i_scores = np.cumsum(i_tmp)/range(1,len(dataset)+1)
    c_scores = np.cumsum(c_tmp)/range(1,len(dataset)+1)
    i_mse = (i_scores - truth)**2
    c_mse = (c_scores - truth)**2
    i_mse = i_mse[100::100]
    c_mse = c_mse[100::100]
    R = RegressionEvaluator(K, L)
    S = SkyRegressionEvaluator(K, L, w_vec=w_vec)
    r_scores = []
    s_scores = []
    for i in range(100, n, 100):
        r_scores.append(R.estimate(dataset[0:i], pi))
        s_scores.append(S.estimate(dataset[0:i], pi))
    r_mse = (r_scores - truth)**2
    s_mse = (s_scores - truth)**2
    return (np.arange(100,n,100), i_mse, c_mse,r_mse,s_mse)
