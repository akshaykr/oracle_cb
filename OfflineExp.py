import numpy as np
import Simulators, Logger, Evaluators
import warnings
import argparse
import pickle
import sys

class OfflineExp(object):
    def __init__(self, n=1000, K=10, L=5, dataset="synth", feat_noise=0.25, reward_noise=1.0):
        self.n = n
        self.K = K
        self.L = L
        self.weight = np.array([1.0/np.log2(i) for i in range(2,self.L+2)])
        self.feat_noise = feat_noise
        self.reward_noise = reward_noise
        self.dataset = dataset
        if self.dataset == "synth":
            print("----Generating Semibandit Simulator----")
            self.Sim = Simulators.OrderedSBSim(100, 100, self.K, self.L, self.feat_noise, w_vec=self.weight, link=None, one_pass=False, reward_noise=self.reward_noise)
            print("----Done----")
            self.Policy = self.Sim.Pi[0]
            if self.Sim.Pistar == 0:
                self.Policy = self.Sim.Pi[1]
        else:
            print("Error invalid dataset")
            sys.exit(1)
        
    def generate_new_dataset(self):
        L = Logger.Logger(self.Sim)
        L.collect_uniform_log(self.n)
        self.dataset = L.data

    def score_policy(self):
        pistar = self.Sim.Pi[self.Sim.Pistar]
        score = 0.0
        for x in range(self.Sim.X):
            A = self.Policy.get_action(self.Sim.contexts[x])
            Astar = pistar.get_action(self.Sim.contexts[x])
            mean_reward = [0.5+self.feat_noise if A[i] in Astar else 0.5-self.feat_noise for i in range(len(A))]
            score += np.dot(self.weight, np.array(mean_reward))
        truth = score/self.Sim.X
        return(truth)

    def eval_estimator(self, Estimator, start, step):
        E = Estimator(self.Sim)
        P = self.Policy
        dataset = self.dataset
        if E.needs_training:
            ## Split dataset in half and train
            E.train(dataset[0:int(self.n/2)])
            dataset = dataset[int(self.n/2):]
        scores = []
        for i in range(start, len(dataset), step):
            scores.append(E.estimate(dataset[0:i], P))
        return np.array(scores)

if __name__=='__main__':
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', action='store',
                        default=1000,
                        help='number of bootstrap samples')
    parser.add_argument('--dataset', action='store', choices=['synth'])
    parser.add_argument('--feat_noise', action='store', default=0.25)
    parser.add_argument('--reward_noise', action='store', default=1.0)
    parser.add_argument('--K', action='store', default=10)
    parser.add_argument('--L', action='store', default=5)
    Args = parser.parse_args(sys.argv[1:])
    Args.n = int(Args.n)
    Args.K = int(Args.K)
    Args.L = int(Args.L)
    Args.feat_noise = float(Args.feat_noise)
    Args.reward_noise = float(Args.reward_noise)
    print("----Arguments----")
    print(Args)
    Estimators = [
        Evaluators.IPSEvaluator,
        Evaluators.CounterfactualEvaluator,
        Evaluators.RegressionEvaluator,
        Evaluators.SkyRegressionEvaluator,
        Evaluators.SemibanditEvaluator,
        ]

    scores = {}
    for alg in Estimators:
        scores[alg.name] = []
        scores[alg.name+"_mse"] = []
        
    O = OfflineExp(n=Args.n, K=Args.K, L=Args.L,
                   dataset=Args.dataset,
                   feat_noise=Args.feat_noise,
                   reward_noise=Args.reward_noise)
    print("----Scoring Target Policy----")
    target = O.score_policy()
    print("----Target Policy Score: %0.2f----" % (target))
    start = 100
    step = 100

    for i in range(100):
        O.generate_new_dataset()
        print("Starting Experiment")
        n = O.n
        for alg in Estimators:
            arr = O.eval_estimator(alg,start,step)
            scores[alg.name].append(arr)
            scores[alg.name+"_mse"].append((scores[alg.name][i]-target)**2)
            print("Iter:%d %s final_est=%0.3f final_mse=%0.3e" % (i, alg.name, scores[alg.name][i][-1], scores[alg.name+"_mse"][i][-1]))
    pickle.dump((scores, range(start, Args.n, step)), open("./data/%s_n=%d_k=%d_l=%d_e1=%0.2f_e2=%0.2f.pkl" % (Args.dataset, Args.n, Args.K, Args.L, Args.feat_noise, Args.reward_noise), "wb"))
