import numpy as np
import sklearn.linear_model
import sklearn.tree
import Simulators, Logger, Evaluators, Semibandits, Metrics
import warnings
import argparse
import pickle
import sys

class RegretExp(object):
    def __init__(self, weight=None, link="linear", K=10, L=5, T=1000, dataset="synth", feat_noise=0.25, reward_noise=1.0, policies="finite", structure='none'):
        self.T = T
        self.K = K
        self.L = L
        if weight == None:
            weight = np.arange(1,self.L+1)
        self.weight = weight
        self.link = link
        self.feat_noise = feat_noise
        self.reward_noise = reward_noise
        self.dataset = dataset
        self.policies = policies
        self.structure = structure
        if self.dataset == "synth":
            print("----Generating Semibandit Simulator----")
            self.Sim = Simulators.OrderedSBSim(100,100,self.K,
                                               self.L,self.feat_noise,
                                               w_vec=self.weight,
                                               link=self.link,
                                               one_pass=False)
            print("----Done----")
        elif self.dataset == "mq2007":
            print("----Generating MQ2007 Simulator----")
            self.Sim = Simulators.DatasetBandit(self.L,loop=True,
                                                dataset='mq2007',
                                                metric=Metrics.NavigationalTTS,
                                                ## metric=Metrics.NDCG,
                                                structure=self.structure)
            if self.policies == "finite":
                trees = pickle.load(open('./mq2007_trees.pkl', 'rb'))
                self.Sim.set_policies(trees)

            print("----Done----")
        elif self.dataset == "mq2008":
            print("----Generating MQ2008 Simulator----")
            self.Sim = Simulators.DatasetBandit(self.L,loop=True,
                                                dataset='mq2008',
                                                metric=Metrics.NavigationalTTS,
                                                structure=self.structure)
            if self.policies == "finite":
                trees = pickle.load(open('./mq2008_trees.pkl', 'rb'))
                self.Sim.set_policies(trees)
            print("----Done----")
        elif self.dataset == 'yahoo':
            print("----Generating Yahoo Simulator----")
            self.Sim = Simulators.DatasetBandit(self.L,loop=True,
                                                dataset='yahoo',
                                                ## metric=Metrics.NDCG,
                                                metric=Metrics.NavigationalTTS,
                                                structure=self.structure)
            if self.policies == "finite":
                trees = pickle.load(open('./yahoo_trees.pkl', 'rb'))
                self.Sim.set_policies(trees)
            print("----Done----")
        else:
            print("Error invalid dataset")
            sys.exit(1)


    def run_alg(self, Alg, params={}):
        A = Alg(self.Sim)
        (reward, regret) = A.play(self.T,params=params,verbose=False)
        return (reward, regret)

if __name__=='__main__':
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store',
                        default=1000,
                        help='number of rounds')
    parser.add_argument('--link', action='store', choices=['linear', 'logistic'], default='linear')
    parser.add_argument('--dataset', action='store', choices=['synth','mq2007','mq2008', 'yahoo'])
    parser.add_argument('--policies', action='store', choices=['finite', 'tree', 'linear'], default='linear')
    parser.add_argument('--K', action='store', default=10)
    parser.add_argument('--L', action='store', default=5)
    parser.add_argument('--structure', action='store', default='none', choices=['none','cluster'])

    Args = parser.parse_args(sys.argv[1:])
    print(Args)
    Args.T = int(Args.T)
    Args.K = int(Args.K)
    Args.L = int(Args.L)

    weight = np.arange(1,Args.L+1)[::-1]  ## np.arange(1,Args.L+1,1)[::-1] ## /np.sum(np.arange(1,Args.L+1))

    Algs = {
        ## 'EELS': Semibandits.EELS,
        'EELS2': Semibandits.EELS2,
        ## 'Eps': Semibandits.EpsGreedy,
        'EpsOracle': Semibandits.EpsGreedy,
        ## 'Random': Semibandits.Semibandit
        }
        
    Params = {
        'EELS': {
            'link': Args.link,
            },
        'EELS2': {
            'link': Args.link,
            },
        'Eps': {
            'reward': True,
            },
        'EpsOracle': {
            'reward': False, 
            'weight': weight,
            'link': Args.link
            },
        'Random': {}
        }

    if Args.dataset != "synth" and Args.policies == 'tree':
        Params['EELS']['learning_alg'] = sklearn.tree.DecisionTreeRegressor
        Params['EELS2']['learning_alg'] = sklearn.tree.DecisionTreeRegressor
        Params['Eps']['learning_alg'] = sklearn.tree.DecisionTreeRegressor
        Params['EpsOracle']['learning_alg'] = sklearn.tree.DecisionTreeRegressor
        
    if Args.dataset != "synth" and Args.policies == 'linear':
        Params['EELS']['learning_alg'] = sklearn.linear_model.LinearRegression
        Params['EELS2']['learning_alg'] = sklearn.linear_model.LinearRegression
        Params['Eps']['learning_alg'] = sklearn.linear_model.LinearRegression
        Params['EpsOracle']['learning_alg'] = sklearn.linear_model.LinearRegression

    Out = {
        'EELS': [],
        'EELS_regret': [],
        'EELS2': [],
        'EELS2_regret': [],
        'Eps': [],
        'Eps_regret': [],
        'EpsOracle': [],
        'EpsOracle_regret': [],
        'Random': [],
        'Random_regret': []
        }

    Exp = RegretExp(weight = weight, link=Args.link, K=Args.K, L=Args.L, T=Args.T, dataset=Args.dataset, policies=Args.policies,structure=Args.structure)
    for i in range(10):
        print('----Iter %d----' % (i))
        for (k,v) in Algs.items():
            print('----Running %s with params %s----' % (k, Params[k]))
            (reward, regret) = Exp.run_alg(v, params=Params[k])
            Out[k].append(reward)
            Out[k+"_regret"].append(regret)
            print('%s final: %0.3f' % (k, reward[-1]))

    pickle.dump(Out, open("./data/%s_%s_%s_link=%s_T=%d_K=%d_L=%d.pkl" %(Args.dataset, Args.policies, Args.structure, Args.link, Args.T, Args.K, Args.L), "wb"))
