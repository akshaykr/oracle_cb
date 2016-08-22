import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import Simulators
import pickle

if __name__=='__main__':
    L = 3

    B = Simulators.DatasetBandit(dataset='mq2008', L=3, loop=True, metric=None)
    Bval = Simulators.DatasetBandit(dataset='mq2008val', L=3, loop=True, metric=None)
    
    forest_sizes = np.arange(10,101, 10)
    depths = np.arange(1, 6, 1)

    iters = 10
    
    out1 = {}
    out2 = {}

    out1['linear'] = []
    out2['linear'] = []

    for i in range(iters):
        model = B.get_best_policy(learning_alg=lambda: sklearn.linear_model.LinearRegression(), classification=False)
        s1 = B.offline_evaluate(model)
        s2 = Bval.offline_evaluate(model)
                
        out1['linear'].append(s1)
        out2['linear'].append(s2)
        print("Linear: %d s1: %0.3f s2: %0.3f" % (i, s1, s2), flush=True)
        

    for a in forest_sizes:
        for b in depths:
            out1['forest_%d_%d' % (a,b)] = []
            out2['forest_%d_%d' % (a,b)] = []
            out1['gradient_%d_%d' % (a,b)] = []
            out2['gradient_%d_%d' % (a,b)] = []            
            for i in range(iters):
                model = B.get_best_policy(learning_alg=lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=a, max_depth=b), classification=False)
                s1 = B.offline_evaluate(model)
                s2 = Bval.offline_evaluate(model)
                
                out1['forest_%d_%d' % (a,b)].append(s1)
                out2['forest_%d_%d' % (a,b)].append(s2)
                print("Forest: %d %d %d s1: %0.3f s2: %0.3f" % (a,b,i, s1, s2), flush=True)

                model = B.get_best_policy(learning_alg=lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=a, max_depth=b), classification=False)
                s1 = B.offline_evaluate(model)
                s2 = Bval.offline_evaluate(model)
                
                out1['gradient_%d_%d' % (a,b)].append(s1)
                out2['gradient_%d_%d' % (a,b)].append(s2)
                print("Gradient: %d %d %d s1: %0.3f s2: %0.3f" % (a,b,i, s1, s2), flush=True)

    pickle.dump(dict(train=out1, val=out2), open("./out/mq2008_offline_L=%d.out" % (L), "wb"))
