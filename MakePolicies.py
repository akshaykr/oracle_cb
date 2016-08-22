import numpy as np
import sklearn.tree
import Policy
import argparse, sys, pickle
import settings

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=['mq2007', 'mq2008', 'yahoo'])
    Args = parser.parse_args(sys.argv[1:])
    print(Args)
    if Args.dataset == 'mq2007':
        npFile = np.load(settings.BASE_DIR+"MQ2007.npz")
        K = 40
    elif Args.dataset == 'mq2008':
        npFile = np.load(settings.BASE_DIR+"MQ2008.npz")
        K = 5
    elif Args.dataset == 'yahoo':
        npFile = np.load(settings.BASE_DIR+"yahoo.npz")
        K = 20
    else:
        sys.exit(0)
    relevances = npFile['relevances']
    features = npFile['features']
    docsPerQuery = npFile['docsPerQuery']
    dim = features.shape[2]
    toretain = np.where(docsPerQuery >= K)[0]
    relevances = relevances[toretain,:K]
    features = features[toretain,:K,:]
    features = np.nan_to_num(features)
    docsPerQuery = K*np.ones(len(toretain),dtype=np.int)
    numQueries = len(docsPerQuery)    
    num_trees = 50
    num_samples = 200
    trees = []
    for t in range(num_trees):
        X = np.zeros((num_samples,dim))
        y = np.zeros(num_samples)
        for i in range(num_samples):
            q = np.random.choice(numQueries)
            d = np.random.choice(docsPerQuery[q])
            X[i,:] = features[q,d,:]
            y[i] = relevances[q,d]
        tree = sklearn.tree.DecisionTreeRegressor(max_depth=3)
        tree.fit(X,y)
        trees.append(tree)
        print("--Done training %d--" % (t))
    policies = []
    for t in range(num_trees):
        actions = {}
        for i in range(numQueries):
            preds = trees[t].predict(features[i,0:docsPerQuery[i],:])
            idx = preds.argsort()[::-1]
            actions[i] = idx
        policies.append(Policy.EnumerationPolicy(actions))
        print("--Done predicting %d--" %(t))
    pickle.dump(policies, open("./%s_trees.pkl" % (Args.dataset), "wb"))
