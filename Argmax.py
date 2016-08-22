import numpy as np
from Policy import *

"""
Factored out implementation of argmax oracle.  Takes in a bandit
object, a dataset, and a policy class type, and optionally a learning
algorithm and produces the policy with the "best" empirical reward on
the dataset.

policy_type can be EnumerationPolicy, ClassificationPolicy, or
RegressionPolicy.

EnumerationPolicy is the fully synthetic setting, where the policy
class can be explicitly enumerated.

Classification policy implements a reduction from cost-sensitive
classification to importance weighted multiclass classification where
each reward sensitive example is converted into K multiclass examples,
one for each label, and the importance weights are related to the
rewards. This is a consistent reduction, but it does increase the
dataset size by a factor of K.

RegreesionPolicy regresses onto the importance weighted rewards, where
the weights are inside the squared loss rather than outside.
"""
def argmax(B, dataset, policy_type=EnumerationPolicy, learning_alg=None):
    """
    Prepare dataset for AMO, make AMO call and
    return a Policy object with the associated model. 
    """
    if policy_type == EnumerationPolicy:
        ## Enumerate over dataset
        rewards = []
        for policy in B.Pi:
            score = 0.0
            for item in dataset:
                score += np.sum(item[1][policy.get_action(item[0])])
            rewards.append(score)
        idx = np.argmax(rewards)
        return B.Pi[idx]
    elif policy_type == ClassificationPolicy:
        x = dataset[0][0]
        X = np.zeros((len(dataset)*B.K, x.get_dim()))
        y = np.zeros(len(dataset)*B.K)
        w = np.zeros(len(dataset)*B.K)
        curr_idx = 0
        for item in dataset:
            context = item[0]
            reward = item[1]
            X[curr_idx:curr_idx+B.K,:] = context.get_features()
            y[curr_idx:curr_idx+B.K] = range(B.K)
            w[curr_idx:curr_idx+B.K] = reward
            curr_idx += B.K
        ## Call importance weighted classification oracle
        pred = learning_alg()
        pred.fit(X,y, sample_weight=w)

        return ClassificationPolicy(pred)
    else:
        assert policy_type == RegressionPolicy, "Unsupported policy type"
        ## Get dataset into sklearn format
        X = np.zeros((len(dataset)*B.K, B.d))
        y = np.zeros(len(dataset)*B.K)
        curr_idx = 0
        for item in dataset:
            context = item[0]
            reward = item[1]
            X[curr_idx:curr_idx+B.K,:] = context.get_ld_features()
            y[curr_idx:curr_idx+B.K] = reward
            curr_idx += B.K
        ## Call regression oracle
        pred = learning_alg()
        pred.fit(X,y)
        ## Return a policy
        return RegressionPolicy(pred)

def argmax2(B, dataset, policy_type=RegressionPolicy,learning_alg=None):
    assert policy_type == RegressionPolicy, "Other policies not supported"
    assert learning_alg != None, "Must specify learning algorithm"

    X = np.zeros((len(dataset)*B.K, B.d))
    y = np.zeros(len(dataset)*B.K)
    w = np.zeros(len(dataset)*B.K)
    curr_idx = 0
    for item in dataset:
        context = item[0]
        act = item[1]
        reward = item[2]
        weight = item[3]
        X[curr_idx:curr_idx+len(act),:] = context.get_ld_features()[act,:]
        y[curr_idx:curr_idx+len(act)] = reward[act]
        w[curr_idx:curr_idx+len(act)] = weight[act]
        curr_idx += len(act)
    X = X[0:curr_idx,:]
    y = y[0:curr_idx]
    w = w[0:curr_idx]
    ## Call regression oracle
    pred = learning_alg()
    pred.fit(X,y,sample_weight=w)
    ## Return a policy
    return RegressionPolicy(pred)

def weighted_argmax(B,dataset,weights,link="linear", policy_type=EnumerationPolicy,learning_alg=None,intercept=0.0):
    if policy_type == EnumerationPolicy:
        rewards = []
        for policy in B.Pi:
            score = 0.0
            for item in dataset:
                yvec = item[1][policy.get_action(item[0])]
                if link == "linear":
                    score += np.dot(np.matrix(yvec),weights) + intercept
                elif link == "logistic":
                    score += 1.0/(1+np.exp(-np.dot(np.matrix(yvec),weights)-intercept))
            rewards.append(score)
        idx = np.argmax(rewards)
        return B.Pi[idx]
    elif policy_type == RegressionPolicy:
        X = np.zeros((len(dataset)*B.K, B.d))
        y = np.zeros(len(dataset)*B.K)
        curr_idx = 0
        for item in dataset:
            context = item[0]
            reward = item[1]
            X[curr_idx:curr_idx+B.K,:] = context.get_ld_features()
            y[curr_idx:curr_idx+B.K] = reward
            curr_idx += B.K
        ## Call regression oracle
        pred = learning_alg()
        pred.fit(X,y)
        ## Return a policy
        return RegressionPolicy(pred)
    else:
        raise Exception("Unsupported Policy Type")

def reward_argmax(B,dataset,policy_type=EnumerationPolicy,learning_alg=None):
    if policy_type == EnumerationPolicy:
        rewards = []
        for policy in B.Pi:
            score = 0.0
            for item in dataset:
                if (np.array(policy.get_action(item[0])) == np.array(item[1])).all():
                    score += item[2]
            rewards.append(score)
        idx = np.argmax(rewards)
        return B.Pi[idx]
    elif policy_type == RegressionPolicy:
        X = np.zeros((len(dataset), B.d*B.L))
        y = np.zeros(len(dataset))
        curr_idx = 0
        for item in dataset:
            context = item[0]
            act = item[1]
            reward = item[2]
            vec = np.hstack(context.get_ld_features()[act,:])
            print(vec.shape)
            X[curr_idx,:] = np.hstack(context.get_ld_features()[act,:])
            y[curr_idx] = reward
            curr_idx += 1
        ## Call regression oracle
        pred = learning_alg()
        pred.fit(X,y)

        ## Return a policy
        return RegressionPolicy(pred)
    else:
        raise Exception("Unsupported Policy Type")
