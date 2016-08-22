import numpy as np

class Policy(object):
    """
    A policy object prescribes actions for contexts
    The default policy class prescribes random actions
    """
    def __init__(self):
        pass
    def get_action(self, features, L):
        return np.random.choice(range(features.shape[0]), L)
    def get_all_actions(self, features, K, L):
        return [np.random.choice(range(features.shape[0]), L) for x in range(int(len(preds)/K))]

class EnumerationPolicy(Policy):
    """
    An EnumerationPolicy object enables problems with explicit enumeration of the policy
    class and the context space. 
    """
    def __init__(self, actions):
        self.actions = actions
    def get_action(self, context):
        return np.array([int(x) for x in self.actions[context.get_name()]])
    def get_weighted_action(self, context, weight):
        return np.array([int(x) for x in self.actions[context.get_name()]])
    def get_all_actions(self, contexts, features=None):
        return [self.get_action(context) for context in contexts]

class RegressionPolicy(Policy):
    """
    A RegressionPolicy object holds a sklearn model and can be used to get actions.
    Models should be regression models
    """
    def __init__(self, model):
        self.model = model
    def get_action(self, context):
        features = context.get_ld_features()
        L = context.get_L()
        preds = self.model.predict(features)
        idx = np.argsort(preds)
        return idx[len(idx)-L:len(idx)]
    def get_weighted_action(self, context, weight):
        features = context.get_ld_features()
        L = context.get_L()
        preds = self.model.predict(features)
        if "clusters" in dir(context):
            act = self.get_best_clustered_slate(preds, weight, context.clusters)
        else:
            act = self.get_best_slate(preds, weight)
        return(np.array(act))
    def get_best_slate(self, preds, weight):
        L = len(weight)
        w_inds = weight.argsort()[::-1]
        p_inds = preds.argsort()[::-1]
        act = [-1 for i in range(len(w_inds))]
        for i in range(len(w_inds)):
            if weight[w_inds[i]] > 0:
                act[w_inds[i]] = p_inds[i]
            else:
                act[w_inds[i]] = p_inds[len(p_inds)-L+i]
        return act
    def get_best_clustered_slate(self, preds, weight, clusters):
        L = len(weight)
        w_inds = weight.argsort()[::-1]
        best_act = None
        best_score = 0.0
        for c in clusters:
            act = [-1 for i in range(len(w_inds))]
            p_inds = preds[c].argsort()[::-1]
            for i in range(len(w_inds)):
                if weight[w_inds[i]] > 0:
                    act[w_inds[i]] = c[p_inds[i]]
                else:
                    act[w_inds[i]] = c[p_inds[len(p_inds)-L+i]]
            score = np.dot(preds[act], weight)
            if best_act == None or score >= best_score:
                best_act = act
                best_score = score
        return best_act
    def get_all_actions(self, contexts, features=None):
        if len(contexts) == 0:
            return []
        if features is None:
            d = contexts[0].get_ld_dim()
            f = np.zeros(d)
            for x in contexts:
                f = np.vstack((f, x.get_ld_features()))
            features=f[1:,:]
        preds = self.model.predict(features)
        out = []
        curr_idx = 0
        for x in contexts:
            K = x.get_K()
            p_tmp = np.reshape(preds[curr_idx:curr_idx+K], K)
            idx = np.argsort(p_tmp)
            out.append(idx[len(idx)-x.get_L():len(idx)])
            curr_idx += K
        return out

class ClassificationPolicy(Policy):
    """
    A ClassificationPolicy object holds a sklearn model and can be used to get actions.
    Models should be classification models
    """
    def __init__(self, model):
        self.model = model
    def get_action(self, context):
        L = context.get_L()
        assert L == 1, "Cannot implement reduction to classification for Semibandits"
        features = context.get_features()
        return [int(x) for x in self.model.predict(features)]
    def get_all_actions(self, contexts, features=None):
        Ls = np.array([x.get_L() for x in contexts])
        assert np.all(Ls == 1), "Cannot implement reduction to classification for Semibandits"
        if len(contexts) == 0:
            return []
        if features is None:
            d = contexts[0].get_dim()
            f = np.zeros((1, d))
            for x in contexts:
                f = np.vstack((f, x.get_features()))
            features = f[1:,:]
        preds = self.model.predict(features)
        return [int(x) for x in preds]

