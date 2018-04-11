import numpy as np
import sys
import sklearn.tree
import Context, Argmax, Metrics
from Policy import *
import string
import ContextIterators
import Metrics

class BanditSim(object):
    def __init__(self, X, N, K, eps, one_pass=False, reward_noise=0.0):
        """
        Simulate a K-armed contextual bandit game with X contexts, N
        policies, and K actions with gap eps.  Use the uniform
        distribution over contexts.
        
        Policies are effectively random.  Pick one policy to be the best
        and for each context, the reward distribution Ber(1/2+eps) on the
        action of that policy and Ber(1/2-eps) on the other actions.

        Exposes methods:
        get_new_context() -- construct and return new context
        get_num_actions() -- enables context-specific K
        get_reward(a) -- get reward for an action
        get_all_rewards() -- return rewards for all actions.
        get_best_reward() -- returns reward for pi^\star for book-keeping purposes. 

        @Deprecated
        """
        
        self.L = 1
        self.N = N
        self.K = K
        self.eps = eps
        self.X = X
        self.one_pass = one_pass
        self.reward_noise = reward_noise

        ## Initialize the contexts.
        self.contexts = []
        for i in range(self.X):
            self.contexts.append(Context.Context(i, np.zeros(1), self.K, self.L))
        ## Initialize the policies.
        piMat = np.matrix(np.random.randint(0, K, size=[N,X,self.L]))
        self.Pi = []
        for i in range(N):
            pi = EnumerationPolicy(dict(zip(range(self.X), [piMat[i,j] for j in range(self.X)])))
            self.Pi.append(pi)
        ## Initialize the optimal policy
        self.Pistar = np.random.randint(0, N)

    def get_new_context(self):
        if self.one_pass:
            self.curr_idx += 1
            if self.curr_idx >= self.X:
                return None
            self.curr_x = self.contexts[self.curr_idx]
        else:
            self.curr_x = self.contexts[np.random.randint(0, self.X)]
        ## r = np.random.binomial(1, 0.5-self.eps, self.K)
        r = (np.random.binomial(1, 0.5-self.eps, self.K)*2)-1
        r[self.Pi[self.Pistar].get_action(self.curr_x)] = (np.random.binomial(1, 0.5+self.eps, self.L)*2)-1
        self.curr_r = r
        self.r_noise = np.random.normal(0, self.reward_noise)
        return self.curr_x

    def get_num_actions(self):
        return self.curr_x.get_K()

    def get_curr_context(self):
        return self.curr_x

    def get_reward(self, a):
        return self.curr_r[a]
    
    def get_all_rewards(self):
        return self.curr_r

    def get_best_reward(self):
        return np.sum(self.curr_r[self.Pi[self.Pistar].get_action(self.curr_x)])


class SemibanditSim(BanditSim):
    def __init__(self, X, N, K, L, eps, one_pass=False):
        """
        Simulate a contextual semi-bandit problem with X contexts, N
        policies, K base actions and action lists of length L. Use the
        uniform distribution over contexts.

        Policies are effectively random, on each context they play L
        random actions.  One policy is optimal and the reward
        distribution is Ber(1/2+eps) on each base action played by
        that policy and Bet(1/2-eps) on each other base action.

        Additionally exposes:
        get_slate_reward(A) -- total reward for a slate
        get_base_rewards(A) -- rewards for each base action in slate.
        """

        self.L = L
        self.N = N
        self.K = K
        self.eps = eps
        self.X = X
        self.K = K
        self.one_pass = False

        ## Initialize the contexts.
        self.contexts = []
        for i in range(self.X):
            self.contexts.append(Context.Context(i, np.zeros((1,1)), self.K, self.L))
        ## Initialize the policies.
        piMat = np.zeros((N,X,L), dtype=np.int)
        for n in range(N):
            for x in range(X):
                piMat[n,x,:] = np.random.choice(range(K), L, replace=False)
        self.Pi = []
        for i in range(N):
            pi = EnumerationPolicy(dict(zip(range(self.X), [piMat[i,j,:] for j in range(self.X)])))
            self.Pi.append(pi)
        ## Initialize the optimal policy
        self.Pistar = np.random.randint(0, N)
        self.curr_idx = -1

    def get_slate_reward(self, A):
        return np.sum(self.curr_r[A]) + self.r_noise

    def get_base_rewards(self, A):
        return self.curr_r[A]

class OrderedSBSim(SemibanditSim):
    def __init__(self, X, N, K, L, eps, w_vec=None, link="linear", one_pass=False,reward_noise=0.1):
        self.X = X
        self.N = N
        self.K = K
        self.L = L
        self.eps = eps
        self.one_pass = one_pass
        self.link = link
        if w_vec is None:
            self.weight = np.ones(L)
        else:
            self.weight = w_vec
        self.reward_noise=0.1
        assert len(self.weight) == self.L
        
        ## Initialize contexts.
        self.contexts = []
        for i in range(self.X):
            self.contexts.append(Context.Context(i,np.zeros((1,1)), self.K, self.L))
        piMat = np.zeros((N,X,L), dtype=np.int)
        for n in range(N):
            for x in range(X):
                piMat[n,x,:] = np.random.choice(range(K), L, replace=False)
        self.Pi = []
        for i in range(N):
            pi = EnumerationPolicy(dict(zip(range(self.X), [piMat[i,j,:] for j in range(self.X)])))
            self.Pi.append(pi)
        ## Initialize the optimal policy 
        self.Pistar = np.random.randint(0, N)
        self.curr_idx = -1

    def get_slate_reward(self, A):
        if self.link == "linear":
            return np.dot(self.weight,self.curr_r[A]) + self.r_noise
        if self.link == "logistic":
            ## return np.random.binomial(1, 1.0/(1+np.exp(-np.dot(self.weight, self.curr_r[A]))))
            return np.random.binomial(1, 1.0/(1.0+np.exp(-np.dot(self.weight, self.curr_r[A]))))
        ## return self.link(np.dot(self.weight,self.curr_r[A])) + self.r_noise

    def get_best_reward(self):
        return self.get_slate_reward(self.Pi[self.Pistar].get_action(self.curr_x))


class MQBandit(SemibanditSim):
    def __init__(self, dataset, L):
        """
        Use the MQ style dataset as a contextual semi-bandit problem.
        This dataset consists of (query, doc, relevance) triples where
        each query,doc pair also comes with a feature vector.
        
        L is slate length

        On each round for this problem, the learner plays L documents
        and the total reward is the sum of the relevances for those
        documents (TODO: maybe incorporate some weights). The
        relevances are expose via the get_base_rewards method. Thus it
        is a semibandit problem.

        We are use a linear policies for now.  TODO: If "policies" is
        set to None or unspecified, then we allow all linear
        functions.  If "policies" is assigned some number N, then we
        draw N random unit vectors and use those as policies. Here we
        explicitly enumerate the policy class and build the (context,
        policy, slate) table.

        """
        self.L = L
        self.policies = (policies != None)
        if dataset == "MQ2008":
            self.loadMQ2008()
        elif dataset == "MQ2007":
            self.loadMQ2007()
        else:
            print("Error misspecified dataset")
            return

        numQueries, numDocs, numFeatures = np.shape(self.features)
        print("Datasets:loadNpz [INFO] Loaded",
            " NumQueries, [Min,Max]NumDocs, totalDocs, MaxNumFeatures: ", numQueries, np.min(self.docsPerQuery), np.max(self.docsPerQuery), numDocs, numFeatures)
        sys.stdout.flush()

        self.K = np.max(self.docsPerQuery)
        self.X = len(self.docsPerQuery)
        self.d = self.features.shape[2]

        pistar = self.get_best_policy(learning_alg=lambda: sklearn.tree.DecisionTreeClassifier(max_depth=5), classification=True)
        self.Pi = [pistar]
        self.Pistar = 0

        self.curr_x = None
        self.curr_r = None

    def loadMQ2008(self):
        npFile = np.load("MQ2008.npz")
        self.relevances = npFile['relevances']/2
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']

        self.relevances = self.relevances[:,:np.min(self.docsPerQuery)]
        self.features = self.features[:,:np.min(self.docsPerQuery),:]
        self.docsPerQuery = np.min(self.docsPerQuery)*np.ones(len(self.docsPerQuery))
        self.docsPerQuery = np.array(self.docsPerQuery, dtype=np.int)

    def loadMQ2007(self):
        npFile = np.load("MQ2007.npz")
        self.relevances = npFile['relevances']/2
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']

        ## Filtering to make K much smaller. 
        toretain = np.where(self.docsPerQuery >= 40)[0]
        self.relevances = self.relevances[toretain,:]
        self.features = self.features[toretain,:]
        self.docsPerQuery = self.docsPerQuery[toretain]

        self.relevances = self.relevances[:,:np.min(self.docsPerQuery)]
        self.features = self.features[:,:np.min(self.docsPerQuery),:]
        self.docsPerQuery = np.min(self.docsPerQuery)*np.ones(len(self.docsPerQuery))
        self.docsPerQuery = np.array(self.docsPerQuery, dtype=np.int)

    def get_new_context(self):
        context_idx = np.random.randint(0, self.X)
        self.curr_x = Context.Context(context_idx, self.features[context_idx,:,:], self.docsPerQuery[context_idx], self.L)
        self.curr_r = self.relevances[context_idx,:]
        return self.curr_x

    def get_best_reward(self):
        if "Pi" in dir(self):
            return np.sum(self.curr_r[self.Pi[self.Pistar, self.curr_x,:]])
        else:
            idx = np.argsort(self.curr_r)
            return np.sum(self.curr_r[idx[len(self.curr_r)-self.L:len(self.curr_r)]])

    def get_best_policy(self, learning_alg=None, classification=True):
        if learning_alg == None:
            return self.Pi[self.Pistar]
        else:
            ## Prepare dataset
            ## Train best depth 3 decision tree.
            dataset = []
            for x in range(self.X):
                context = Context.Context(x, self.features[x,:,:], self.docsPerQuery[x], self.L)
                dataset.append((context, self.relevances[x,:]))
            if classification:
                return Argmax.argmax(self, dataset, policy_type=ClassificationPolicy, learning_alg=learning_alg)
            else:
                return Argmax.argmax(self, dataset, policy_type=RegressionPolicy, learning_alg=learning_alg)

    def offline_evaluate(self, policy):
        score = 0.0
        for x in range(self.X):
            score += np.sum(self.relevances[x,policy.get_action(Context.Context(x, self.features[x,:,:], self.docsPerQuery[x], self.L))])
        return score

class LinearBandit(SemibanditSim):
    class LinearContext():
        def __init__(self,name,features):
            self.features = features
            self.name = name
        def get_ld_features(self):
            return self.features
        def get_K(self):
            return self.features.shape[0]
        def get_L(self):
            return 1
        def get_ld_dim(self):
            return self.features.shape[1]
        def get_name(self):
            return self.name

    def __init__(self, d, L, K, noise=False, seed=None, pos=False, quad=False):
        """
        A Linear semi-bandit simulator. Generates a random unit weight
        vector upon initialization. At each round, random unit-normed
        feature vectors are drawn for each action, and reward is (if
        noise) a bernoulli with mean (1+x_a^Tw)/2 or just (1+x_a^Tw)/2. 
        
        The learner plays a slate of L actions and K actions are
        available per round.

        d is the dimension of the feature space
        L actions per slate
        K actions per context

        """
        self.d = d
        self.L = L
        self.K = K
        self.N = None
        self.X = None
        self.noise = noise
        self.seed = seed
        self.pos = pos
        self.quad = quad

        if seed is not None:
            np.random.seed(574)

        if self.pos:
            self.weights = np.matrix(np.random.dirichlet(d*np.ones(self.d))).T
            self.weights = self.weights/np.linalg.norm(self.weights)
        else:
            self.weights = np.matrix(np.random.normal(0, 1, [self.d,1]))
            self.weights = self.weights/np.sqrt(self.weights.T*self.weights)

        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.curr_x = None
        self.features = None
        self.all_features = []
        self.curr_r = None
        self.curr_x = None

    def get_new_context(self):
        ## Generate random feature matrix and normalize.
        if self.seed is not None:
            np.random.seed((self.t+17)*(self.seed+1) + 37)

        if self.pos:
            self.features = np.matrix(np.random.dirichlet(1.0/self.d*np.ones(self.d), self.K))
        else:
            self.features = np.matrix(np.random.normal(0, 1, [self.K, self.d]))

        self.features = np.diag(1./np.sqrt(np.diag(self.features*self.features.T)))*self.features
        self.all_features.append(self.features)

        self.curr_means = np.array((self.features*self.weights).T)[0]
        if self.quad:
            self.curr_means = self.curr_means**2
        if self.noise and type(self.noise) == float:
            self.noise_term = np.random.normal(0,self.noise)
            self.curr_r = np.array(self.curr_means+self.noise_term)
        elif self.noise:
            self.noise_term = np.random.normal(0, 0.1)
            self.curr_r = np.array(self.curr_means+self.noise_term)
        else:
            self.curr_r = np.array(self.curr_means)

        old_t = self.t
        self.t += 1
        self.curr_x = LinearBandit.LinearContext(self.t, self.features)
        return self.curr_x

    def get_best_reward(self):
        idx = np.argsort(self.curr_means)
        return np.sum(self.curr_r[idx[len(idx)-self.L:len(idx)]])

    def get_slate_reward(self, A):
        return self.curr_r[A]

class SemiparametricBandit(SemibanditSim):
    class LinearContext():
        def __init__(self,name,features):
            self.features = features
            self.name = name
        def get_ld_features(self):
            return self.features
        def get_K(self):
            return self.features.shape[0]
        def get_L(self):
            return 1
        def get_ld_dim(self):
            return self.features.shape[1]
        def get_name(self):
            return self.name

    def __init__(self, d, L, K, noise=False, seed=None, pos=False):
        """
        A Linear semi-bandit simulator. Generates a random unit weight
        vector upon initialization. At each round, random unit-normed
        feature vectors are drawn for each action, and reward is (if
        noise) a bernoulli with mean (1+x_a^Tw)/2 or just (1+x_a^Tw)/2.

        The learner plays a slate of L actions and K actions are
        available per round.

        d is the dimension of the feature space
        L actions per slate
        K actions per context
        noise = Add gaussian noise?
        seed = random seed
        pos = all vectors in positive orthant?
        """
        self.d = d
        self.L = L
        self.K = K
        self.N = None
        self.X = None
        self.noise = noise
        self.seed = seed
        self.pos = pos

        if seed is not None:
            np.random.seed(574)
        if self.pos:
            self.weights = np.matrix(np.random.dirichlet(np.ones(self.d))).T
            self.weights = self.weights/np.linalg.norm(self.weights)
        else:
            self.weights = np.matrix(np.random.normal(0, 1, [self.d,1]))
            self.weights = self.weights/np.sqrt(self.weights.T*self.weights)

        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.curr_x = None
        self.features = None
        self.all_features = []
        self.curr_r = None
        self.curr_x = None

    def get_new_context(self):
        ## Generate random feature matrix and normalize.
        if self.seed is not None:
            np.random.seed((self.t+17)*(self.seed+1) + 37)

        if self.pos:
            self.features = np.matrix(np.random.dirichlet(1.0/self.d*np.ones(self.d), self.K))
        else:
            self.features = np.matrix(np.random.normal(0, 1, [self.K, self.d]))

        self.features = np.diag(1./np.sqrt(np.diag(self.features*self.features.T)))*self.features
        self.all_features.append(self.features)

        self.curr_means = np.array((self.features*self.weights).T)[0]
        self.curr_offset = -1*np.max(self.curr_means)
        ## self.curr_offset = 0
        ## self.curr_offset = (self.features[0,:]*self.weights)**2
        if self.noise and type(self.noise) == float:
            self.noise_term = np.random.normal(0, self.noise)
            self.curr_r = np.array(self.curr_means+self.curr_offset+self.noise_term)
        elif self.noise:
            self.noise_term = np.random.normal(0, 0.1)
            self.curr_r = np.array(self.curr_means+self.curr_offset+self.noise_term)
        else:
            self.curr_r = np.array(self.curr_means+self.curr_offset)
        ## self.curr_r = np.array(self.curr_r.T)[0]
        old_t = self.t
        self.t += 1
        self.curr_x = SemiparametricBandit.LinearContext(self.t, self.features)
        return self.curr_x

    def get_best_reward(self):
        idx = np.argsort(self.curr_means)
        return np.sum(self.curr_r[idx[len(idx)-self.L:len(idx)]])

    def get_slate_reward(self, A):
        return self.curr_r[A]


class MSLRBandit(SemibanditSim):
    """
    Currently deprecated
    """
    def __init__(self, L, path_to_mslr="/home/akshay/Downloads/"):
        self.L = L
        self.curr_fold = 1
        self.path = path_to_mslr
        self.f = open(self.path+"Fold%d/train.txt" % (self.curr_fold), "r")
        self.contexts = MSLRBandit.ContextIterator(self.f)

        self.features = {}
        self.Ks = {}
        self.curr_x
        self.curr_r

    def get_new_context(self):
        context = self.contexts.next()
        if context == None:
            return context
        (query, features, relevances) = context
        self.features[query] = features
        self.Ks[query] = features.shape[0]
        self.curr_x = query
        self.curr_r = relevances
        self.curr_k = self.Ks[query]
        return query

class DatasetBandit(SemibanditSim):
    DATASETS = {
        'mq2008': ContextIterators.MQ2008ContextIterator,
        'mq2008val': ContextIterators.MQ2008ValContextIterator,
        'mq2007': ContextIterators.MQ2007ContextIterator,
        'mq2007val': ContextIterators.MQ2007ValContextIterator,
        'mslr': ContextIterators.MSLRContextIterator2,
        'mslrsmall': ContextIterators.MSLRSmall,
        'mslr30k': ContextIterators.MSLR30k,
        'yahoo': ContextIterators.YahooContextIterator,
        'xor': ContextIterators.XORContextIterator
        }

    def __init__(self, L=5, loop=False, dataset="xor", metric=Metrics.NDCG, structure="none", noise=0.1):
        self.L = L
        self.dataset = dataset
        self.contexts = DatasetBandit.DATASETS[dataset](L=self.L,loop=loop)
        self.K = self.contexts.K
        self.d = self.contexts.d
        self.has_ldf = self.contexts.has_ldf
        self.structure = structure
        self.noise_rate = noise
        self.gaussian = True
        self.seed = None

        if self.structure == "cluster":
            self.contexts.cluster_docs()

        self.curr_x = None
        self.curr_r = None
        if metric == Metrics.NDCG:
            assert 'get_all_relevances' in dir(self.contexts), "Cannot initialize metric"
            self.metric = metric(self.L, self.contexts.get_all_relevances(), False)
        elif metric == Metrics.NavigationalTTS:
            assert 'get_all_relevances' in dir(self.contexts), "Cannot initialize metric"
            self.metric = metric(self.L, 60, np.max(self.contexts.get_all_relevances())+1)
        else:
            self.metric = None

    def set_policies(self, policies):
        ## First make sure all policies respect the L requirement
        self.Pi = []
        for pi in policies:
            assert type(pi) == EnumerationPolicy, "cannot set to non EnumerationPolicy"
            actions = pi.actions
            new_actions = {}
            for (k,v) in actions.items():
                new_actions[k] = v[0:self.L]
            self.Pi.append(EnumerationPolicy(new_actions))

        ## Compute Pistar
        print("---- Evaluating All Policies ----")
        Pistar = None
        best_score = 0.0
        for p in range(len(self.Pi)):
            pi = self.Pi[p]
            score = 0.0
            for i in range(len(self.contexts.docsPerQuery)):
                curr_x = Context.Context(i, self.contexts.features[i,:,:], self.contexts.docsPerQuery[i], self.L)
                curr_r = self.contexts.relevances[i,:]
                A = pi.get_action(curr_x)
                if self.metric != None:
                    (val, clickeddocs, blah) = self.metric.computeMetric(curr_r[A], self.L, curr_x.get_name())
                else: 
                    val = np.sum(curr_r[A])
                score += val

            print("Policy %d Score %0.3f" % (p, score))
            if Pistar == None or score >= best_score:
                Pistar = p
                best_score = score
        print("---- Best Policy is %d ----" % (Pistar))
        self.Pistar = Pistar

    def set_best_policy(self, policy):
        self.Pi = [policy]
        self.Pistar = 0

    def set_seed(self, i):
        self.seed = i

    def get_new_context(self):
        tmp = self.contexts.next()
        if tmp == None:
            return tmp
        self.curr_x = tmp[0]
        if self.noise_rate == None:
            self.curr_r = self.transform_reward(tmp[1])
        else:
            if self.seed is not None:
                np.random.seed(self.curr_x.name+self.seed)
            if self.gaussian:
                self.curr_r = np.random.normal(self.transform_reward(tmp[1]), self.noise_rate)
            else:
                self.curr_r = np.random.binomial(1, self.transform_reward(tmp[1]))
        self.clickeddocs = None
        self.played = None
        return self.curr_x

    def get_slate_reward(self, A):
        if self.metric != None:
            self.played = A
            (val, clicked_docs, dwell_times) = self.metric.computeMetric(self.curr_r[A], self.L, self.curr_x.get_name())
            self.clicked_docs = clicked_docs
            self.dwell_times = dwell_times
            return val
        else:
            return np.sum(self.curr_r[A])

    def get_base_rewards(self, A):
        if self.metric != None:
            assert self.played is not None and (A == self.played).all(), "Cannot call get_base_rewards before get_slate_rewards"
            return self.clicked_docs ## *self.dwell_times
        else:
            return self.curr_r[A]

    def offline_evaluate(self, policy, T=None, train=True):
        score = 0.0
        context_iter = DatasetBandit.DATASETS[self.dataset](train=train,L=self.L)
        t = 0
        while True:
            if T is not None and t >= T:
                return score/T
            tmp = context_iter.next()
            if tmp == None:
                return score/t
            (x,reward) = tmp
            score += np.sum(self.transform_reward(reward[policy.get_action(x)]))
            t += 1
        
    def get_best_reward(self):
        if "Pi" in dir(self):
            if self.metric != None:
                (val, clickeddocs, t) = self.metric.computeMetric(self.curr_r[self.Pi[self.Pistar].get_action(self.curr_x)], self.L, self.curr_x.get_name())
                return val
            else:
                return np.sum(self.transform_reward(self.curr_r[self.Pi[self.Pistar].get_action(self.curr_x)]))
        else:
            idx = np.argsort(self.curr_r)[::-1][:self.L]
            if self.metric != None:
                val = self.metric.computeMetric(self.curr_r[idx], self.L, self.curr_x.get_name())[0]
                return val
            else:
                return np.sum(self.transform_reward(self.curr_r[idx]))

    def get_max_achievable(self, T=None):
        ctx_iter = DatasetBandit.DATASETS[self.dataset](L=self.L, train=False, loop=False)
        t = 0
        score = 0.0
        while True:
            if T is not None and t >= T:
                break
            t += 1
            tmp = ctx_iter.next()
            if tmp == None:
                break
            reward = tmp[1]
            idx = np.argsort(reward)
            score += np.sum(self.transform_reward(reward[idx[len(reward)-self.L:len(reward)]]))
        return score

    def get_best_policy(self, T=None, learning_alg=None, classification=True):
        if learning_alg == None:
            return self.Pi[self.Pistar]
        else:
            ## Prepare dataset
            dataset = []
            if T == None:
                ctx_iter = DatasetBandit.DATASETS[self.dataset](L=self.L, train=True, loop=False)
            else:
                ctx_iter = DatasetBandit.DATASETS[self.dataset](L=self.L, train=True, loop=True)
            t = 0
            while True:
                if T is not None and t >= T:
                    break
                t += 1
                tmp = ctx_iter.next()
                if tmp == None:
                    break
                dataset.append((tmp[0], self.transform_reward(tmp[1])))
            if classification:
                return Argmax.argmax(self, dataset, policy_type=ClassificationPolicy, learning_alg=learning_alg)
            else:
                return Argmax.argmax(self, dataset, policy_type=RegressionPolicy, learning_alg=learning_alg)


    def transform_reward(self, r):
        if self.noise_rate == None or self.gaussian:
            return r
        else:
            return np.minimum((1.0+r)*self.noise_rate, 1)
            

class SemiSynthBandit(OrderedSBSim):

    def __init__(self,L=5,loop=True,dataset="letter",N=100,metric="ndcg"):
        self.one_pass = not loop
        self.L = L
        self.dataset = dataset
        self.data = DatasetBandit.DATASETS[dataset](L=self.L)
        self.K = self.data.K
        self.d = self.data.d
        self.has_ldf = self.data.has_ldf
        self.X = self.data.X
        self.contexts = []
        for i in range(self.X):
            self.contexts.append(Context.Context(i, np.zeros((1,1)), self.K, self.L))
        self.r_feats = []
        for i in range(self.X):
            self.r_feats.append(self.data.get_r_feat(i))

        if metric == "ndcg":
            self.metric = Metrics.NDCG(self.L, self.data.relevances,False)
        else:
            self.metric = Metrics.SumRelevance(self.L)

        self.N = N
        tmp_policies = self.build_policies(N)
        piMat = np.zeros((N,self.X,self.L),dtype=np.int)
        for n in range(N):
            for x in range(self.X):
                act = tmp_policies[n].get_action(self.data.get_context(x))
                piMat[n,x,:] = act
        self.Pi = []
        for i in range(N):
            pi = EnumerationPolicy(dict(zip(range(self.X), [piMat[i,j,:] for j in range(self.X)])))
            self.Pi.append(pi)
        
        self.Pistar = self.get_best_policy()
        self.curr_idx = 0
        self.curr_x = None
        self.curr_r = None

    def build_policies(self,n):
        c2 = DatasetBandit.DATASETS[self.dataset](L=1,loop=True)
        Policies = []
        for p in range(n):
            X = np.zeros((100, c2.d))
            r = np.zeros((100,))
            for n in range(100):
                (curr_x, curr_r) = c2.next()
                a = np.random.choice(curr_x.get_K())
                X[n,:] = curr_x.get_ld_features()[a,:]
                r[n] = curr_r[a]
            tree = sklearn.tree.DecisionTreeRegressor(max_depth=3)
            tree.fit(X,r)
            Policies.append(RegressionPolicy(tree))
        return(Policies)

    def get_best_policy(self):
        scores = np.zeros(len(self.Pi))
        for i in range(self.X):
            self.curr_x = self.contexts[i]
            self.curr_r = self.r_feats[i]
            scores += np.array([self.get_slate_reward(pi.get_action(self.curr_x)) for pi in self.Pi])
        return np.argmax(scores)

    def get_new_context(self):
        if self.one_pass:
            self.curr_idx += 1
            if self.curr_idx >= self.X:
                return None
            self.curr_x = self.contexts[self.curr_idx]
        else:
            self.curr_x = self.contexts[np.random.randint(0, self.X)]
        self.curr_r = self.r_feats[self.curr_x.name]
        return self.curr_x

    def get_slate_reward(self, A):
        ## TODO: implement some metric here.
        return self.metric.computeMetric(self.curr_r[A],self.curr_x.name)[0]


