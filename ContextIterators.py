import Context
import numpy as np
import string
import settings

"""
This is the main place where we parse different datasets.
For each dataset, implement a ContextIterator with a next method,
that reads the data file and returns a context and reward object.
"""
class MQ2008ContextIterator(object):
    def __init__(self, K=8, L=1, train=True, loop=False):
        self.d = 47

        self.loop = loop
        if train:
            npFile = np.load(settings.BASE_DIR+"data/MQ2008_train.npz")
        else:
            npFile = np.load(settings.BASE_DIR+"data/MQ2008_val.npz")            
        self.L = L
        self.K = K
        self.relevances = npFile['relevances']
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']
        self.X = len(self.docsPerQuery)

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int)

        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, len(self.docsPerQuery))
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        curr_x = Context.Context(self.curr_idx, self.features[self.curr_idx,:,:], self.docsPerQuery[self.curr_idx], self.L)
        curr_r = self.relevances[self.curr_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

class MQ2008ValContextIterator(object):
    def __init__(self, K=8, L=1,train=True, loop=False):
        ## Ignore train flag we don't have test data
        self.d = 47

        self.loop = loop
        npFile = np.load(settings.BASE_DIR+"data/MQ2008_val.npz")
        self.L = L
        self.K = K
        self.relevances = npFile['relevances']
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']
        self.X = len(self.docsPerQuery)

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int)

        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, len(self.docsPerQuery))
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        curr_x = Context.Context(self.curr_idx, self.features[self.curr_idx,:,:], self.docsPerQuery[self.curr_idx], self.L)
        curr_r = self.relevances[self.curr_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

class MQ2007ContextIterator(object):
    def __init__(self, K=40, L=1, train=True, loop=False):
        ## Ignore train flag we don't have test data
        self.d = 47

        self.loop = loop
        npFile = np.load(settings.BASE_DIR+"data/mq2007_train.npz")
        self.L = L
        self.K = K
        self.relevances = npFile['relevances']
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int)
        self.X = len(self.docsPerQuery)

        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, len(self.docsPerQuery))
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        if "clusters" in dir(self):
            curr_x = Context.Context(self.curr_idx, self.features[self.curr_idx,:,:], self.docsPerQuery[self.curr_idx], self.L, clusters=self.clusters[self.curr_idx])
        else:
            curr_x = Context.Context(self.curr_idx, self.features[self.curr_idx,:,:], self.docsPerQuery[self.curr_idx], self.L)
        curr_r = self.relevances[self.curr_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

    def cluster_docs(self):
        """
        For each query, cluster the documents into four groups of size 10.
        """
        self.clusters = []
        for q in range(len(self.docsPerQuery)):
            inds = self.relevances[q,:].argsort()
            clusters = [inds[0:40:4], inds[1:40:4], inds[2:40:4], inds[3:40:4]]
            self.clusters.append(clusters)

class MQ2007ValContextIterator(object):
    def __init__(self, K=40, L=1, train=True, loop=False):
        ## Ignore train flag we don't have test data
        self.d = 47

        self.loop = loop
        npFile = np.load(settings.BASE_DIR+"data/mq2007_val.npz")
        self.L = L
        self.K = K
        self.relevances = npFile['relevances']
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int)
        self.X = len(self.docsPerQuery)

        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, len(self.docsPerQuery))
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        if "clusters" in dir(self):
            curr_x = Context.Context(self.curr_idx, self.features[self.curr_idx,:,:], self.docsPerQuery[self.curr_idx], self.L, clusters=self.clusters[self.curr_idx])
        else:
            curr_x = Context.Context(self.curr_idx, self.features[self.curr_idx,:,:], self.docsPerQuery[self.curr_idx], self.L)
        curr_r = self.relevances[self.curr_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

    def cluster_docs(self):
        """
        For each query, cluster the documents into four groups of size 10.
        """
        self.clusters = []
        for q in range(len(self.docsPerQuery)):
            inds = self.relevances[q,:].argsort()
            clusters = [inds[0:40:4], inds[1:40:4], inds[2:40:4], inds[3:40:4]]
            self.clusters.append(clusters)

class YahooContextIterator(object):
    def __init__(self, K=6, L=2, train=True, loop=False):
        ## Ignore train flag we don't have test data
        self.d = 415

        self.loop = loop
        npFile = np.load(settings.DATA_DIR+"yahoo/yahoo_big.npz")
        self.L = L
        self.K = K
        self.relevances = npFile['relevances']
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']
        self.features = np.nan_to_num(self.features)

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int)
        self.X = len(self.docsPerQuery)

        self.order = np.random.permutation(self.X)
        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, len(self.docsPerQuery))
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        q_idx = self.order[self.curr_idx]
        curr_x = Context.Context(q_idx, self.features[q_idx,:,:], self.docsPerQuery[q_idx], self.L)
        curr_r = self.relevances[q_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

class MSLRContextIterator(object):
    def __init__(self,K=50,L=5,train=True, loop=False):
        self.d = 136
        self.curr_file = 1
        self.train = train
        if train:
            self.trainflag = "train"
        else:
            self.trainflag = "vali"
        self.fname = settings.DATA_DIR+"mslr/mslr_%s%d.txt" % (self.trainflag, self.curr_file)
        self.loop = loop
        self.f = open(self.fname)
        self.L = L
        self.K = K
        self.curr_line = None
        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        features = []
        relevances = []
        while self.curr_line == None:
            try:
                self.curr_line = self.f.__next__()
            except Exception:
                self.curr_file += 1
                self.curr_line = None
                if self.curr_file > 5:
                    return None
                self.fname = settings.DATA_DIR+"mslr/mslr_%s%d.txt" % (self.trainflag, self.curr_file)
                print("Opening new file: %s" % (self.fname))
                self.f = open(self.fname)
        data = self.curr_line.split(" ")
        curr_q = int(data[1].split(":")[-1])
        relevances.append(int(data[0]))
        features.append(np.array([float(x.split(":")[-1]) for x in data[2:-1]]))
        while True:
            try:
                self.curr_line = self.f.__next__()
            except Exception:
                if self.loop:
                    self.f = open(self.fname)
                else:
                    self.curr_file += 1
                    self.curr_line = None
                    if self.curr_file > 5:
                        return None
                    self.fname = settings.DATA_DIR+"mslr/mslr_%s%d.txt" % (self.trainflag, self.curr_file)
                    print("Opening new file: %s" % (self.fname))
                    self.f = open(self.fname)
                self.curr_line = self.f.__next__()
            data = self.curr_line.split(" ")
            if int(data[1].split(":")[-1]) != curr_q:
                ## We have moved onto the next query
                features = np.array(features, dtype=np.float32)
                if self.K == None:
                    toret = Context.Context(self.curr_idx, features, features.shape[0], self.L)
                    self.curr_idx += 1
                    return (toret, np.array(relevances,dtype=np.float32))
                else:
                    if features.shape[0] < self.K:
                        ## We want to ignore this context. Doesn't have enough actions
                        return self.next()
                    else:
                        toret = Context.Context(self.curr_idx, features[0:self.K,:], self.K, self.L)
                        self.curr_idx += 1
                        return (toret, np.array(relevances[0:self.K], dtype=np.float32))
            relevances.append(int(data[0]))
            features.append(np.array([float(x.split(":")[-1]) for x in data[2:-1]]))

class MSLRContextIterator2(object):
    def __init__(self,K=50,L=5,train=True,loop=False,feat='full'):
        self.d = 136
        self.K = K
        self.L = L
        self.feat = feat
        self.doc_features = [5*i-1 for i in range(1,26)]

        if train:
            self.fname = settings.DATA_DIR+"mslr/mslr_train.npz"
        else:
            self.fname = settings.DATA_DIR+"mslr/mslr_vali.npz"
        self.loop = loop
        self.f = np.load(self.fname)
        self.relevances = self.f['relevances']
        self.features = self.f['features']
        self.docsPerQuery = self.f['docsPerQuery']
        self.X = len(self.docsPerQuery)

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        if self.feat == 'full':
            self.features = self.features[toretain,:self.K,:]
        else:
            self.features = self.features[toretain,:self.K,self.doc_features]
            self.d = 25
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int32)
        self.X = len(self.docsPerQuery)

        self.order = np.random.permutation(self.X)
        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, self.X)
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        q_idx = self.order[self.curr_idx]
        curr_x = Context.Context(q_idx, self.features[q_idx,:,:], self.docsPerQuery[q_idx], self.L)
        curr_r = self.relevances[q_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

class MSLRSmall(object):
    def __init__(self,K=10,L=3,train=True,loop=False):
        self.d = 136
        self.K = K
        self.L = L
        self.doc_features = [5*i-1 for i in range(1,26)]
        self.d = len(self.doc_features)

        if train:
            self.fname = settings.DATA_DIR+"mslr/mslr_train.npz"
        else:
            self.fname = settings.DATA_DIR+"mslr/mslr_vali.npz"

        self.loop = loop
        self.f = np.load(self.fname)
        self.relevances = self.f['relevances']
        self.features = self.f['features']
        self.docsPerQuery = self.f['docsPerQuery']
        self.X = len(self.docsPerQuery)

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.features = self.features[:,:,self.doc_features]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int32)
        self.X = len(self.docsPerQuery)

        self.order = np.random.permutation(self.X)
        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, self.X)
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        q_idx = self.order[self.curr_idx]
        curr_x = Context.Context(q_idx, self.features[q_idx,:,:], self.docsPerQuery[q_idx], self.L)
        curr_r = self.relevances[q_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

class MSLR30k(object):
    def __init__(self,K=10,L=3,train=True,loop=False):
        self.d = 136
        self.K = K
        self.L = L
        self.doc_features = range(136)
        self.d = len(self.doc_features)

        if train:
            self.fname = settings.DATA_DIR+"mslr/mslr30k_train.npz"
        else:
            self.fname = settings.DATA_DIR+"mslr/mslr30k_vali.npz"

        self.loop = loop
        self.f = np.load(self.fname)
        self.relevances = self.f['relevances']
        self.features = self.f['features']
        self.docsPerQuery = self.f['docsPerQuery']
        self.X = len(self.docsPerQuery)

        toretain = np.where(self.docsPerQuery >= self.K)[0]
        self.relevances = self.relevances[toretain,:self.K]
        self.features = self.features[toretain,:self.K,:]
        self.features = self.features[:,:,self.doc_features]
        self.docsPerQuery = self.K*np.ones(len(toretain),dtype=np.int32)
        self.X = len(self.docsPerQuery)

        self.order = np.random.permutation(self.X)
        self.curr_idx = 0
        self.has_ldf = True

    def next(self):
        if self.loop:
            self.curr_idx = np.random.randint(0, self.X)
        if self.curr_idx >= len(self.docsPerQuery):
            return None
        q_idx = self.order[self.curr_idx]
        curr_x = Context.Context(q_idx, self.features[q_idx,:,:], self.docsPerQuery[q_idx], self.L)
        curr_r = self.relevances[q_idx,:]
        self.curr_idx += 1
        return (curr_x, curr_r)

    def get_all_relevances(self):
        return self.relevances

class XORContextIterator(object):
    def __init__(self, K=10, L=5, train=True, loop=False):
        self.d = 5
        self.loop=loop
        self.K = K
        self.L = L
        self.has_ldf = True
        self.curr_idx = 0

    def next(self):
        """
        Generate a new context with ld_features that are uniform 1/0
        and with reward that is x_1 XOR x_2
        """
        
        if self.loop == False and self.curr_idx > 200:
            return None
        features = np.matrix(np.zeros([self.K, self.d]))
        features = np.matrix(np.random.binomial(1, 0.5, [self.K, self.d]))
        features = 2*features-1

        curr_x = Context.Context(self.curr_idx, features, self.K, self.L)
        curr_r = np.array([features[i,0]*features[i,1] for i in range(self.K)])
        self.curr_idx += 1
        return (curr_x, curr_r)
    
    
