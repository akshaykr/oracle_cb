import numpy as np

"""
Interface for accessing contexts, because different datasets have different looking formats. 
This handles both label-dependent features and non-label dependent features. 
"""
class Context(object):
    def __init__(self, name, features, K, L, clusters=None):
        self.name = name
        self.K = K
        self.L = L
        if clusters != None:
            self.clusters = clusters
        if features.shape[0] > 1:
            assert features.shape[0] == K, "Multi-row feature vec but not K rows"
            self.ld_features = features
            self.ld_dim = features.shape[1]
            self.features = np.reshape(features, self.K*self.ld_dim)
            self.dim = len(self.features)
        else:
            self.ld_dim = None
            self.ld_features = None
            self.features = features
            self.dim = self.features.shape[1]

    def get_ld_features(self):
        assert self.ld_features is not None, "Dataset does not support label dependent features"
        return self.ld_features

    def get_features(self):
        assert self.L == 1, "Cannot implement reduction to classification for semibandits"
        return self.features

    def get_L(self):
        return self.L

    def get_K(self):
        return self.K

    def get_name(self):
        return self.name

    def get_ld_dim(self):
        return self.ld_dim

    def get_dim(self):
        return self.dim
