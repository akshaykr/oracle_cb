import numpy as np
from Semibandits import Semibandit
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn.functional import softmax, mse_loss

class LinearNetwork(nn.Module):
    def __init__(self, input_size):
        super(LinearNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size,1)
        )
            
    def forward(self, feat):
        x = self.network(feat)
        return x

class PolicyGradient(Semibandit):

    def __init__(self,B):
        self.B = B
        self.name='pg'


    def init(self, T, params={}):
        """
        Initialize the policy parameters.
        Three parameters here:
        1. epsilon: parameter for uniform exploration, default = 0
        2. delta: softmax parameter, default =1 
        3. lr: learning rate
        """
        
        self.T = T
        self.d = self.B.d

        self.t = 1

        if "delta" in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 1.0

        if "eps" in params.keys():
            self.eps = params['eps']
        else:
            self.eps = 0

        if "lr" in params.keys():
            self.lr = params['lr']
        else:
            self.lr = 0.01

        self.reward = []
        self.opt_reward = []

        self.model = LinearNetwork(self.d)
        self.optim = optim.Adam(self.model.parameters(),lr=0.0001)


    def get_action(self, x):
        features = np.matrix(x.get_ld_features())
        # print(features, flush=True)
        logits = self.model(torch.Tensor(features)).detach()/self.delta
        logits = logits.T[0,:]
        # print(logits)
        outputs = softmax(logits, dim=0).numpy()
        # print(outputs, flush=True)
        proba = (1-self.eps)*outputs + self.eps*np.ones((x.get_K(),))/x.get_K()
        # print(proba)
        a = torch.multinomial(torch.Tensor(proba), x.get_L(), replacement=False).numpy()
        # print(a)
        self.imp_weight = proba[a]
        # print(self.imp_weight)
        self.action = a
        return a

    def update(self, x, A, y_vec, r):
        self.optim.zero_grad()
        features = np.matrix(x.get_ld_features())
        logits = self.model(torch.Tensor(features))/self.delta
        logits = logits.T[0,:]
        outputs = softmax(logits,dim=0)
        iw_rew = np.zeros((x.get_K(),))
        iw_rew[A] = y_vec/self.imp_weight
        # print(outputs)
        # print(iw_rew)
        loss = -1*torch.dot(torch.Tensor(iw_rew), outputs)
        loss.backward()
        self.optim.step()

class OnlineEpsGreedy(Semibandit):

    def __init__(self,B):
        self.B = B
        self.name='eps'
    
    def init(self, T, params={}):
        """
        Initialize the policy parameters.
        Three parameters here:
        1. epsilon: parameter for uniform exploration, default = 0
        2. delta: softmax parameter, default =1 
        3. lr: learning rate
        """
        
        self.T = T
        self.d = self.B.d

        self.t = 1

        if "delta" in params.keys():
            self.eps = params['delta']
        else:
            self.eps = 0.0

        if "lr" in params.keys():
            self.lr = params['lr']
        else:
            self.lr = 0.01

        self.reward = []
        self.opt_reward = []

        self.model = LinearNetwork(self.d)
        self.optim = optim.Adam(self.model.parameters(),lr=0.0001)

    def get_action(self, x):
        features = np.matrix(x.get_ld_features())
        # print(features, flush=True)
        logits = self.model(torch.Tensor(features)).detach()
        logits = logits.T[0,:]
        greedy_acts = np.argsort(logits)[-x.get_L():]
        proba = self.eps*np.ones((x.get_K(),))/x.get_K()
        proba[greedy_acts] += (1-self.eps)
        # print(proba)
        a = torch.multinomial(torch.Tensor(proba), x.get_L(), replacement=False).numpy()
        # print(a)
        self.imp_weight = proba[a]
        # print(self.imp_weight)
        self.action = a
        return a

    def update(self, x, A, y_vec, r):
        self.optim.zero_grad()
        features = np.matrix(x.get_ld_features())
        outputs = self.model(torch.Tensor(features))
        outputs = outputs.T[0,:]
        iw_rew = np.zeros((x.get_K(),))
        iw_rew[A] = y_vec/self.imp_weight
        # print(outputs)
        # print(iw_rew)
        loss = mse_loss(torch.Tensor(iw_rew), outputs)
        loss.backward()
        self.optim.step()


if __name__=='__main__':
    import sys, os, argparse, time
    import Simulators, Semibandits

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=50000, help='number of rounds', type=int)
    parser.add_argument('--iters', action='store', default=1, type=int)
    parser.add_argument('--d', action='store', default=20, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)

    parser.add_argument('--alg', action='store', default='all', choices=['pg','linucb','egreedy'])
    parser.add_argument('--param', action='store', default=None)
    parser.add_argument('--lr', action='store', type=float, default=0.1)
    parser.add_argument('--eps', action='store', type=float, default=0)
    parser.add_argument('--noise', action='store', default=None)
    parser.add_argument('--outdir_pref', action='store', type=str, default='./')

    Args = parser.parse_args(sys.argv[1:])
    print(Args,flush=True)
    if Args.noise is not None:
        Args.noise = float(Args.noise)

    outdir = '%spg_resultsT=%d_d=%d_K=%d/' % (Args.outdir_pref,Args.T, Args.d, Args.K)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if Args.param is not None:
        Args.param = float(Args.param)

    rewards = []
    regrets = []
    times = []
    for i in range(Args.iters):
        S = Simulators.LinearBandit(Args.d, 1, Args.K, noise=Args.noise, seed=i)
        if Args.alg == 'linucb':
            Alg = Semibandits.LinUCB(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'schedule': 1})
                stop = time.time()
        if Args.alg == 'pg':
            Alg = PolicyGradient(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'eps': Args.eps, 'lr': Args.lr})
                stop = time.time()
        if Args.alg == 'egreedy':
            Alg = OnlineEpsGreedy(S)
            if Args.param is not None:
                start = time.time()
                (r,reg,val_tmp) = Alg.play(Args.T, verbose=True, params={'delta': Args.param, 'lr': Args.lr})
                stop = time.time()
        times.append(stop-start)
        rewards.append(r)
        regrets.append(reg)
