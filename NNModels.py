import torch
from torch.autograd import Variable
import torch.nn
import numpy as np
import torch.nn.functional as F

class LinearModel(torch.nn.Module):
    """
    Cost sensitive linear model with label dependent features and shared parameter. 
    Input: a [n, 1, d*K] tensor
    Output: [n, 1, K] tensor 
    """

    def __init__(self, d, K):
        super(LinearModel, self).__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=d, stride=d, bias=False)

    def forward(self,x):
        h = self.conv(x)
        z = h - torch.mean(h)
        return (z)

class TwoLayer(torch.nn.Module):
    """
    Cost sensitive two-layer NN with label dependent features and shared parameter. 
    Input: a [n, 1, d*K] tensor
    Output: [n, 1, K] tensor 
    """
    def __init__(self, d, K, hidden=2):
        super(TwoLayer, self).__init__()
        self.fc1 = torch.nn.Conv1d(1, hidden, kernel_size=d,stride=d, bias=False)
        self.fc2 = torch.nn.Conv1d(hidden, 1, kernel_size=1, stride=1, bias=False)

    def forward(self,x):
        h1 = F.relu(self.fc1(x))
        h = self.fc2(h1)
        z = h - torch.mean(h)
        return (z)

def model_to_action(model, x):
    X = np.zeros((1, 1, x.get_ld_dim()*x.get_K()))
    X[0,0,:] = x.get_ld_features().reshape(1,x.get_ld_dim()*x.get_K())
    X = Variable(torch.FloatTensor(X))
    output = model(X)
    # print(output.data)
    p = torch.clamp(1+output,0)/torch.sum(torch.clamp(1+output,0))
    return (np.array(p.data)[0,0])

def langevin_step(model, features, targets, lr, iters=512, noise=True, debug=False):
    X = Variable(torch.FloatTensor(features))
    Y = Variable(torch.FloatTensor(targets))
    for i in range(1,iters+1):
        model.zero_grad()
        output = model(X)
        loss = Y.dot(torch.clamp(1+output, 0.0))/Y.shape[0]
        if np.log2(i) == np.floor(np.log2(i)) and debug:
            print("Iteration %d, Loss %0.3f" % (i, loss.data[0]))
        loss.backward()
        for f in model.parameters():
            if noise:
                r = torch.FloatTensor(np.random.normal(0,1,np.shape(f.grad.data)))
            else:
                r = torch.zeros(f.grad.data.size())
            f.data.sub_(f.grad.data*lr/2 - np.sqrt(lr)*r)
            f.data = torch.clamp(f.data, -1, 1)

if __name__=='__main__':
    import Simulators
    d = 20
    K = 5
    S = Simulators.LinearBandit(d, 1, K, noise=True, pos=False, quad=False)
    
    n = 1
    
    X = np.zeros([n, 1, d*K])
    Y = np.zeros([n, 1, K])
    for i in range(n):
        x = S.get_new_context()
        X[i,0,:] = x.get_ld_features().reshape(1,d*K)
        r = np.array(S.get_all_rewards())
        Y[i,0,:] = np.array(S.get_all_rewards())

    model = LinearModel(d,K)
    print(model)
    print([param for param in model.parameters()])
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    langevin_step(model, X, Y, 0.1, iters=2048, noise=True)

    # model = TwoLayer(d,K)
    # print(model)
    # print([param for param in model.parameters()])

    # langevin_step(model, X, Y, 0.1, iters=2048, noise=True)

    # print([np.linalg.norm(param) for param in model.parameters()])

    print(model_to_action(model,S.get_new_context()))
