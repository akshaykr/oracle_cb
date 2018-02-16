import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import argparse
import scipy
import scipy.stats

## plt.style.use('ggplot')

def plot_mse(D):
    x = np.arange(100,10*len(D['EpsOracle'][0]),10)
    l1 = plt.plot(x, np.median(D['GoldStandard_mse'],axis=0)[10:])
    plt.fill_between(x,
                     np.percentile(D['GoldStandard_mse'],25,axis=0)[10:],
                     np.percentile(D['GoldStandard_mse'],75,axis=0)[10:],
                     color=l1[0].get_color(), alpha=0.2)
    l2 = plt.plot(x, np.median(D['IPS_mse'],axis=0)[10:])
    plt.fill_between(x,
                     np.percentile(D['IPS_mse'],25,axis=0)[10:],
                     np.percentile(D['IPS_mse'],75,axis=0)[10:],
                     color=l2[0].get_color(), alpha=0.2)
    l3 = plt.plot(x, np.median(D['PRreg_mse'],axis=0)[10:])
    plt.fill_between(x,
                     np.percentile(D['PRreg_mse'],25,axis=0)[10:],
                     np.percentile(D['PRreg_mse'],75,axis=0)[10:],
                     color=l3[0].get_color(), alpha=0.2)

    llist = ['Gold', 'IPS', 'PR']
    if 'DirectTree_mse' in D.keys():
        x = np.arange(100, 10*len(D['DirectTree_mse'][0]),10)
        l4 = plt.plot(x, np.median(D['DirectTree_mse'],axis=0)[10:])
        plt.fill_between(x,
                         np.percentile(D['DirectTree_mse'],25,axis=0)[10:],
                         np.percentile(D['DirectTree_mse'],75,axis=0)[10:],
                         color=l4[0].get_color(), alpha=0.2)
        llist.append('DirectTree')
    if 'DirectLasso_mse' in D.keys():
        x = np.arange(10*len(D['DirectLasso_mse'][0]),2*10*len(D['DirectLasso_mse'][0]), 10)
        l4 = plt.plot(x, np.median(D['DirectLasso_mse'],axis=0))
        plt.fill_between(x,
                         np.percentile(D['DirectLasso_mse'],25,axis=0),
                         np.percentile(D['DirectLasso_mse'],75,axis=0),
                         color=l4[0].get_color(), alpha=0.2)
        llist.append('DirectLasso')
    
    llist.append('y=1/x')
    x = np.arange(100,10*len(D['GoldStandard_mse'][0]),10)
    l5 = plt.plot(x, 1.0/x)
    plt.legend(llist, loc='best')
    plt.xlabel('n')
    plt.ylabel('MSE')
    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")

def plot_mse_stderr(D):

    llist = []

    if 'EpsOracle' in D.keys() and len(D['EpsOracle']) > 0:
        x = np.arange(100,10*len(D['EpsOracle'][0]),10)
        trials = len(D['EpsOracle'])
        mu = np.mean(D['EpsOracle'],axis=0)[10:]/x
        stdev = np.std(D['EpsOracle'],axis=0)[10:]/x
        l1 = plt.plot(x, mu)
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color=l1[0].get_color(), alpha=0.2)
        llist.append('EpsOracle')

    if 'EELS' in D.keys() and len(D['EELS']) > 0:
        x = np.arange(100,10*len(D['EELS'][0]),10)
        trials = len(D['EELS'])
        mu = np.mean(D['EELS'],axis=0)[10:]/x
        stdev = np.std(D['EELS'],axis=0)[10:]/x
        l2 = plt.plot(x, mu)
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color=l2[0].get_color(), alpha=0.2)
        llist.append('EELS')

    if 'EELS2' in D.keys() and len(D['EELS2']) > 0:
        x = np.arange(100,10*len(D['EELS2'][0]),10)
        trials = len(D['EELS2'])
        mu = np.mean(D['EELS2'],axis=0)[10:]/x
        stdev = np.std(D['EELS2'],axis=0)[10:]/x
        l3 = plt.plot(x, mu)
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color=l3[0].get_color(), alpha=0.2)
        llist.append('EELS2')

    if 'DirectTree_mse' in D.keys():
        x = np.arange(100, 10*len(D['DirectTree_mse'][0]),10)
        trials = len(D['DirectTree_mse'])
        mu = np.mean(np.log10(D['DirectTree_mse']),axis=0)[10:]
        stdev = np.std(np.log10(D['DirectTree_mse']),axis=0)[10:]
        
        l4 = plt.plot(x, mu)
        plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l4[0].get_color(), alpha=0.2)
        llist.append('Direct-Tree')
    if 'DirectLasso_mse' in D.keys():
        x = np.arange(100, 10*len(D['DirectLasso_mse'][0]),10)
        trials = len(D['DirectLasso_mse'])
        mu = np.mean(np.log10(D['DirectLasso_mse']),axis=0)[10:]
        stdev = np.std(np.log10(D['DirectLasso_mse']),axis=0)[10:]
        
        l4 = plt.plot(x, mu)
        plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l4[0].get_color(), alpha=0.2)
        llist.append('Direct-Lasso')
    
    plt.legend(llist, loc='best')
    plt.xlabel('n')
    plt.ylabel('Average Reward')
    ax = plt.gca()

#     ticks = ax.get_yticks()
#     ticks = ["$10^{%d}$" % t for t in ticks]
#     ax.set_yticklabels(ticks,size=ax.get_xticklabels()[0].get_fontsize(), usetex=False)
#     ticks = ax.get_xticks()
#     ticks = ["$%d$" % t for t in ticks]
#     ax.set_xticklabels(ticks,size=ax.get_yticklabels()[0].get_fontsize(), usetex=False)


def plot_band(D):
    arr = [D['PRreg_mse'][i] - D['IPS_mse'][i] for i in range(len(D['PRreg_mse']))]
    x = np.arange(100,10*len(D['PRreg_mse'][0]),10)
    l1 = plt.plot(x, np.median(arr,axis=0)[10:])
    plt.fill_between(x, np.percentile(arr,25,axis=0)[10:], np.percentile(arr,75,axis=0)[10:],
                     color=l1[0].get_color(), alpha=0.2)
    
    plt.plot(x,[0 for i in x])
    plt.xlabel('n')
    plt.ylabel('Paired PR_mse - IPS_mse')

def plot_box(D):
    total = len(D['PRreg_mse'][0])
    locs = range(int(total/20), total, int(total/20))
    arr = [D['PRreg_mse'][i] - D['IPS_mse'][i] for i in range(len(D['PRreg_mse']))]
    d = []
    for s in locs:
        d.append([arr[i][s] for i in range(len(D['PRreg_mse']))])
    plt.boxplot(d)
    plt.plot(range(len(locs)+1), [0 for i in range(len(locs)+1)], color='green')
    plt.xticks(range(1, len(locs)+1), [10*i for i in locs])
    plt.xlabel('n')
    plt.ylabel('Paired PR_mse - IPS_mse')

def plot_test(D):
    total = len(D['PRreg_mse'])
    x = np.arange(100,10*len(D['PRreg_mse'][0]),10)
    ps = []
    for i in x:
        (W,p) = wilcoxon([D['IPS_mse'][j][i/10] for j in range(total)], [D['PRreg_mse'][j][i/10] for j in range(total)])
        ps.append(p)
    l1 = plt.plot(x, ps)
    plt.xlabel('n')
    plt.ylabel('p-value')
    plt.title('P-values for One-sided Wilcoxon Signed-Rank Test')


def wilcoxon(x,y):
    """
    One-sided wilcoxon sign-rank test. 
    p-value is small if \EE x >> \EE y.
    """
    d = np.array(x)-np.array(y)
    d = np.compress(np.not_equal(d,0), d, axis=-1)
    n = len(d)

    inds = np.argsort(np.abs(d))
    sign_diff = np.sign(d)
    W = np.sum([sign_diff[inds[i]]*(i+1) for i in range(len(inds))])

    mn = 0.0
    std = np.sqrt(n*(n+1.)*(2*n+1.)/6)
    z = W/std

    p = 1-scipy.stats.norm.cdf(z)
    return (W,p)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store',
                        help='pkl file with data')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.set_defaults(save=False)
    parser.add_argument('--type', dest='type', action='store', choices=['mse','mse_std','band','box','test'])

    Args = parser.parse_args(sys.argv[1:])
    print(Args)
    name = Args.file.split("/")[-1]
    name = name.split(".")[0]
    print(name)

    D = pickle.load(open(Args.file,"rb"))

    if Args.type=='box':
        plot_box(D)
    elif Args.type=='mse':
        plot_mse(D)
    elif Args.type=='mse_std':
        plot_mse_stderr(D)
    elif Args.type=='band':
        plot_band(D)
    else:
        plot_test(D)

    if Args.save:
        plt.savefig("../figs/%s_%s.png" % (name, Args.type), format="png", dpi=100)
    else:
        plt.show()
