import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import argparse
import scipy
import scipy.stats

def plot_mse_stderr(D, x):
    ## x = np.arange(100,50*(len(D['CF_mse'][0])+2),50)

    trials = len(D['CF_mse'])
    mu = np.mean(np.log10(D['CF_mse']),axis=0)
    stdev = np.std(np.log10(D['CF_mse']),axis=0)
    l1 = plt.plot(x, mu)
    plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l1[0].get_color(), alpha=0.2)

    trials = len(D['IPS_mse'])
    mu = np.mean(np.log10(D['IPS_mse']),axis=0)
    stdev = np.std(np.log10(D['IPS_mse']),axis=0)
    l2 = plt.plot(x, mu)
    plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l2[0].get_color(), alpha=0.2)

    trials = len(D['Regress_mse'])
    mu = np.mean(np.log10(D['Regress_mse']),axis=0)
    stdev = np.std(np.log10(D['Regress_mse']),axis=0)
    l3 = plt.plot(x, mu)
    plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l3[0].get_color(), alpha=0.2)

    trials = len(D['SkyRegress_mse'])
    mu = np.mean(np.log10(D['SkyRegress_mse']),axis=0)
    stdev = np.std(np.log10(D['SkyRegress_mse']),axis=0)
    l3 = plt.plot(x, mu)
    plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l3[0].get_color(), alpha=0.2)

    trials = len(D['Semibandit_mse'])
    mu = np.mean(np.log10(D['Semibandit_mse']),axis=0)
    stdev = np.std(np.log10(D['Semibandit_mse']),axis=0)
    l3 = plt.plot(x, mu)
    plt.fill_between(x,
                     mu - 2/np.sqrt(trials)*stdev,
                     mu + 2/np.sqrt(trials)*stdev,
                     color=l3[0].get_color(), alpha=0.2)

    llist = ['CF', 'IPS', 'Regress', 'SkyRegress', 'Semibandit']

    llist.append('y=1/x')
    y = x[0]*np.mean(D['CF_mse'],axis=0)[0]/x
    l5 = plt.plot(x,np.log10(y))
    
    plt.legend(llist, loc='best')
    plt.xlabel('n')
    plt.ylabel('MSE')
    ax = plt.gca()
    ax.set_xscale("log")

    ticks = ax.get_yticks()
    ticks = ["$10^{%d}$" % t for t in ticks]
    ax.set_yticklabels(ticks,size=ax.get_xticklabels()[0].get_fontsize(), usetex=False)
    ticks = ax.get_xticks()
    ticks = ["$%d$" % t for t in ticks]
    ax.set_xticklabels(ticks,size=ax.get_yticklabels()[0].get_fontsize(), usetex=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store',
                        help='pkl file with data')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.set_defaults(save=False)
    parser.add_argument('--type', dest='type', action='store', choices=['mse_std'], default='mse_std')

    Args = parser.parse_args(sys.argv[1:])
    print(Args)
    name = Args.file.split("/")[-1]
    name = name.split(".")[0]
    print(name)

    D = pickle.load(open(Args.file,"rb"))

    if Args.type=='mse_std':
        plot_mse_stderr(D[0], D[1])

    if Args.save:
        plt.savefig("./figs/%s_%s.pdf" % (name, Args.type), format="pdf", dpi=1000)
    else:
        plt.show()
