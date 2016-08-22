import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle, argparse, sys, os, re

"""
Library to plot results from the routines in Experiments.
"""
Names = {
    'random': 'Random',
    'best_tree': 'OffTree',
    'best_gb': 'OffGB',
    'best_forest': 'OffForest',
    'best_lin': 'OffLin',        
    'best_nn': 'OffNN',
    'best_gb5': 'OffGB5',
    'mini_tree': 'VCTree',
    'mini_forest': 'VCForest',
    'mini_gb_': 'VCGB',
    'mini_gb2': 'VCGB2',
    'mini_gb5': 'VCGB5',
    'mini_lin': 'VCLin',
    'eps_tree': 'EpsTree',
    'eps_forest': 'EpsForest',
    'eps_gb': 'EpsGB',
    'eps_lin': 'EpsLin',
    'epsall_gb_': '$\epsilon$-GB',
    'epsall_gb2': '$\epsilon$-GB2',
    'epsall_gb5': '$\epsilon$-GB5',
    'epsall_lin': '$\epsilon$-Lin',
    'epsall_tree': '$\epsilon$-Tree',
    'lin': 'LinUCB',
    'epsall_nn': 'EpsAllNN',
    'eps_nn': 'EpsNN',
    'mini_nn': 'VCNN'
}
def PlotLinearExperiment(out):
    ks = [k for k in out.keys()]
    ks.sort()
    ds = [d for d in out[ks[0]].keys()]
    ds.sort()
    T = len(out[ks[0]][ds[0]][0][0])
    f, axes = plt.subplots(len(ks), len(ds))
    for i in range(len(ks)):
        for j in range(len(ds)):
            data = out[ks[i]][ds[j]]
            axes[i,j].errorbar(range(T), np.mean([x[0] for x in data],axis=0), np.std([x[0] for x in data],axis=0))
            axes[i,j].errorbar(range(T), np.mean([x[1] for x in data],axis=0), np.std([x[1] for x in data],axis=0), color='green')
            axes[i,j].set_title('k = %d, d = %d' % (ks[i], ds[j]))

def plotSynthetic(f):
    data = pickle.load(open(f, "rb"))
    rewards = data['rewards']
    regrets = data['regrets']

    arr = f.split('_')
    L = int(arr[4][2:arr[4].index(".")])
    T = int(arr[1][2:])
    N = int(arr[2][2:])
    K = int(arr[3][2:])
    dataset_name = arr[0].split("/")[-1]

    print("Plotting results for %s with T=%d, N=%d, K=%d, L=%d" % (dataset_name, T, N, K, L))
    f = plt.figure()
    ls = []
    names = []
    maxes = []
    for (k,v) in rewards.items():
        maxes.append(np.max(np.mean(v,axis=0)))
        l = plt.plot(range(T), np.mean(v,axis=0), linewidth=2)[0]
        ls.append(l)
        plt.fill_between(range(T), np.mean(v,axis=0)-np.std(v,axis=0), np.mean(v,axis=0)+np.std(v,axis=0), color=l.get_color(), alpha=0.2)
        names.append(k)
    plt.legend(ls,names,loc="best")
    plt.xlabel('T')
    plt.ylabel('Cumulative Reward')
    plt.ylim([0, np.max(maxes)])
    plt.title('%s Reward' % (dataset_name))
    plt.savefig('../figs/%s_reward_T=%d_N=%d_K=%d_L=%d.pdf' % (dataset_name, T, N, K, L), dpi=1000, format='pdf')

    f = plt.figure()
    ls = []
    names = []
    maxes = []
    for (k,v) in regrets.items():
        maxes.append(np.max(np.mean(v,axis=0)))
        l = plt.plot(range(T), np.mean(v,axis=0), linewidth=2)[0]
        ls.append(l)
        plt.fill_between(range(T), np.mean(v,axis=0)-np.std(v,axis=0), np.mean(v,axis=0)+np.std(v,axis=0), color=l.get_color(), alpha=0.2)
        names.append(k)
    plt.legend(ls,names,loc="best")
    plt.xlabel('T')
    plt.ylabel('Cumulative Regret')
    plt.ylim([0, np.max(maxes)])
    plt.title('%s Regret' % (dataset_name))
    plt.savefig('../figs/%s_regret_T=%d_N=%d_K=%d_L=%d.pdf' % (dataset_name, T, N, K, L), dpi=1000, format='pdf')

def PlotMQExperiment(f):
    data = pickle.load(open(f, "rb"))
    ls = []
    names = []
    rewards = data['rewards']
    risks = data['risks']

    arr = f.split('_')
    L = int(arr[1][2:])
    T = int(arr[2][2:arr[2].index(".")])
    dataset_name = arr[0].split("/")[-1]

    print(dataset_name)
    print("L = %d T = %d" % (L, T))
    f = plt.figure()
    maxes = []
    for (k,v) in rewards.items():
        if k != "exp":
            maxes.append(np.max(np.mean(v,axis=0)))
            l = plt.plot(np.mean(v,axis=0),linewidth=2)[0]
            ls.append(l)
            plt.fill_between(range(len(np.mean(v,axis=0))), np.mean(v,axis=0)-np.std(v,axis=0), np.mean(v,axis=0)+np.std(v,axis=0), color=l.get_color(), alpha=0.2)
            names.append(k)
    plt.legend(ls,names,loc='best')
    plt.xlabel('T')
    plt.ylabel('Cumulative Reward')
    plt.ylim([0, np.max(maxes)])
    plt.title('%s Performance' % (dataset_name))
    plt.savefig('../figs/%s_reward_L=%d_T=%d.pdf' % (dataset_name, L,T), dpi=1000, format='pdf')

    f = plt.figure()
    risks = [(k,v) for (k,v) in risks.items()]
    ind = range(len(risks))
    plt.bar(ind, [np.mean(x[1]) for x in risks], width=0.35, color='r', yerr=[np.std(x[1]) for x in risks], capsize=10, error_kw=dict(elinewidth=10,capthick=5,ecolor='b'))
    plt.xlabel('Algorithms')
    plt.ylabel('Expected Reward')
    plt.xticks(ind, [x[0] for x in risks])
    plt.title('%s Offline Performance' % (dataset_name))

def plot_cumulative(D):
    x = np.arange(10, 10*len(D[0]['random'][0]), 10)
    llist = []
    legendHandles = []

    if 'random' in D[0].keys() and len(D[0]['random']) != 0:
        mu = np.mean(D[0]['random'],axis=0)[1:]
        stdev = np.std(D[0]['random'],axis=0)[1:]
        l1 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
        llist.append('random')
        legendHandles.append(matplotlib.patches.Patch(color=l1[0].get_color(), label='random'))
        trials = len(D[0]['random'])
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color = l1[0].get_color(), alpha=0.2, rasterized = True)

    if 'mini_tree_reg' in D[0].keys() and len(D[0]['mini_tree_reg']) != 0:
        mu = np.mean(D[0]['mini_tree_reg'],axis=0)[1:]
        stdev = np.std(D[0]['mini_tree_reg'],axis=0)[1:]
        l2 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
        llist.append('mini_tree')
        legendHandles.append(matplotlib.patches.Patch(color=l2[0].get_color(), label='VCTree'))
        trials = len(D[0]['mini_tree_reg'])
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color = l2[0].get_color(), alpha=0.2, rasterized = True)

    elif 'mini_reg' in D[0].keys() and len(D[0]['mini_reg']) != 0:
        mu = np.mean(D[0]['mini_reg'],axis=0)[1:]
        stdev = np.std(D[0]['mini_reg'],axis=0)[1:]
        l2 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
        llist.append('mini_tree')
        legendHandles.append(matplotlib.patches.Patch(color=l2[0].get_color(), label='VCTree'))
        trials = len(D[0]['mini_reg'])
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color = l2[0].get_color(), alpha=0.2, rasterized = True)

    if 'mini_lin_reg' in D[0].keys() and len(D[0]['mini_lin_reg']) != 0:
        mu = np.mean(D[0]['mini_lin_reg'],axis=0)[1:]
        stdev = np.std(D[0]['mini_lin_reg'],axis=0)[1:]
        l2 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
        llist.append('mini_lin')
        legendHandles.append(matplotlib.patches.Patch(color=l2[0].get_color(), label='VCLin'))
        trials = len(D[0]['mini_lin_reg'])
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color = l2[0].get_color(), alpha=0.2, rasterized = True)

    if 'eps_reg' in D[0].keys() and len(D[0]['eps_reg']) != 0:
        mu = np.mean(D[0]['eps_reg'],axis=0)[1:]
        stdev = np.std(D[0]['eps_reg'],axis=0)[1:]
        l3 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
        llist.append('Eps')
        legendHandles.append(matplotlib.patches.Patch(color=l3[0].get_color(), label='Eps'))
        trials = len(D[0]['eps_reg'])
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color = l3[0].get_color(), alpha=0.2, rasterized = True)

    if 'lin' in D[0].keys() and len(D[0]['lin']) != 0:
        mu = np.mean(D[0]['lin'],axis=0)[1:]
        stdev = np.std(D[0]['lin'],axis=0)[1:]
        l4 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
        llist.append('LinUCB')
        legendHandles.append(matplotlib.patches.Patch(color=l4[0].get_color(), label='LinUCB'))
        trials = len(D[0]['lin'])
        plt.fill_between(x,
                         mu - 2/np.sqrt(trials)*stdev,
                         mu + 2/np.sqrt(trials)*stdev,
                         color = l4[0].get_color(), alpha=0.2, rasterized = True)

    leg = plt.legend(handles = legendHandles, loc='best', linewidth=5.0)
    plt.rc('font', size=15)
    plt.xlabel('Number of interactions (T)')
    plt.ylabel('Cumulative reward')


def plot_average(D, std=True, sliding=None, dataset='mslr30k'):
    llist = []
    lines = []
    legendHandles = []
    to_plot = ['mini_gb2', 'lin', 'epsall_gb2', 'mini_lin', 'epsall_lin', 'mini_tree', 'epsall_tree', 'eps_tree', 'mini_gb5', 'epsall_gb5']
    for k in to_plot:
        print("key: %s" % (k))
        params = []
        mus = []
        stds = []
        for (k1,v1) in D[0].items():
            if k1.find(k) == 0 and len(D[0][k1]) != 0:
                if sliding is not None:
                    x = np.arange(10, 10*len(D[0][k1][0]), 10)
                    tmp_data = []
                    for i in range(len(D[0][k1])):
                        y = np.diff(D[0][k1][i])/10
                        y = pd.rolling_mean(y, window=sliding)
                        tmp_data.append(y)
                    mus.append(np.mean(tmp_data, axis=0))
                    stds.append(2/np.sqrt(len(D[0][k1]))*np.std(tmp_data, axis=0))
                else:
                    x = np.arange(100, 10*len(D[0][k1][0])+1, 100)
                    mus.append(np.mean(D[0][k1],axis=0)[9::10]/x)
                    stds.append(2/np.sqrt(len(D[0][k1]))*(np.std(D[0][k1],axis=0)[9::10]/x))
                params.append(k1.split("_")[-1])
        if len(mus) == 0:
            continue
        A = np.vstack(mus)
        B = np.vstack(stds)
        ids = np.argmax(A, axis=0)
        print("Final best: %s" % (params[ids[-1]]))
        # print([params[x] for x in ids])
        mu = np.array([A[ids[i], i] for i in range(len(ids))])
        stdev = np.array([B[ids[i], i] for i in range(len(ids))])

        # if k.find('mini_gb5') == 0:
        #     cv = '0.022'
        #     key = k+'_'+cv
        #     x = np.arange(100, 10*len(D[0][key][0])+1, 100)
        #     mu = np.mean(D[0][key], axis=0)[9::10]/x
        #     stdev = 2/np.sqrt(len(D[0][key]))*(np.std(D[0][key],axis=0)[9::10]/x)
        l1 = plt.plot(x, mu, rasterized=True, linewidth=2.0, label=Names[k])
        lines.append(l1[0])
        llist.append(k)
        legendHandles.append((matplotlib.patches.Patch(color=l1[0].get_color(), label=Names[k]), Names[k]))
        if std:
            plt.fill_between(x,
                             mu - stdev,
                             mu + stdev,
                             color = l1[0].get_color(), alpha=0.2, rasterized = True)
    plt.legend([x[0] for x in legendHandles], [x[1] for x in legendHandles], loc='best')
    plt.rc('font', size=15)
    plt.rcParams['text.usetex'] = True
    plt.rc('font', family='serif')
    ax = plt.gca()
    ticks = ax.get_yticks()
    print(ticks)
    if dataset=="mslr30k":
        plt.ylim([2.0, 2.35])
        plt.title("Dataset: MSLR")
        ticks = ["", "1.8", "", "2.0", "", "2.2", "", "2.4", ""]
    if dataset=='yahoo':
        plt.ylim([2.8,3.15])
        plt.title("Dataset: Yahoo!")
        ticks = ["", "2.8", "", "2.9", "", "3.0", "", "3.1", ""]
    ax.set_yticklabels(ticks, size=15)
    ticks = ax.get_xticks()
    ticks = ["0", "", "10000", "", "20000", "", "30000", ""]
    ax.set_xticklabels(ticks, size=15)
    plt.xlabel('Number of interactions (T)')
    plt.ylabel('Average reward')


def plot_validation(D,cv=False):
    llist = []
    legendHandles = []

    to_plot = ['epsall_gb5', 'mini_gb5', 'lin', 'best_lin', 'best_gb5']
    for k in to_plot:
        if k == 'random' or k[0:4] == 'best':
            x = np.arange(500, 500*len(D[2]['lin_0.001'][0])+1,500)
            mu = np.mean(D[2][k])
            stdev = np.std(D[2][k])
            l1 = plt.plot(x, np.ones(len(x))*mu, rasterized=True, linewidth=2.0)
            llist.append(Names[k])
            legendHandles.append((matplotlib.patches.Patch(color=l1[0].get_color(), label=Names[k]), Names[k]))
            trials = len(D[0][k])
            plt.fill_between(x,
                             np.ones(len(x))*(mu - 2/np.sqrt(trials)*stdev),
                             np.ones(len(x))*(mu + 2/np.sqrt(trials)*stdev),
                             color = l1[0].get_color(), alpha=0.2, rasterized = True)
        else:
            params = []
            cv_mus = []
            mus = []
            stds = []
            for (k1,v1) in D[2].items():
                if k1.find(k) == 0 and len(D[2][k1]) != 0:
                    x = np.arange(500, 500*len(D[2][k1][0])+1, 500)
                    data = [np.cumsum(lst) for lst in D[2][k1]]
                    mus.append(500*np.mean(data,axis=0)/x)
                    cv_mus.append(np.mean(D[0][k1],axis=0)[9:-1:10])
                    stds.append(2/np.sqrt(len(D[2][k1]))*(500*np.std(data,axis=0)/x))
                    params.append(k1.split("_")[-1])
            if len(mus) == 0:
                continue
            A = np.vstack(mus)
            B = np.vstack(stds)
            C = np.vstack(cv_mus)
            if cv:
                ids = np.argmax(C,axis=0)
            else:
                ids = np.argmax(A, axis=0)
            mu = np.array([A[ids[i], i] for i in range(len(ids))])
            print(k)
            print("Final best: %s" % (params[ids[-1]]))
            print([params[x] for x in ids])
            stdev = np.array([B[ids[i], i] for i in range(len(ids))])
            l2 = plt.plot(x, mu, rasterized=True, linewidth=2.0)
            llist.append(k)
            legendHandles.append((matplotlib.patches.Patch(color=l2[0].get_color(), label=Names[k]), Names[k]))
            trials = len(D[2][k])
            plt.fill_between(x,
                             mu - stdev,
                             mu + stdev,
                             color = l2[0].get_color(), alpha=0.2, rasterized = True)
            
    plt.legend([x[0] for x in legendHandles], [x[1] for x in legendHandles], loc='best')
    plt.rc('font', size=15)
    plt.xlabel('Number of interactions (T)')
    plt.ylabel('Average validation reward')

def read_dir(f):
    D = {}
    D2 = {}
    for k in Names.keys():
        D[k] = []
        D2[k] = []

    files = os.listdir(f)
    for x in files:
        t1 = x.split("_")[0]
        t2 = x.split("_")[1]
        t3 = x.split("_")[2]
        z = x.split("_")[-2]
        data = np.loadtxt(f+x)
        t4 = x.split("_")[-1]
        if z=='rewards':
            if t1 == 'lin' and t2 == z:
                D['lin'].append(data)
            elif t1 == 'lin' and t2 != z:
                if 'lin_%0.3f' % (float(t2)) not in D.keys():
                    D['lin_%0.3f' % (float(t2))] = []
                if 'lin_%0.5f' % (float(t2)) not in D.keys():
                    D['lin_%0.5f' % (float(t2))] = []
                if '%0.3f' % (float(t2)) == "0.000":
                    D['lin_%0.5f' % (float(t2))].append(data)
                else:
                    D['lin_%0.3f' % (float(t2))].append(data)
            elif t1 == 'random' and 'random' in D.keys():
                D['random'].append(data)
            elif t1 == 'random':
                continue
            elif t3 == z and '%s_%s' % (t1,t2) in D.keys():
                D['%s_%s' % (t1, t2)].append(data)
            elif t3 != z:
                if '%s_%s_%0.3f' % (t1, t2, float(t3)) not in D.keys():
                    D['%s_%s_%0.3f' % (t1, t2, float(t3))] = []
                D['%s_%s_%0.3f' % (t1, t2, float(t3))].append(data)
        elif z=='validation':
            if t1 == 'lin' and t2 == z:
                D2['lin'].append(data)
            elif t1 == 'lin' and t2 != z:
                if 'lin_%0.3f' % (float(t2)) not in D2.keys():
                    D2['lin_%0.3f' % (float(t2))] = []
                if 'lin_%0.5f' % (float(t2)) not in D2.keys():
                    D2['lin_%0.5f' % (float(t2))] = []
                if '%0.3f' % (float(t2)) == "0.000":
                    D2['lin_%0.5f' % (float(t2))].append(data)
                else:
                    D2['lin_%0.3f' % (float(t2))].append(data)
            elif t1 == 'random' and 'random' in D2.keys():
                D2['random'].append(data)
            elif t1 == 'random':
                continue
            elif t3 == z and  '%s_%s' % (t1,t2) in D2.keys():
                D2['%s_%s' % (t1, t2)].append(data)
            elif t3 != z:
                if '%s_%s_%0.3f' % (t1, t2, float(t3)) not in D2.keys():
                    D2['%s_%s_%0.3f' % (t1, t2, float(t3))] = []
                D2['%s_%s_%0.3f' % (t1, t2, float(t3))].append(data)
    return (D, None, D2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--type', dest='type', action='store', choices=['cumulative', 'average', 'val', 'both'])
    
    Args = parser.parse_args(sys.argv[1:])

    print(Args)

    if Args.file[-1] == "/":
        name = Args.file.split("/")[-2].split("_")[0]
        L = int(Args.file.split("/")[-2].split("_")[-2].split("=")[-1])
        D = read_dir(Args.file)
    else:
        name = Args.file.split("_")[1]
        L = int(Args.file.split("_")[-3].split("=")[-1])
        D = pickle.load(open(Args.file, "rb"))

    print (name)
    if Args.type == 'cumulative':
        plot_cumulative(D)
    elif Args.type == 'average':
        plot_average(D, dataset=name)
    elif Args.type == 'val':
        plot_validation(D)
    elif Args.type == 'both':
        plt.figure(1)
        plot_validation(D)
        plt.figure(2)
        plot_average(D)
                        
    if Args.save:
        plt.savefig("./figs/%s_%s_L=%d.png" % (name, Args.type, L), format="png", dpi=100)
        plt.savefig("./figs/%s_%s_L=%d.pdf" % (name, Args.type, L), format="pdf", dpi=100)
    else:
        plt.show()
