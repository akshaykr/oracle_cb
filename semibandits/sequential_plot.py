import pickle
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib as mpl
import numpy as np
import sys, argparse
sys.path.append("../")
import Plotting

colors=[
    '#d7191c',
    '#fdae61',
    '#abd9e9',
    '#2c7bb6',
]

Names = {
    'mini_gb2': 'VC-GB2',
    'mini_gb5': 'VC-GB5',   
    'mini_lin': 'VC-Lin',
    'epsall_gb2': '$\epsilon$-GB2',
    'epsall_gb5': '$\epsilon$-GB5',
    'epsall_lin': '$\epsilon$-Lin',
    'lin': 'LinUCB'
}

Styles = {
    'mini_gb2': ['k', 'solid'],
    'mini_gb5': [colors[1], 'solid'],   
    'mini_lin': [colors[0], 'solid'],
    'epsall_gb2': ['k', 'dashed'],
    'epsall_gb5': [colors[1], 'dashed'],
    'epsall_lin': [colors[0], 'dashed'],
    'lin': [colors[3], 'solid']
    }

marker=10
band=False

parser = argparse.ArgumentParser()
parser.add_argument('--save', dest='save', action='store_true')
Args = parser.parse_args(sys.argv[1:])

D1 = Plotting.read_dir("../results/mslr30k_T=36000_L=3_e=0.1/")

fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0],mpl.rcParams['figure.figsize'][1]-1))
ax = fig.add_subplot(111)
plt.rc('font', size=18)
plt.rcParams['text.usetex'] = True
plt.rc('font', family='sans-serif')

ticks=ax.get_yticks()
print(ticks)
ax.set_ylim(2.15, 2.35)
print("Setting ylim to %0.2f, %0.2f" % (ticks[3], ticks[len(ticks)-2]))
ticks = ax.get_yticks()
print(ticks)
# ticks = ["", "", "2.2", "", "2.3", ""]
# ax.set_yticklabels(ticks,size=16)
ticks = ['', '', '10000', '', '20000', '', '30000']
ax.set_xlim(1000, 31000)
ax.set_xticklabels(ticks,size=16)
plt.ylabel('Average reward', fontsize=16)
plt.xlabel('Rounds (T)', fontsize=16)
# ax.tick_params(labelsize=16)

plt.gcf().subplots_adjust(bottom=0.25)

plt.savefig('../figs/mslr_blank.pdf', format='pdf')

keys = ['epsall_lin', 'epsall_gb5']
for k in keys:
    params = []
    mus = []
    for (k1,v1) in D1[0].items():
        if k1.find(k) == 0 and len(D1[0][k1]) != 0:
            x = np.arange(100, 10*len(D1[0][k1][0])+1, 100)
            mus.append(np.mean(D1[0][k1],axis=0)[9::10]/x)
        params.append(k1.split("_")[-1])
    if len(mus) == 0:
        continue
    A = np.vstack(mus)
    ids = np.argmax(A, axis=0)
    mu = np.array([A[ids[i], i] for i in range(len(ids))])

    if k == 'mini_gb5':
        mu = np.mean(D1[0]['mini_gb5_0.008'], axis=0)[9::10]/x
    l1 = ax.plot(x,mu,rasterized=True, linewidth=2.0, label=Names[k], color=Styles[k][0], linestyle=Styles[k][1])
    
plt.savefig('../figs/mslr_noninteractive.pdf', format='pdf')


keys = ['mini_lin', 'mini_gb5', 'lin']
for k in keys:
    params = []
    mus = []
    for (k1,v1) in D1[0].items():
        if k1.find(k) == 0 and len(D1[0][k1]) != 0:
            x = np.arange(100, 10*len(D1[0][k1][0])+1, 100)
            mus.append(np.mean(D1[0][k1],axis=0)[9::10]/x)
        params.append(k1.split("_")[-1])
    if len(mus) == 0:
        continue
    A = np.vstack(mus)
    ids = np.argmax(A, axis=0)
    mu = np.array([A[ids[i], i] for i in range(len(ids))])

    if k == 'mini_gb5':
        mu = np.mean(D1[0]['mini_gb5_0.008'], axis=0)[9::10]/x
        l1 = ax.plot(x,mu,rasterized=True, linewidth=5.0, label=Names[k], color=Styles[k][0], linestyle=Styles[k][1])
    else:
        l1 = ax.plot(x,mu,rasterized=True, linewidth=2.0, label=Names[k], color=Styles[k][0], linestyle=Styles[k][1])
    
plt.savefig('../figs/mslr_all.pdf', format='pdf')
