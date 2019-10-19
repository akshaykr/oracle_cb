import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib as mpl

eps = [float('%0.3f' % x) for x in np.logspace(-3,1,10)]

def cv(prefix, Args):
    best_lime_m = None
    best_lime_l = None
    best_lin = None
    best_oracle_m = None
    best_oracle_l = None

    for e in eps:
        x = []; 
        name = prefix+"linucb_minimonster_%0.5f_regrets.out" % (e)
        f = open(name).readlines()
        for i in range(len(f)):
            x.append([float(a) for a in f[i].split(" ")])
        score = np.mean(x,axis=0)[-1]
        print('linucb delta=%0.3f score=%0.2f' % (e, score))
        if best_lin is None or score < best_lin[1]:
            best_lin = [e, score]

        x = []; 
        name = prefix+"limecb_minimonster_%0.5f_regrets.out" % (e)
        try:
            f = open(name).readlines()
            for i in range(len(f)):
                x.append([float(z) for z in f[i].split(" ")])
            score = np.mean(x,axis=0)[-1]
            print('limecb_m eps=%0.3f score=%0.2f' % (e, score))
            if best_lime_m is None or score < best_lime_m[1]:
                best_lime_m = [e, score]
        except FileNotFoundError:
            pass

        x = []; 
        name = prefix+"limecb_linucb_%0.5f_regrets.out" % (e)
        try:
            f = open(name).readlines()
            for i in range(len(f)):
                x.append([float(z) for z in f[i].split(" ")])
            score = np.mean(x,axis=0)[-1]
            print('limecb_l eps=%0.3f score=%0.2f' % (e, score))
            if best_lime_l is None or score < best_lime_l[1]:
                best_lime_l = [e, score]
        except FileNotFoundError:
            pass

        x = []; 
        name = prefix+"oracle_minimonster_%0.5f_regrets.out" % (e)
        try:
            f = open(name).readlines()
            for i in range(len(f)):
                x.append([float(z) for z in f[i].split(" ")])
            score = np.mean(x,axis=0)[-1]
            print('oracle_m eps=%0.3f score=%0.2f' % (e, score))
            if best_oracle_m is None or score < best_oracle_m[1]:
                best_oracle_m = [e, score]
        except FileNotFoundError:
            pass

        x = []; 
        name = prefix+"oracle_linucb_%0.5f_regrets.out" % (e)
        try:
            f = open(name).readlines()
            for i in range(len(f)):
                x.append([float(z) for z in f[i].split(" ")])
            score = np.mean(x,axis=0)[-1]
            print('oracle_l eps=%0.3f score=%0.2f' % (e, score))
            if best_oracle_l is None or score < best_oracle_l[1]:
                best_oracle_l = [e, score]
        except FileNotFoundError:
            pass

    return (best_lime_m[0], best_lime_l[0], best_lin[0], best_oracle_m[0], best_oracle_l[0])

def cv_plot(prefix, best_lime_m, best_lime_l, best_lin, best_oracle_m, best_oracle_l, ax, Args, kind='regrets',ylabel=True):
    x = []; y = []; z = []; a = []; b = [];
    f = open(prefix+"limecb_minimonster_%0.5f_%s.out" % (best_lime_m, kind)).readlines()
    for i in range(len(f)):
        x.append([float(a) for a in f[i].split(" ")])
    f = open(prefix+"linucb_minimonster_%0.5f_%s.out" % (best_lin,kind)).readlines()
    for i in range(len(f)):
        y.append([float(a) for a in f[i].split(" ")])  
    f = open(prefix+"oracle_minimonster_%0.5f_%s.out" % (best_oracle_m,kind)).readlines()
    for i in range(len(f)):
        z.append([float(a) for a in f[i].split(" ")])  
#     f = open(prefix+"limecb_linucb_%0.5f_%s.out" % (best_lime_l, kind)).readlines()
#     for i in range(len(f)):
#         a.append([float(a) for a in f[i].split(" ")])
#     f = open(prefix+"oracle_linucb_%0.5f_%s.out" % (best_oracle_l,kind)).readlines()
#     for i in range(len(f)):
#         b.append([float(a) for a in f[i].split(" ")])  

    xcoords = range(1,Args.T+1, 10)
    print(np.shape(x))
    l1 = ax.plot(xcoords, np.mean(x,axis=0))
    ax.fill_between(xcoords, np.mean(x,axis=0) - 2/np.sqrt(np.shape(x)[0])*np.std(x,axis=0), np.mean(x,axis=0) + 2/np.sqrt(np.shape(x)[0])*np.std(x,axis=0),alpha=0.1)
    l2 = ax.plot(xcoords, np.mean(y,axis=0))
    ax.fill_between(xcoords, np.mean(y,axis=0) - 2/np.sqrt(np.shape(y)[0])*np.std(y,axis=0), np.mean(y,axis=0) + 2/np.sqrt(np.shape(y)[0])*np.std(y,axis=0),alpha=0.1)
    l3 = ax.plot(xcoords, np.mean(z,axis=0))
    ax.fill_between(xcoords, np.mean(z,axis=0) - 2/np.sqrt(np.shape(z)[0])*np.std(z,axis=0), np.mean(z,axis=0) + 2/np.sqrt(np.shape(z)[0])*np.std(z,axis=0),alpha=0.1)
#     l4 = ax.plot(xcoords, np.mean(a,axis=0))
#     ax.fill_between(xcoords, np.mean(a,axis=0) - 2/np.sqrt(np.shape(a)[0])*np.std(a,axis=0), np.mean(a,axis=0) + 2/np.sqrt(np.shape(a)[0])*np.std(a,axis=0),alpha=0.1)
#     l5 = ax.plot(xcoords, np.mean(b,axis=0))
#     ax.fill_between(xcoords, np.mean(b,axis=0) - 2/np.sqrt(np.shape(b)[0])*np.std(b,axis=0), np.mean(b,axis=0) + 2/np.sqrt(np.shape(b)[0])*np.std(b,axis=0),alpha=0.1)
    plt.sca(ax)
    plt.xlim([0,Args.T])
    plt.xticks([0, 1000, 2000, 3000, 4000], fontsize=16)
    plt.ylim([0,400])
    plt.yticks([0,100,200,300,400], fontsize=16)
    plt.xlabel('T', fontsize=16)
    if ylabel:
        plt.ylabel('Regret', fontsize=16)
    legendHandles = []
    legendHandles.append((matplotlib.patches.Patch(color=l1[0].get_color(), label="ModCB"), "ModCB"))
    legendHandles.append((matplotlib.patches.Patch(color=l2[0].get_color(), label="LinUCB"), "LinUCB"))
    legendHandles.append((matplotlib.patches.Patch(color=l2[0].get_color(), label="Oracle"), "Oracle"))
#     legendHandles.append((matplotlib.patches.Patch(color=l1[0].get_color(), label="LimeCB_L"), "LimeCB_L"))
#     legendHandles.append((matplotlib.patches.Patch(color=l2[0].get_color(), label="Oracle_L"), "Oracle_L"))
    # plt.legend(['Ours', 'LinUCB', 'ILTCB', 'EpsGreedy'])

    print ("ModCB_M %0.2f, LinUCB %0.2f, Oracle_M %0.2f" % (np.mean(x,axis=0)[-1], np.mean(y,axis=0)[-1], np.mean(z,axis=0)[-1]))
    return(legendHandles)

if __name__=='__main__':
    import sys, os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=1000, help='number of rounds', type=int)
    parser.add_argument('--d', action='store', default=500, type=int)
    parser.add_argument('--K', action='store', default=2, type=int)
    parser.add_argument('--s', action='store', default=4, type=int)
    parser.add_argument('--noise', action='store', default=0.1, type=float)
    parser.add_argument('--save', action='store', default=False, type=bool)
    parser.add_argument('--kind', action='store', default='regrets', type=str)
    parser.add_argument('--final', action='store', default=False, type=bool)

    Args = parser.parse_args(sys.argv[1:])
    eps = [float('%0.3f' % x) for x in np.logspace(-3,1,10)]

    fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0], mpl.rcParams['figure.figsize'][1]))
    ax = fig.add_subplot(111,frameon=False)
    ax1 = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.subplots_adjust(left=0.03, right=0.99,bottom=0.3)
    
    prefix = '../results/T=%d_d=%d_s=%d_K=%d_sig=1.0/' % (Args.T, Args.d, Args.s, Args.K)
    (best_lime_m, best_lime_l, best_lin, best_oracle_m, best_oracle_l) = cv(prefix, Args)
    (legendHandles) = cv_plot(prefix, best_lime_m, best_lime_l, best_lin, best_oracle_m, best_oracle_l, ax1, Args,kind=Args.kind)
    plt.title('d=%d, $d_{m*}$=%d, K=%d' % (Args.d, Args.s, Args.K), fontsize=16)

    leg = ax1.legend([x[1] for x in legendHandles], loc='upper left', shadow=False, frameon=True, fontsize=16)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    if Args.save:
        plt.savefig('../figs/model_selection_T=%d_d=%d_s=%d_K=%d.pdf' % (Args.T, Args.d, Args.s, Args.K), format='pdf', dpi=100, bbox_inches='tight')
    else:
        plt.show()
