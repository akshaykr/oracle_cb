import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib as mpl

def cv(prefix, Args):
    best_bose = None
    best_lin = None
    best_thompson = None
    best_mini = None
    best_eps = None

    for delta in deltas:
        x = []; y = []; z = [];
        name = prefix+"bose_%0.5f_regrets.out" % (delta)
        f = open(name).readlines()
        for i in range(len(f)):
            x.append([float(a) for a in f[i].split(" ")])
        name = prefix+"linucb_%0.5f_regrets.out" % (delta)
        f = open(prefix+"linucb_%0.5f_regrets.out" % (delta)).readlines()
        for i in range(len(f)):
            y.append([float(a) for a in f[i].split(" ")])
        name = prefix+"thompson_%0.5f_regrets.out" % (delta)
        f = open(prefix+"thompson_%0.5f_regrets.out" % (delta)).readlines()
        for i in range(len(f)):
            z.append([float(a) for a in f[i].split(" ")])
        score = np.mean(x,axis=0)[-1]
        print('bose delta=%0.3f score=%0.2f' % (delta, score))
        if best_bose is None or score < best_bose[1]:
            best_bose = [delta, score]
        score = np.mean(y,axis=0)[-1]
        print('linucb delta=%0.3f score=%0.2f' % (delta, score))
        if best_lin is None or score < best_lin[1]:
            best_lin = [delta, score]
        score = np.mean(z,axis=0)[-1]
        print('thompson delta=%0.3f score=%0.2f' % (delta, score))
        if best_thompson is None or score < best_thompson[1]:
            best_thompson = [delta, score]

    for e in eps:
        x = []; y = [];
        name = prefix+"minimonster_%0.5f_regrets.out" % (e)
        try:
            f = open(name).readlines()
            for i in range(len(f)):
                x.append([float(z) for z in f[i].split(" ")])
            score = np.mean(x,axis=0)[-1]
            print('minimonster eps=%0.3f score=%0.2f' % (e, score))
            if best_mini is None or score < best_mini[1]:
                best_mini = [e, score]
        except FileNotFoundError:
            pass
        name = prefix+"epsgreedy_%0.5f_regrets.out" % (e)
        f = open(name).readlines()
        for i in range(len(f)):
            y.append([float(z) for z in f[i].split(" ")])
        score = np.mean(y,axis=0)[-1]
        print('epsgreedy eps=%0.3f score=%0.2f' % (e, score))
        if best_eps is None or score < best_eps[1]:
            best_eps = [e, score]

    return (best_bose[0], best_lin[0], best_thompson[0], best_mini[0], best_eps[0])

def cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax, Args, kind='regrets',ylabel=True):
    x = []; y = []; z = []; m = []; e = []
    f = open(prefix+"bose_%0.5f_%s.out" % (best_bose, kind)).readlines()
    for i in range(len(f)):
        x.append([float(a) for a in f[i].split(" ")])
    f = open(prefix+"linucb_%0.5f_%s.out" % (best_lin,kind)).readlines()
    for i in range(len(f)):
        y.append([float(a) for a in f[i].split(" ")])  
    f = open(prefix+"thompson_%0.5f_%s.out" % (best_lin,kind)).readlines()
    for i in range(len(f)):
        z.append([float(a) for a in f[i].split(" ")])  
    f = open(prefix+"minimonster_%0.5f_%s.out" % (best_mini,kind)).readlines()
    for i in range(len(f)):
        m.append([float(a) for a in f[i].split(" ")])
    f = open(prefix+"epsgreedy_%0.5f_%s.out" % (best_eps,kind)).readlines()
    for i in range(len(f)):
        e.append([float(a) for a in f[i].split(" ")])  
    xcoords = range(1,Args.T+1, 10)
    print(np.shape(x))
    l1 = ax.plot(xcoords, np.mean(x,axis=0))
    ax.fill_between(xcoords, np.mean(x,axis=0) - 2/np.sqrt(np.shape(x)[0])*np.std(x,axis=0), np.mean(x,axis=0) + 2/np.sqrt(np.shape(x)[0])*np.std(x,axis=0),alpha=0.1)
    l2 = ax.plot(xcoords, np.mean(y,axis=0))
    ax.fill_between(xcoords, np.mean(y,axis=0) - 2/np.sqrt(np.shape(y)[0])*np.std(y,axis=0), np.mean(y,axis=0) + 2/np.sqrt(np.shape(y)[0])*np.std(y,axis=0),alpha=0.1)
    l3 = ax.plot(xcoords, np.mean(m,axis=0))
    ax.fill_between(xcoords, np.mean(m,axis=0) - 2/np.sqrt(np.shape(m)[0])*np.std(m,axis=0), np.mean(m,axis=0) + 2/np.sqrt(np.shape(m)[0])*np.std(m,axis=0),alpha=0.1)
    l4= ax.plot(xcoords, np.mean(e,axis=0))
    ax.fill_between(xcoords, np.mean(e,axis=0) - 2/np.sqrt(np.shape(e)[0])*np.std(e,axis=0), np.mean(e,axis=0) + 2/np.sqrt(np.shape(e)[0])*np.std(e,axis=0),alpha=0.1)
    l5= ax.plot(xcoords, np.mean(z,axis=0))
    ax.fill_between(xcoords, np.mean(z,axis=0) - 2/np.sqrt(np.shape(z)[0])*np.std(z,axis=0), np.mean(z,axis=0) + 2/np.sqrt(np.shape(z)[0])*np.std(z,axis=0),alpha=0.1)
    plt.sca(ax)
    plt.xlim([0,Args.T])
    plt.xticks([0, 500, 1000, 1500, 2000], fontsize=16)
    # plt.ylim([0,100])
    # plt.yticks([0,50,100], fontsize=16)
    plt.xlabel('T', fontsize=16)
    if ylabel:
        plt.ylabel('Regret', fontsize=16)
    legendHandles = []
    legendHandles.append((matplotlib.patches.Patch(color=l1[0].get_color(), label="BOSE"), "BOSE"))
    legendHandles.append((matplotlib.patches.Patch(color=l2[0].get_color(), label="LinUCB"), "LinUCB"))
    legendHandles.append((matplotlib.patches.Patch(color=l3[0].get_color(), label="ILTCB"), "ILTCB"))
    legendHandles.append((matplotlib.patches.Patch(color=l4[0].get_color(), label="EpsGreedy"), "EpsGreedy"))
    legendHandles.append((matplotlib.patches.Patch(color=l5[0].get_color(), label="Thompson"), "Thompson"))
    # plt.legend(['Ours', 'LinUCB', 'ILTCB', 'EpsGreedy'])
    return(legendHandles)

if __name__=='__main__':
    import sys, os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=50000, help='number of rounds', type=int)
    parser.add_argument('--d', action='store', default=20, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)
    parser.add_argument('--noise', action='store', default=0.1, type=float)
    parser.add_argument('--save', action='store', default=False, type=bool)
    parser.add_argument('--kind', action='store', default='regrets', type=str)
    parser.add_argument('--final', action='store', default=False, type=bool)

    Args = parser.parse_args(sys.argv[1:])
    deltas = [float('%0.3f' % x) for x in np.logspace(-3,1,20)]
    eps = [float('%0.3f' % x) for x in np.logspace(-3,0,20)]

    if Args.final:
        fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0]*3, mpl.rcParams['figure.figsize'][1]))
        ax = fig.add_subplot(111,frameon=False)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.subplots_adjust(left=0.03, right=0.99,bottom=0.3)

        prefix = '../results/linear_sphere_T=2000_d=10_K=2_sig=0.5/'
        (best_bose, best_lin, best_mini, best_eps) = cv(prefix, Args)
        cv_plot(prefix, best_bose, best_lin, best_mini, best_eps, ax1, Args,kind=Args.kind)
        plt.title('Linear Sphere', fontsize=16)
        
        prefix = '../results/semiparametric_sphere_T=2000_d=10_K=2_sig=0.5/'
        (best_bose, best_lin, best_mini, best_eps) = cv(prefix, Args)
        cv_plot(prefix, best_bose, best_lin, best_mini, best_eps, ax2, Args,kind=Args.kind,ylabel=False)
        plt.title('Confounded Sphere', fontsize=16)

        prefix = '../results/semiparametric_pos_T=2000_d=10_K=2_sig=0.5/'
        (best_bose, best_lin, best_mini, best_eps) = cv(prefix, Args)
        (legendHandles) = cv_plot(prefix, best_bose, best_lin, best_mini, best_eps, ax3, Args,kind=Args.kind,ylabel=False)
        plt.title('Confounded Orthant', fontsize=16)

        leg = ax2.legend([x[1] for x in legendHandles], loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, ncol=7, frameon=False,fontsize=16)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)

        plt.tight_layout()
        if Args.save:
            plt.savefig("../figs/final.pdf", format='pdf', dpi=100,bbox_inches='tight')
        else:
            plt.show()

    else:
        fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0]*2, mpl.rcParams['figure.figsize'][1]-1))
        ax = fig.add_subplot(111,frameon=False)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        
        prefix = '../results/%s_sphere_T=%d_d=%d_K=%d_sig=%0.1f/' % ('linear', Args.T, Args.d, Args.K, float(Args.noise))
        (best_bose, best_lin, best_thompson, best_mini, best_eps) = cv(prefix, Args)
        cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax1, Args,kind=Args.kind)
        plt.title('Linear Sphere d=%d K=%d' % (Args.d, Args.K))
        
        prefix = '../results/%s_pos_T=%d_d=%d_K=%d_sig=%0.1f/' % ('linear', Args.T, Args.d, Args.K, float(Args.noise))
        (best_bose, best_lin, best_thompson, best_mini, best_eps) = cv(prefix, Args)
        cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax2, Args,kind=Args.kind)
        plt.title('Linear Orthant d=%d K=%d' % (Args.d, Args.K))

        if Args.save:
            plt.savefig('../figs/linear_%s_T=%d_d=%d_K=%d_sig=%0.1f.pdf' % (Args.kind, Args.T, Args.d, Args.K, float(Args.noise)), format='pdf', dpi=100,bbox_inches='tight')

        fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0]*2, mpl.rcParams['figure.figsize'][1]-1))
        ax = fig.add_subplot(111,frameon=False)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        
        prefix = '../results/%s_sphere_T=%d_d=%d_K=%d_sig=%0.1f/' % ('semiparametric', Args.T, Args.d, Args.K, float(Args.noise))
        (best_bose, best_lin, best_thompson, best_mini, best_eps) = cv(prefix, Args)
        cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax1, Args, kind=Args.kind)
        plt.title('Non-linear Sphere d=%d K=%d' % (Args.d, Args.K))
        
        prefix = '../results/%s_pos_T=%d_d=%d_K=%d_sig=%0.1f/' % ('semiparametric', Args.T, Args.d, Args.K, float(Args.noise))
        (best_bose, best_lin, best_thompson, best_mini, best_eps) = cv(prefix, Args)
        cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax2, Args, kind=Args.kind)
        plt.title('Non-linear Orthant d=%d K=%d' % (Args.d, Args.K))

        if Args.save:
            plt.savefig('../figs/semiparametric_%s_T=%d_d=%d_K=%d_sig=%0.1f.pdf' % (Args.kind, Args.T, Args.d, Args.K, float(Args.noise)), format='pdf', dpi=100,bbox_inches='tight')

        if not Args.save:
            plt.show()
