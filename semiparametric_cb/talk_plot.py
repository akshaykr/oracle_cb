import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib as mpl
import cv_plot


if __name__=='__main__':
    import sys, os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store', default=50000, help='number of rounds', type=int)
    parser.add_argument('--d', action='store', default=20, type=int)
    parser.add_argument('--K', action='store', default=5, type=int)
    parser.add_argument('--noise', action='store', default=0.1, type=float)
    parser.add_argument('--save', action='store', default=False, type=bool)
    parser.add_argument('--final', action='store', default=False, type=bool)

    Args = parser.parse_args(sys.argv[1:])
    deltas = [float('%0.3f' % x) for x in np.logspace(-3,1,20)]
    eps = [float('%0.3f' % x) for x in np.logspace(-3,0,20)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    prefix = '../results/%s_sphere_T=%d_d=%d_K=%d_sig=%0.1f/' % ('linear', Args.T, Args.d, Args.K, float(Args.noise))
    (best_bose, best_lin, best_thompson, best_mini, best_eps) = cv_plot.cv(prefix, Args)
    cv_plot.cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax, Args)
    plt.title('Linear Reward')

    plt.savefig('../figs/linear_%s_T=%d_d=%d_K=%d_sig=%0.1f.pdf' % ('regrets', Args.T, Args.d, Args.K, float(Args.noise)), format='pdf', dpi=100,bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    prefix = '../results/%s_sphere_T=%d_d=%d_K=%d_sig=%0.1f/' % ('semiparametric', Args.T, Args.d, Args.K, float(Args.noise))
    (best_bose, best_lin, best_thompson, best_mini, best_eps) = cv_plot.cv(prefix, Args)
    cv_plot.cv_plot(prefix, best_bose, best_lin, best_thompson, best_mini, best_eps, ax, Args)
    plt.title('Confounded Reward')

    plt.savefig('../figs/semiparametric_%s_T=%d_d=%d_K=%d_sig=%0.1f.pdf' % ('regrets', Args.T, Args.d, Args.K, float(Args.noise)), format='pdf', dpi=100,bbox_inches='tight')
