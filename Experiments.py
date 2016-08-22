import Simulators, Semibandits
import numpy as np
import scipy.linalg
import sklearn.linear_model, sklearn.tree, sklearn.ensemble
import Policy
import pickle, argparse, warnings, sys, os

"""
Some scripts for running algorithms on various problems and logging the results. 
"""
def SyntheticExperiment(iters=5, T=1000, N=2000, K=25, L=5, eps=0.5):
    """
    Run an experiment using the EnumerationPolicy class.  Here we can
    keep track of the optimal policy performance and therefore the
    regrets as well as the rewards.
    """
    Mon = Semibandits.MiniMonster
    Eps = Semibandits.EpsGreedy
    Exp = Semibandits.SemiExp4
    params = {}
    ## Generate mini_monster params_dicts:
    params['mini'] = [dict(mu=x) for x in np.arange(0.05,0.5,0.05)]
    params['eps'] = [dict(eps=x) for x in np.arange(0.05,0.5,0.05)]
    params['exp'] = [dict(gamma=x,eta=y) for x in np.arange(0.05,0.2,0.05) for y in np.arange(0.05,0.2,0.05)]

    algs = {
        'mini': Mon,
        'eps': Eps,
        'exp': Exp
        }
    rewards = {}
    regrets = {}
    for (k,v) in params.items():
        for d in v:
            rewards[k+"_"+"_".join([a+"="+str(b) for (a,b) in d.items()])] = []
            regrets[k+"_"+"_".join([a+"="+str(b) for (a,b) in d.items()])] = []
    for i in range(iters):
        for (k,v) in algs.items():
            for p in params[k]:
                alg_name=k+"_"+"_".join([a+"="+str(b) for (a,b) in p.items()])
                B = Simulators.SemibanditSim(T,N,K,L,eps)
                alg = v(B)
                (r,reg) = alg.play(T, params=p, verbose=False)
                rewards[alg_name].append(r)
                regrets[alg_name].append(reg)
                print ("%s reward=%0.2f" % (alg_name, r[len(r)-1]), flush = True)
    pickle.dump(dict(rewards=rewards,regrets=regrets), open("./out/synth_T=%d_N=%d_K=%d_L=%d.out" % (T,N,K,L),"wb"))
    return (rewards, regrets)

def LinearExperiment(ks, ds, iters=5, T = 100):
    """
    Run LinUCB against MiniMonster on a synthetic LinearBandit problem.
    """
    outs = {}
    for k in ks:
        outs[k] = {}
        for d in ds:
            outs[k][d] = []
            for i in range(iters):
                print("Starting simulation with k=%d d=%d i=%d" % (k,d,i), flush=True)
                B = Simulators.LinearBandit(d, 1, k, noise=True)
                M = Semibandits.MiniMonster(B, sklearn.linear_model.LinearRegression)
                (r,reg) = M.play(T, verbose=False)
                L = Semibandits.LinUCB(B)
                (r2,reg2) = L.play(T, verbose=False)
                print("MiniMonster: %0.2f LinUCB: %0.2f" % (r[len(r)-1], r2[len(r2)-1]), flush=True)
                outs[k][d].append((r, r2))
    return outs


def run_mini_monster(Bconstructor, Bval, learning_algs, rewards, risks, val, T, slate_len, i, start, params=None):
    for (k,v) in learning_algs.items():
        if params is None:
            B = Bconstructor()
            M = Semibandits.MiniMonster(B, learning_alg=v, classification=False)
            (r,reg,val_tmp) = M.play(T,verbose=True, params={'weight': np.ones(slate_len)},validate=Bval)
            rewards['mini_%s' % (k)].append(r)
            print("Num Uniform Actions: %d" % (M.num_unif), flush=True)
            np.savetxt(outdir+"mini_%s_rewards_%d.out" % (k, i), r)
            if Bval is not None:
                val['mini_%s' % (k)].append(val_tmp)
                print('Mini %s Final Validation %0.2f' % (k, val['mini_%s' % (k)][i-start][-1]), flush=True)
                np.savetxt(outdir+"mini_%s_validation_%d.out" % (k, i), val_tmp)
        if params is not None:
            for a in params:
                B = Bconstructor()
                M = Semibandits.MiniMonster(B, learning_alg=v, classification=False)
                (r,reg,val_tmp) = M.play(T,verbose=True, params={'weight': np.ones(slate_len), 'mu': a},validate=Bval)

                rewards['mini_%s_%0.3f' % (k,a)].append(r)
                print("Num Uniform Actions: %d" % (M.num_unif), flush=True)
                np.savetxt(outdir+"mini_%s_%0.3f_rewards_%d.out" % (k, a, i), r)
                if Bval is not None:
                    val['mini_%s_%0.3f' % (k,a)].append(val_tmp)
                    print('Mini %s %0.3f Final Validation %0.2f' % (k, a, val['mini_%s_%0.3f' % (k,a)][i-start][-1]), flush=True)
                    np.savetxt(outdir+"mini_%s_%0.3f_validation_%d.out" % (k, a, i), val_tmp)
    return (rewards, risks, val)

def run_eps_greedy(Bconstructor, Bval, learning_algs, rewards, risks, val, T, slate_len, i, start, params=None):
    for (k,v) in learning_algs.items():
        if params is None:
            B = Bconstructor()
            M = Semibandits.EpsGreedy(B, learning_alg=v, classification=False)
            (r,reg,val_tmp) = M.play(T,verbose=True, params={'weight': np.ones(slate_len)},validate=Bval)
            rewards['eps_%s' % (k)].append(r)
            print("Num Uniform Actions: %d" % (M.num_unif), flush=True)
            np.savetxt(outdir+"eps_%s_rewards_%d.out" % (k, i), r)
            if Bval is not None:
                val['eps_%s' % (k)].append(val_tmp)
                print('Eps %s Final Validation %0.2f' % (k, val['eps_%s' % (k)][i-start][-1]), flush=True)
                np.savetxt(outdir+"eps_%s_validation_%d.out" % (k, i), val_tmp)

            B = Bconstructor()
            M = Semibandits.EpsGreedy(B, learning_alg=v, classification=False)
            (r,reg,val_tmp) = M.play(T,verbose=True, params={'weight': np.ones(slate_len), 'train_all': True},validate=Bval)
            rewards['epsall_%s' % (k)].append(r)
            print("Num Uniform Actions: %d" % (M.num_unif), flush=True)
            np.savetxt(outdir+"epsall_%s_rewards_%d.out" % (k, i), r)
            if Bval is not None:
                val['epsall_%s' % (k)].append(val_tmp)
                print('EpsAll %s Final Validation %0.2f' % (k, val['epsall_%s' % (k)][i-start][-1]), flush=True)
                np.savetxt(outdir+"epsall_%s_validation_%d.out" % (k, i), val_tmp)
        if params is not None:
            for a in params:
                B = Bconstructor()
                M = Semibandits.EpsGreedy(B, learning_alg=v, classification=False)
                (r,reg,val_tmp) = M.play(T,verbose=True, params={'weight': np.ones(slate_len), 'eps': a, 'train_all': False},validate=Bval)
                rewards['eps_%s_%0.3f' % (k,a)].append(r)
                print("Num Uniform Actions: %d" % (M.num_unif), flush=True)
                np.savetxt(outdir+"eps_%s_%0.3f_rewards_%d.out" % (k, a, i), r)
                if Bval is not None:
                    val['eps_%s_%0.3f' % (k,a)].append(val_tmp)
                    print('Eps %s %0.3f Final Validation %0.2f' % (k, a, val['eps_%s_%0.3f' % (k,a)][i-start][-1]), flush=True)
                    np.savetxt(outdir+"eps_%s_%0.3f_validation_%d.out" % (k, a, i), val_tmp)

                
                B = Bconstructor()
                M = Semibandits.EpsGreedy(B, learning_alg=v, classification=False)
                (r,reg,val_tmp) = M.play(T,verbose=True, params={'weight': np.ones(slate_len), 'eps': a, 'train_all': True},validate=Bval)
                rewards['epsall_%s_%0.3f' % (k,a)].append(r)
                print("Num Uniform Actions: %d" % (M.num_unif), flush=True)
                np.savetxt(outdir+"epsall_%s_%0.3f_rewards_%d.out" % (k, a, i), r)
                if Bval is not None:
                    val['epsall_%s_%0.3f' % (k,a)].append(val_tmp)
                    print('EpsAll %s %0.3f Final Validation %0.2f' % (k, a, val['epsall_%s_%0.3f' % (k,a)][i-start][-1]), flush=True)
                    np.savetxt(outdir+"epsall_%s_%0.3f_validation_%d.out" % (k, a, i), val_tmp)
    return (rewards, risks, val)

def run_lin_ucb(Bconstructor, Bval, rewards, risks, val, T, slate_len, i, start, params=None):
    if params is None:
        B = Bconstructor()
        L = Semibandits.LinUCB(B)
        (r,reg,val_tmp) = L.play(T,verbose=True, validate=Bval)
        rewards['lin'].append(r)
        np.savetxt(outdir+"lin_rewards_%d.out" % (i), r)
    
        if Bval is not None:
            np.savetxt(outdir+"lin_validation_%d.out" % (i), val_tmp)
            val['lin'].append(val_tmp)
            print('Lin Final Validation %0.2f' % (val['lin'][i-start][-1]), flush=True)
    if params is not None:
        for a in params:
            B = Bconstructor()
            L = Semibandits.LinUCB(B)
            (r,reg,val_tmp) = L.play(T,verbose=True, validate=Bval, params={'delta': a})
            rewards['lin_%0.5f' % (a)].append(r)
            np.savetxt(outdir+"lin_%0.5f_rewards_%d.out" % (a, i), r)
            if Bval is not None:
                np.savetxt(outdir+"lin_%0.5f_validation_%d.out" % (a, i), val_tmp)
                val['lin_%0.5f' % (a)].append(val_tmp)
                print('Lin %0.5f Final Validation %0.2f' % (a, val['lin_%0.5f' % (a)][i-start][-1]), flush=True)
    return (rewards, risks, val)

def DatasetExperiment(outdir="./results/", dataset="letter", start = 0, iters = 5, T = 1000, L=1, grid=True, noise=None, alg='all'):
    """
    Compare several algorithms on a real dataset. 
    """
    loop = True
    if dataset=="mslr" or dataset=='mslrsmall' or dataset=='mslr30k':
        loop = False
    slate_len=L
    min_samples_leaf=20
    forest_size = 50
    if grid:
        eps_vals = np.logspace(-3, 0, 10) ## np.arange(0.01,0.21,0.01)
        mu_vals = np.logspace(-3, 1, 10) ## np.arange(0.1, 2.1, 0.1)
        delta_vals = np.logspace(-3, 1, 10) ## np.arange(0.1, 2.1, 0.1)
    else:
        mu_vals = None
        eps_vals = None
        delta_vals = None

    B = Simulators.DatasetBandit(L=slate_len, loop=False, dataset=dataset, metric=None, noise=noise)
    if dataset == 'mslr' or dataset == 'mslrsmall':
        Bval = None
    else:
        Bval = Simulators.DatasetBandit(L=slate_len, loop=False, dataset=dataset, metric=None, noise=noise)

    rewards = {'random': []}
    risks = {'random': []}
    val = {'random': []}

    if dataset=='xor':
        Algs = {
            'lin': lambda: sklearn.linear_model.LinearRegression(),
            'tree': lambda: sklearn.tree.DecisionTreeRegressor(max_depth=2)
            }
    elif dataset=='mslr30k':
        Algs = {'gb5': lambda: sklearn.ensemble.GradientBoostingRegressor(max_depth=5, n_estimators=forest_size),
                'gb2': lambda: sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=forest_size),
                'lin': lambda: sklearn.linear_model.LinearRegression()}
    else:
        Algs = {'gb2': lambda: sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=forest_size),
                'lin': lambda: sklearn.linear_model.LinearRegression()}

    Models = {}

    if B.has_ldf:
        ## Initialize output data structure
        for k in Algs.keys():
            for a in ['best', 'mini', 'eps', 'epsall']:
                rewards['%s_%s' % (a,k)] = []
                risks['%s_%s' % (a,k)] = []
                val['%s_%s' % (a,k)] = []
        rewards['lin'] = []
        risks['lin'] = []
        val['lin'] = []

        if grid:
            for k in Algs.keys():
                for a in eps_vals:
                    rewards['eps_%s_%0.3f' % (k,a)] = []
                    risks['eps_%s_%0.3f' % (k,a)] = []
                    val['eps_%s_%0.3f' % (k,a)] = []
                    rewards['epsall_%s_%0.3f' % (k,a)] = []
                    risks['epsall_%s_%0.3f' % (k,a)] = []
                    val['epsall_%s_%0.3f' % (k,a)] = []
                for b in mu_vals:
                    rewards['mini_%s_%0.3f' % (k,b)] = []
                    risks['mini_%s_%0.3f' % (k,b)] = []
                    val['mini_%s_%0.3f' % (k,b)] = []
            for c in delta_vals:
                rewards['lin_%0.5f' % (c)] = []
                risks['lin_%0.5f' % (c)] = []
                val['lin_%0.5f' % (c)] = []

        ## Compute the best linear and best_tree policies
        for (k,v) in Algs.items():
            print("Calling get_best_policy for %s" % (k), flush=True)
            Models[k] = B.get_best_policy(T=T, learning_alg=v, classification=False)
            if Bval is not None:
                val['best_%s' % (k)].append(Bval.offline_evaluate(Models[k], train=False, T=T))
                for i in range(start,start+iters):
                    np.savetxt(outdir+"best_%s_validation_%d.out" % (k, i), np.array(val['best_%s' % (k)]))
                    print('Best %s Reg Val %0.2f' % (k, val['best_%s' % (k)][0]), flush=True)


    for i in range(start,start+iters):
        print("Starting iteration %d" % (i+1), flush=True)

        if B.has_ldf:
            if alg == 'eps' or alg == 'all':
                ### RUN EPS-GREEDY ###
                (rewards, risks, val) = run_eps_greedy(
                    lambda: Simulators.DatasetBandit(L=slate_len,loop=loop,dataset=dataset, metric=None, noise=noise),
                    Bval, Algs, rewards, risks, val, T, slate_len, i, start, params=eps_vals) 
            if alg == 'mini' or alg == 'all':
                ### RUN MINI-MONSTER ###
                (rewards, risks, val) = run_mini_monster(
                    lambda: Simulators.DatasetBandit(L=slate_len,loop=loop,dataset=dataset,metric=None,noise=noise),
                    Bval, Algs, rewards, risks, val, T, slate_len, i, start, params=mu_vals)
            if alg == 'lin' or alg == 'all':
                ### RUN LinUCB ###
                (rewards, risks, val) = run_lin_ucb(
                    lambda: Simulators.DatasetBandit(L=slate_len,loop=loop,dataset=dataset,metric=None,noise=noise),
                    Bval, rewards, risks, val, T, slate_len, i, start, params=delta_vals)
        ### OFFLINE EVALUATIONS ###
        tmp_rewards = {'random': []}
        for k in Models.keys():
            tmp_rewards[k] = []

        B = Simulators.DatasetBandit(L=slate_len,loop=loop,dataset=dataset, metric=None, noise=noise)
        for t in range(T):
            x = B.get_new_context()
            if x == None:
                continue
            ## Random policy
            p = np.random.choice(range(x.get_K()), slate_len)
            tmp_rewards['random'].append(B.get_slate_reward(p))
            if B.has_ldf:
                for (k,v) in Models.items():
                    tmp_rewards[k].append(B.get_slate_reward(Models[k].get_action(x)))

        rewards['random'].append(np.cumsum(tmp_rewards['random'])[9::10])
        np.savetxt(outdir+'random_rewards_%d.out' % (i), np.cumsum(tmp_rewards['random'])[9::10])

        ## Random validation
        if Bval is not None:
            class RandomPolicy():
                def __init__(self, L):
                    self.L = L
                def get_action(self, x):
                    return np.random.choice(range(x.get_K()), self.L)
            random_policy = RandomPolicy(slate_len)
            val_tmp = Bval.offline_evaluate(random_policy, train=False, T=T)
            val['random'].append(val_tmp)
            np.savetxt(outdir+"random_validation_%d.out" % (i), np.array([val_tmp]))

        if B.has_ldf:
            for k in Models.keys():
                rewards['best_%s' % (k)].append(np.cumsum(tmp_rewards[k])[9::10])
                np.savetxt(outdir+"best_%s_rewards_%d.out" % (k, i), rewards['best_%s' % (k)][i-start])
    
    return (rewards, risks, val)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store',
                        default=1000,
                        help='number of rounds')
    parser.add_argument('--dataset', action='store', choices=['synth','mq2007','mq2008', 'yahoo', 'mslr', 'mslrsmall', 'mslr30k', 'xor'])
    parser.add_argument('--L', action='store', default=5)
    parser.add_argument('--start', action='store', default=0)
    parser.add_argument('--I', action='store', default=5)
    parser.add_argument('--noise', action='store', default=None)
    parser.add_argument('--grid', action='store', default=False)
    parser.add_argument('--alg', action='store' ,default='all', choices=['all', 'mini', 'eps', 'lin'])

    Args = parser.parse_args(sys.argv[1:])
    print(Args, flush=True)
    Args.T = int(Args.T)
    Args.L = int(Args.L)
    Args.I = int(Args.I)
    Args.start = int(Args.start)
    if Args.noise != None:
        Args.noise = float(Args.noise)
        outdir = './results/%s_T=%d_L=%d_e=%0.1f/' % (Args.dataset, Args.T, Args.L, Args.noise)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    else:
        outdir = './results/%s_T=%d_L=%d_e=0.0/' % (Args.dataset, Args.T, Args.L)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    out = DatasetExperiment(outdir = outdir, dataset=Args.dataset, start = Args.start, iters = Args.I, T = Args.T, L=Args.L, noise=Args.noise, grid=Args.grid, alg=Args.alg)
    print("---- DONE ----", flush=True)
