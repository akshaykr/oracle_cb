import sys, argparse, os
sys.path.append('../')
import settings

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store',
                        default=1000,
                        help='number of rounds')
    parser.add_argument('--dataset', action='store', choices=['synth','mq2007','mq2008', 'yahoo'])
    ## parser.add_argument('--policies', action='store', choices=['finite', 'tree', 'linear'], default='linear')
    parser.add_argument('--L', action='store', default=3)
    parser.add_argument('--I', action='store', default=5)
    parser.add_argument('--start', action='store', default=0)
    parser.add_argument('--server', action='store')
    parser.add_argument('--user', action='store')

    Args = parser.parse_args(sys.argv[1:])
    print(Args)

    

    string = "ssh %s@%s 'cd ~/projects/oracle_cb/code/; %s Experiments.py --dataset %s --T %s --I %s --L %s --noise 0.1 --grid True --start %s > logs/%s_T=%s_L=%s_S=%s.log 2>&1 &'" % (Args.user, Args.server, settings.REMOTE_PATH_TO_PYTHON, Args.dataset, Args.T, Args.I, Args.L, Args.start, Args.dataset, Args.T, Args.L, Args.start)
    
    print(string)
    os.system(string)

