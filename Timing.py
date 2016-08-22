import os, sys
import argparse
import numpy as np

mslr_names = [
    'mini_gb2_3.594',
    'lin_0.022',
    'epsall_gb2_0.100',
    'mini_lin_3.594',
    'epsall_lin_0.100',
    'mini_gb5_0.022',
    'espall_gb5_0.100'
]

yahoo_names = [
    'mini_gb2_3.594',
    'lin_0.060',
    'epsall_gb2_0.046',
    'mini_lin_0.008',
    'epsall_lin_0.100',
    'mini_gb5_0.022',
    'espall_gb5_0.001'
]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store')

    Args = parser.parse_args(sys.argv[1:])

    if Args.file.find('mslr') != -1:
        names = mslr_names
    else:
        names = yahoo_names
    Timing = {}
    f = Args.file
    for x in os.listdir(f):
        if True in [x.find(y) == 0 for y in names]:
            t = np.loadtxt(f+x)
            name ="_".join(x.split("_")[0:3])
            if name in Timing.keys():
                Timing[name].append(t)
            else:
                Timing[name] = [t]
    for (k,v) in Timing.items():
        print("%s: %d" % (k, np.mean(v)/60))
        print([int(x/60) for x in v])
