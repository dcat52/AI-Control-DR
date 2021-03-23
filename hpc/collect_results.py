import csv
import multiprocessing
import glob
import os
import time
import itertools

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm

matplotlib.use("Agg")
np.set_printoptions(precision=4, linewidth=180, suppress=True)

parser = argparse.ArgumentParser(description="Result collector")
parser.add_argument('-f',   '--folder',   dest="FOLDERS", nargs='+',                     type=str,   help='Folder to process',   required=True)
parser.add_argument('-s',   '--sigma',    dest="SIGMA",   nargs='+',  default=4,         type=float, help='Sigma for gaussian')
parser.add_argument('-n',   '--num',      dest="NUM",                 default=20,        type=int,   help='Number of values to avg together')
parser.add_argument('-x',   '--xvals',    dest='X',       nargs='+',                     type=float, help='xvals for graph')
parser.add_argument('-y',   '--yvals',    dest='Y',       nargs='+',                     type=float, help='yvals for graph')
parser.add_argument('-c',   '--cutoff',   dest="CUTOFF",              default=None,      type=str,   help='Dont check dirs that exist after this one')
parser.add_argument('--xlabel',           dest="X_LABEL",             default='X',       type=str,   help='Name for X Axis')
parser.add_argument('--ylabel',           dest="Y_LABEL",             default='Y',       type=str,   help='Name for Y Axis')
parser.add_argument('--zlabel',           dest="Z_LABEL",             default='Reward',  type=str,   help='Name for Z Axis')
args = parser.parse_args()

print(args)

def f(d):
    arr = np.genfromtxt('{}/data_values.csv'.format(d),delimiter=',')

    n = args.NUM

    max_avg = -9E9
    for i in range(n, arr.shape[0]):
        a = np.sum(arr[i-n:i,1])/n
        max_avg = max(max_avg, a)

    return d, max_avg

def main():

    base_dirs = args.FOLDERS

    dirs = []
    for d in base_dirs:
        dirs.extend(glob.glob('{}/*/'.format(d)))
    

    if args.CUTOFF is not None:
        print("CUTOFF:", args.CUTOFF)
        dirs.sort()

        for i in range(len(dirs)):
            if args.CUTOFF in dirs[i]:
                break

        dirs = dirs[:i]

    res = []
    with multiprocessing.Pool() as p:
        res = p.map(f, dirs)

    res.sort(key=lambda e: e[1], reverse=True)

    print("TOP 5:")
    for e in range(0,5):
        print(res[e])

    print("---------")
    print("---------")

    best = ("", -9E9)
    for e in res:
        if e[1] > best[1]:
            best = e

    # print(best)
    print("BEST FOLDER: ", best[0])
    print("BEST SCORE: ", best[1])

    res.sort(key=lambda e: e[0], reverse=False)

    x = np.array(list(range(1, 5)))
    y = np.array(list(range(1, 100, 2))) / 100

    X = args.X
    Y = args.Y
    if X is not None and len(X) > 0:
        x = np.array(X)
    if Y is not None and len(Y) > 0:
        y = np.array(Y)

    
    z = []
    z2 = []
    for e in res:
        z.append(e[1])
        z2.append(e[0])

    z = np.reshape(z, (len(y), len(x)))
    z2 = np.reshape(z2, (len(y), len(x)))

    xarr = np.array(x)
    yarr = np.array(y)
    x, y = np.meshgrid(xarr, yarr)

    np.save('x.npy', x)
    np.save('y.npy', y)
    np.save('z.npy', z)

    print("--------------")
    print("--------------")
    print("X mat")
    print(x[0:3,0:3])
    print("--------------")
    print("Y mat")
    print(y[0:3,0:3])
    print("--------------")
    print("Z mat")
    print(z[0:3,0:3])
    print("--------------")
    print("Z dir mat")
    print(z2[0:3,0:3])
    print("--------------")

    def plot3(x, y, z, filename, angle=None):

        plt.ioff()
        fig = plt.figure()

        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, edgecolor='none', cmap=cm.coolwarm,
                        linewidth=1, antialiased=False, zorder = 0.1)

        ax.set_xlabel(args.X_LABEL)
        ax.set_ylabel(args.Y_LABEL)
        ax.set_zlabel(args.Z_LABEL)

        if angle is not None:
            ax.view_init(angle[0], angle[1])

        if angle == (90, -90):
            ax.set_zticklabels([])
        if angle == (0, -90):
            ax.set_yticklabels([])
        if angle == (0, 0):
            ax.set_xticklabels([])

        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print("Saved plot {}.".format(filename))

    def plot2(x, y, filename):

        plt.ioff()
        fig = plt.figure()
        
        ax = fig.gca()
        plt.plot(x, y)

        ax.set_xlabel(args.X_LABEL)
        ax.set_ylabel(args.Y_LABEL)

        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print("Saved plot {}".format(filename))


    plt2d = yarr.shape == (1,)
    plt3d = not plt2d
    
    if plt2d:
        x = x[0]
        z = z[0]

        plot2(x, z, "plt1_raw.png")

        import scipy
        from scipy.ndimage import gaussian_filter
        z = scipy.ndimage.gaussian_filter(z, sigma=args.SIGMA)
        
        plot2(x, z, "plt2_smoothed.png")


    if plt3d:
        plot3(x, y, z, "plt1_raw.png")

        import scipy
        from scipy.ndimage import gaussian_filter
        z = scipy.ndimage.gaussian_filter(z, sigma=args.SIGMA)
        
        plot3(x, y, z, "plt2_gaussian.png")

        plot3(x, y, z, "plt3_xy.png", angle=(90, -90))
        plot3(x, y, z, "plt4_zx.png", angle=(0, -90))
        plot3(x, y, z, "plt5_zy.png", angle=(0, 0))

if __name__=='__main__':
    main()
