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
parser.add_argument('-f',   '--folder',   dest="FOLDERS", nargs='+',                type=str,   help='Folder to process',   required=True)
parser.add_argument('-s',   '--sigma',    dest="SIGMA",   nargs='+',  default=4,    type=float, help='Sigma for gaussian')
parser.add_argument('-n',   '--num',      dest="NUM",                 default=20,   type=int,   help='Number of values to avg together')
parser.add_argument('-x',   '--xvals',    dest='X',       nargs='+',                type=float, help='xvals for graph')
parser.add_argument('-y',   '--yvals',    dest='Y',       nargs='+',                type=float, help='yvals for graph')
parser.add_argument('-c',   '--cutoff',   dest="CUTOFF",              default=None, type=str,   help='Dont check dirs that exist after this one')
args = parser.parse_args()

print(args)

def f(d):
    arr = np.genfromtxt('{}/data_values.csv'.format(d),delimiter=',')

    n = args.NUM

    max_avg = -9E9
    for i in range(n, arr.shape[0]):
        a = np.sum(arr[i-n:i,1])/n
        max_avg = max(max_avg, a)

    # arr = arr.tolist()
    
    # arr = sorted(arr, reverse=True, key=lambda e: e[1])

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

    # y = range(1,21)
    # x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    # y = [50000, 40000, 30000, 20000, 10000, 5000, 3000, 1000]
    # x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

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

    def plot(x, y, z, filename, angle=None):

        def draw_sphere2(xCenter, yCenter, zCenter, r, ax):
            #draw sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x=np.cos(u)*np.sin(v)
            y=np.sin(u)*np.sin(v)
            z=np.cos(v)
            # shift and scale sphere
            # xlen = (max(xarr)-min(xarr))/10
            # ylen = (max(yarr)-min(yarr))/10
            # zlen = (np.max(np.max(z))-np.min(np.min(z)))/10

            xmin, xmax = ax.get_xlim()
            xlen = (xmax-xmin)/100
            ymin, ymax = ax.get_ylim()
            ylen = (ymax-ymin)/100
            zmin, zmax = ax.get_zlim()
            zlen = (zmax-zmin)/100

            print(xlen, ylen, zlen)

            x = r*xlen*x + xCenter
            y = r*ylen*y + yCenter
            z = r*zlen*z + zCenter
            return (x,y,z)

        # def draw_crosshair(xCenter, yCenter, zCenter, r):
        #     #draw sphere
        #     xlen = (max(xarr)-min(xarr))/10
        #     ylen = (max(yarr)-min(yarr))/10
        #     x, y = np.mgrid[0:xlen, 0:ylen]
        #     z = np.zeros((xlen,ylen))
        #     z[:, ylen//2] = 50
        #     z[xlen//2, :] = 50
        #     return (x,y,z)


        # xpt = 10000
        # ypt = 32
        # zpt = z[np.where(yarr == ypt), np.where(xarr == xpt)][0][0]

        plt.ioff()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.plot([xpt], [ypt], [zpt], 'go', alpha=0.5)

        # xmin, xmax = min(xarr), max(xarr)
        # ymin, ymax = min(yarr), max(yarr)
        # zmin, zmax = (-50,80)
        # c = np.linspace(zmin,zmax,100)
        # a = np.ones(c.shape)*xpt
        # b = np.ones(c.shape)*ypt
        # ax.plot(a,b,c)


        ax.plot_surface(x, y, z, rstride=1, cstride=1, edgecolor='none', cmap=cm.coolwarm,
                        linewidth=1, antialiased=False, zorder = 0.1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # print(xpt, ypt, zpt)

        # print("WHERE:", np.where(yarr == 32), np.where(xarr == 10000))
        # ax.plot(xpt, ypt, zpt, 'ro', alpha=0.5)

        # (xs,ys,zs) = draw_crosshair(10000, 32, z[ypt[0]//400, xpt[0]//4000], 20)
        # print(xs, ys, zs)

        # ax.plot_wireframe(xs, ys, zs, color="r")

        if angle is not None:
            ax.view_init(angle[0], angle[1]) 

        plt.savefig(filename, dpi=600, bbox_inches='tight')

    plot(x, y, z, "plt1_raw.png")

    import scipy
    from scipy.ndimage import gaussian_filter
    
    z = scipy.ndimage.gaussian_filter(z, sigma=args.SIGMA)
    plot(x, y, z, "plt2_gaussian.png")

    plot(x, y, z, "plt3_xy.png", angle=(90, -90))
    plot(x, y, z, "plt4_zx.png", angle=(0, -90))
    plot(x, y, z, "plt5_zy.png", angle=(0, 0))

if __name__=='__main__':
    main()
