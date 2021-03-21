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

import scipy
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
np.set_printoptions(precision=2, linewidth=180, suppress=True)


def parse():
    parser = argparse.ArgumentParser(description="Result collector")
    parser.add_argument('-f',   '--folder',   dest="FOLDER",  type=str,   help='Folder to process')

    args = parser.parse_args()
    return args

def f(d):
    arr = np.genfromtxt('{}/data_values.csv'.format(d),delimiter=',')

    n = 20

    max_avg = -9E9
    for i in range(n, arr.shape[0]):
        a = np.sum(arr[i-n:i,1])/n
        max_avg = max(max_avg, a)

    # arr = arr.tolist()
    
    # arr = sorted(arr, reverse=True, key=lambda e: e[1])

    return d, max_avg

def main():
    args = parse()

    base_dir = args.FOLDER

    dirs = glob.glob('{}/*/'.format(base_dir))
    
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

    y = range(1,21)
    x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    y = [50000, 40000, 30000, 20000, 10000, 5000, 3000, 1000]
    x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    x = np.array(range(1,101))/100
    y = range(1,21)
    
    z = []
    z2 = []
    for e in res:
        z.append(e[1])
        z2.append(e[0])

    z = np.reshape(z, (len(y), len(x)))
    z2 = np.reshape(z2, (len(y), len(x)))

    x = np.array(x)
    y = np.array(y)
    x, y = np.meshgrid(x, y)

    np.save('x.npy', x)
    np.save('y.npy', y)
    np.save('z.npy', z)

    print("---------")
    print("---------")
    print(x[0:3,0:3])
    print("---------")
    print(y[0:3,0:3])
    print("---------")
    print(z2[0:3,0:3])

    def plot(x, y, z, filename):

        plt.ioff()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, z, rcount=100, ccount=100, edgecolor='none', cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.savefig(filename, dpi=400)

    plot(x, y, z, "out.png")
    z = scipy.ndimage.gaussian_filter(z, sigma=4)
    plot(x, y, z, "out2.png")

if __name__=='__main__':
    main()
