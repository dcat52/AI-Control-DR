import csv
import multiprocessing
import glob
import os
import time
import itertools

import argparse

import numpy as np

np.set_printoptions(precision=1, suppress=True)


def parse():
    parser = argparse.ArgumentParser(description="Result collector")
    # parser.add_argument('-T', '--train_dqn',    action='store_true',    help='Whether train mode')
    parser.add_argument('-f',   '--folder',   dest="FOLDER",  type=str,   help='Folder to process')

    args = parser.parse_args()
    return args

def f(d):
    # with open('{}/data_values.csv') as csvFile:
    #     reader = csv.reader(csvFile, delimiter=',')
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

    # dirs = os.listdir(base_dir)
    # print(dirs)

    dirs = glob.glob('{}/*/'.format(base_dir))
    print(dirs)
    
    res = []
    with multiprocessing.Pool() as p:
        res = p.map(f, dirs)
    
    # def fx(out, val):
    #     print(out)
    #     if val[1][0] < out[1][0]:
    #         return (val)

    # best = itertools.accumulate(res, fx)
    # print(list(best))

    # print(res)

    # best = ("", np.array([[9E9, -9E9]]))
    # for e in res:
    #     if e[1][0][1] > best[1][0][1]:
    #         best = e

    res.sort(key=lambda e: e[1], reverse=True)

    print("---------")
    print("---------")

    for e in range(0,5):
        print(res[e])

    print("---------")
    print("---------")

    best = ("", -9E9)
    for e in res:
        if e[1] > best[1]:
            best = e

    print("---------")
    print("---------")

    print(best)
    print(best[0])
    print(best[1])



if __name__=='__main__':
    main()
