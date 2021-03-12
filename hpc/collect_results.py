import csv
import multiprocessing
import glob
import os
import time
import itertools

import numpy as np

np.set_printoptions(precision=1, suppress=True)

def f(d):
    # with open('{}/data_values.csv') as csvFile:
    #     reader = csv.reader(csvFile, delimiter=',')
    arr = np.genfromtxt('{}/data_values.csv'.format(d),delimiter=',')
    arr = arr.tolist()
    arr = sorted(arr, reverse=True, key=lambda e: e[1])

    return d, arr




def main():
    base_dir = 'runs/2021.03.11__18.07.45'

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

    print(res)

    best = ("", np.array([[9E9, -9E9]]))
    for e in res:
        if e[1][0][1] > best[1][0][1]:
            best = e

    print("---------")
    print("---------")

    print(best[0])
    print(best[1][0])



if __name__=='__main__':
    main()