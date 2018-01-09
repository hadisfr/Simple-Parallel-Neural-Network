#!/usr/bin/env python3


import math


def func(in1, in2, in3):
    return -in2 + math.sqrt(abs(in2**2 - 4*in1*in3)) / (2*in1 + math.sin(in1*math.pi))


def main():
    data = []
    with open("InputFile.txt") as f:
        for line in f:
            data.append([float(x) for x in line.split()])
    res = [func(*i) for i in data]
    with open("res.txt", "w") as f:
        for i in res:
            print(i, file=f)


if __name__ == '__main__':
    main()
