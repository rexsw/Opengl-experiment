import numpy as np
from math import floor, factorial


def noise(size):
    ret = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            i -= 1
            j -= 1
            if i < 0 or j < 0:
                ix = i
                jx = j
                if jx < 0:
                    jx = 0
                if ix < 0:
                    ix = 0
                f = np.array((factorial(ix),factorial(jx)))
            else:
                f = np.array((factorial(i),factorial(j)))

            a = np.random.rand(1)
            b = np.random.rand(1)
            c = np.random.rand(1)
            d = np.random.rand(1)
            u = f * f * (3 - 2*f);

            ret[i][j] = (a*(1-u[0]) +b*u[0])  + (c - a)* u[1] * (1.0 - u[0]) + (d - b) * u[0] * u[1]
    return ret

def bmnoise(size):
    indexs = np.zeros((size[0],size[1],2))
    for i in range(size[0]):
        for j in range(size[1]):
            indexs[i][j] = (i,j)
    value = 0
    freq = 1
    amp = .5
    octaves = 10

    for i in range(octaves):
        value += amp * (noise(size) * freq)
        freq *= 1.9;
        amp *= .6;
    return value;


