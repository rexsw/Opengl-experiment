import numpy as np
import pyrr

def gen_sphere(resoultion):
    space = np.linspace(-1, 1, num=resoultion)
    indices = []
    ret = np.zeros((6,resoultion,resoultion,3))
    index = -1

    for k in range(6):
        mark = index
        for i_x, i in enumerate(space):
            for i_y, j in enumerate(space):
                index += 1
                vertex = pyrr.vector.normalise(order(i,j,k))
                x,y,z = vertex
                ret[k,i_x,i_y,:] = [x,y,z]
                if index > mark + resoultion and not(index % resoultion == 0):
                    indices.extend([index-resoultion-1,index-resoultion,index-1])
                    indices.extend([index-resoultion,index-1,index])
    return ret, indices

def order(i,j,k):
    if k == 0:
        return [i,j,1]
    if k == 1:
        return [i,j,-1]
    if k == 2:
        return [1,i,j]
    if k == 3:
        return [-1,i,j]
    if k == 4:
        return [j,1,i]
    if k == 5:
        return [j,-1,i]


def gen_cube(resoultion):
    space = np.linspace(-1, 1, num=resoultion)
    indices = []
    ret = np.zeros((6,resoultion,resoultion,3))
    index = -1
    for k in range(6):
        mark = index
        for i_x, i in enumerate(space):
            for i_y, j in enumerate(space):
                index += 1
                vertex = order(i,j,k)
                x,y,z = vertex
                ret[k,i_x,i_y,:] = [x,y,z]
                if index > mark + resoultion and not(index % resoultion == 0):
                    indices.extend([index-resoultion-1,index-resoultion,index-1])
                    indices.extend([index-resoultion,index-1,index])
    return ret, indices