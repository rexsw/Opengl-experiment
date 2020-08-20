import numpy as np
import random 

def lsystem(size):
    noise = np.zeros(size)
    current = ["X"]
    index = (random.randint(0, size[0]-1),random.randint(0, size[1]-1))
    direction = (1,0)
    pos = []
    for i in range(7):
        new = []
        for x in current:
            if x == "X":
                new +="F+[[X]-X]-F[-FX]+X"

            if x == "F":
                new += "FF"
                noise[index] += 1
                index = ((index[0]+direction[0])%size[0],(index[1]+direction[1])%size[1])
            if x == '+':
                if direction == (1,0):
                    direction = (0,1)
                elif direction == (0,1):
                    direction = (-1,0)
                elif direction == (-1,0):
                    direction = (0,-1)
                elif direction == (0,-1):
                    direction = (1,0)
            if x == '-':
                if direction == (1,0):
                    direction = (0,-1)
                elif direction == (0,1):
                    direction = (1,0)
                elif direction == (-1,0):
                    direction = (0,1)
                elif direction == (0,-1):
                    direction = (-1,0)
            if x == '[':
                pos.append([index,direction])
            if x == ']':
                index,direction = pos.pop(-1)
        current = new
    return noise
