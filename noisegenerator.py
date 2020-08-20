from perlinnoisemaker import perlinnoise
from dsnoisemaker import diamondSquare
from bmnoisemaker import bmnoise
from lsnoisemaker import lsystem
import numpy as np

class noisegenerator:

    def __init__(self,size, layers,scales,faces=6):
        self.faces = faces
        self.layers = [np.zeros(size)]*self.faces
        for i in range(self.faces):
            weight = np.ones((size))
            for k,x in enumerate(layers):
                if x == "perlin":
                    noise = perlinnoise(size, [1,1])
                    noise = noise / np.linalg.norm(noise)
                    noise = np.multiply(noise, weight)
                    weight = noise

                    self.layers += noise*scales[k]
                if x == "ds":
                    noise = -0.35 + np.random.rand(size[0],size[1])
                    diamondSquare(noise, 50)
                    noise = noise / np.linalg.norm(noise)
                    noise = np.multiply(noise, weight)
                    weight = noise
                    self.layers += noise*scales[k]
                if x == "white":
                    noise = -0.45 + np.random.rand(size[0],size[1])
                    noise = np.multiply(noise, weight)
                    weight = noise
                    self.layers += noise*scales[k]
                if x == "bm":
                    noise = bmnoise(size)
                    noise = noise / np.linalg.norm(noise)
                    noise = np.multiply(noise, weight)
                    weight = noise
                    self.layers += noise*scales[k]
                if x == "ls":
                    noise = -0.3 + lsystem(size)
                    noise = noise / np.linalg.norm(noise)
                    noise = np.multiply(noise, weight)
                    weight = noise
                    self.layers += noise*scales[k]


