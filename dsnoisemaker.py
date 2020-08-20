import numpy as np

def diamondSquare(array,size):
  
  half = size//2
  if half<1:
    return
  for z in range(half, array.shape[1], size):
    for x in range(half, array.shape[0], size):
      squareStep(array, x % array.shape[0], z % array.shape[1], half)
  col = 0
  for x in range(0,array.shape[0],half):
    col += 1
    if col % 2 == 1:
      for z in range(half, array.shape[1],size):
        diamondStep(array, x % array.shape[0], z % array.shape[1], half)
    else:
      for z in range(0, array.shape[1],size):
        diamondStep(array, x % array.shape[0], z % array.shape[1], half)
  diamondSquare(array,size//2)
  
def squareStep(array, x, z, reach):
  count = 0
  avg = 0.0
  if x - reach >= 0 and z - reach >= 0:
    avg += array[x-reach,z-reach]
    count += 1
  if x - reach >= 0 and z + reach < array.shape[1]:
    avg += array[x-reach,z+reach]
    count += 1
  if x + reach < array.shape[0] and z - reach >= 0:
    avg += array[x+reach,z-reach]
    count += 1 
  if x + reach < array.shape[0] and z + reach < array.shape[1]:
    avg += array[x+reach,z+reach]
    count += 1 
  avg += random(reach)
  avg /= count
  array[x,z]=round(avg)

def diamondStep(array, x, z, reach):
  count = 0
  avg = 0.0
  if x - reach >= 0:
    avg += array[x-reach,z]
    count += 1
  if x + reach < array.shape[0]:
    avg += array[x+reach,z]
    count += 1
  if z - reach >= 0:
    avg += array[x,z-reach]
    count += 1 
  if z + reach < array.shape[1]:
    avg += array[x,z+reach]
    count += 1 
  avg += random(reach)
  avg /= count
  array[x,z]=round(avg)
  
def random(range_):
  return (np.random.randint(-range_,range_))