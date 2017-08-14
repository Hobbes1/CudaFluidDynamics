import sys 
import numpy as np

array = np.zeros((256, 3), dtype=np.float)
for i in range(85):
	array[i, 0] = i*(168.0/85)
	array[i, 1] = i*(40.0/85)
	array[i, 2] = i*(15.0/85)
for i in range(85):
	array[i+85, 0] = 168.0 + i*((243.0-168.0)/85.0)
	array[i+85, 1] = 40.0 + i*((194.0-40.0)/85.0)
	array[i+85, 2] = 15.0 + i*((93.0-15.0)/85.0)
for i in range(86):
	array[i+170, 0] = 243.0 + i*((255.0-243.0)/86.0)
	array[i+170, 1] = 194.0 + i*((255.0-194.0)/86.0)
	array[i+170, 2] = 93.0 + i*((255.0-93.0)/86.0)

for i in range(256):
	for j in range(3):
		array[i, j] /= 255.0

np.savetxt("Gwyd_Color_Map", array, fmt='%12.6f',delimiter=' ')


