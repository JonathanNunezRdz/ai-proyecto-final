import pandas as pd
import math
import numpy as np

x = np.array([[158,58],[158,59],[158,63],[160,59],[160,60],[163,60],[163,61],[160,64],[163,64],[165,61],[165,62],[165,65],[168,62],[168,63],[168,66],[170,63],[170,64],[170,68]], dtype=float)
x_test = np.array([[164,60]], dtype=float)

# mean = np.array([[0,0]])
# for i in range(x_scaled.shape[0]):
#     for j in range(x_scaled.shape[1]):
#         mean[0,i] += x_scaled[i,j]
#         x_scaled[i][j] = (x[i][j] - x[i].mean()) / x[i].std()

x_scaled = np.zeros_like(x)
mean = np.zeros(shape = (x_scaled.shape[1], 1))
deviation = np.zeros(shape = (x_scaled.shape[1], 1))

for i in range(x.shape[1]):
    col = np.zeros_like(x)
    for j in range(x.shape[0]):
        col[j] = x[j][i]    
    mean[i] = col.mean()
    deviation[i] = col.std()
    for j in range(x.shape[0]):
        x_scaled[j][i] = (x[j][i] - col.mean()) / col.std()


for i in range(x_test.shape[1]):
    for j in range(x_test.shape[0]):
        x_test[j][i] = (x_test[j][i] - mean[i]) / deviation[i]

zeros = np.zeros([x_scaled.shape[0],1])

for i in range(x_scaled.shape[0]):
    for j in range(x_scaled.shape[1]):
       zeros[i,0] += (x_scaled[i,j] - x_test[0,j])**2
    zeros[i,0] = np.sqrt(zeros[i,0])

x_scaled = np.concatenate((x_scaled, zeros) ,axis=1)

y = np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]])
y = y.T

x_scaled = np.concatenate((x_scaled, y) ,axis=1)
x_scaled = x_scaled[x_scaled[:,-2].argsort()]

print(x_scaled[0:5,:])

# print(x)
# print(x_scaled)

# matrix = np.zeros([2,1])
# matrix[0] = 1
# matrix[1] = 2
# matrix = np.append(matrix, [[3]], axis=0)
# print(matrix)

# x = np.array([[159,58]])
# print(x.mean(), x.std())