import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3467)
#generate data
x = np.random.normal(0, 10, 150)
x = np.sort(x)
dx = np.transpose(np.array([np.ones((150)), x, x**2], dtype=np.float64))
trueTheta = np.random.uniform(-1.0, 1.0, 3)
print("True Theta:", trueTheta)
y = np.empty((150), dtype=np.float32)
for i in range(150):
    y[i] = trueTheta[0] + trueTheta[1]*x[i] + trueTheta[2]*(x[i]**2)


#regression
theta = np.dot(np.transpose(dx), dx)
theta = np.linalg.inv(theta)
theta = np.dot(theta, np.transpose(dx))
theta = np.dot(theta, np.transpose(y))
print("DX Theta:", theta)

predictedY = np.empty((150), dtype=np.float32)
E = 0.0
for i in range(150):
    predictedY[i] = theta[0] + theta[1]*x[i] + theta[2]*(x[i]**2)
    E += (y[i] - predictedY[i])**2
E = E/2
print("DX Training Error:", E)

# #validaiton
# np.random.seed(1523)
# Valx = np.random.normal(0, 10, 150)
# Valx = np.sort(Valx)

# ValpredictedY = np.empty((150), dtype=np.float32)
# ValTrueY = np.empty((150), dtype=np.float32)
# ValE = 0.0
# for i in range(150):
#     ValTrueY[i] = trueTheta[0] + trueTheta[1]*Valx[i] + trueTheta[2]*(Valx[i]**2)
#     ValpredictedY[i] = theta[0] + theta[1]*Valx[i] + theta[2]*(Valx[i]**2)
#     ValE += (ValTrueY[i] - ValpredictedY[i])**2
# ValE = ValE/2
# print("Validation Error:", ValE)

#GD training
alpha = 1.0e-05
theta = np.array([-0.09, -0.2, 0.6], dtype=np.float64)
print("Starting Theta:",theta)
epoch = 100
Errors = np.zeros((epoch), dtype=np.float64)
for j in range(epoch):
    newpredictedY = np.empty((150), dtype=np.float64)
    E = 0.0
    for i in range(150):
        newpredictedY[i] = theta[0] + theta[1]*x[i] + theta[2]*(x[i]**2)
        E += (y[i] - newpredictedY[i])**2
        theta[0] = theta[0] - alpha*(newpredictedY[i]- y[i])
        theta[1] = theta[1] - alpha*(newpredictedY[i]- y[i])*x[i]
        theta[2] = theta[2] - alpha*(newpredictedY[i]- y[i])*(x[i]**2)
    E = E/2
    Errors[j] = E

Errors = Errors[1:]
print("Final Theta:", theta)

fig, (ax, ax2) = plt.subplots(2, figsize=(7,7))
ax.scatter(x,predictedY, s=70, marker='o', facecolors='none', edgecolors='b', label='Training Data')
#ax2.scatter(Valx,ValpredictedY, s=70, marker='o', facecolors='none', edgecolors='r', label='Validation Data')
#ax.plot(x, trueTheta[0] + trueTheta[1]*x + trueTheta[2]*(x**2), color="green", label='True Curve')
ax.plot(x, theta[0] + theta[1]*x + theta[2]*(x**2), color="red", label='Regression Curve')
#ax2.plot(x, trueTheta[0] + trueTheta[1]*x + trueTheta[2]*(x**2), color="green")

ax2.plot(np.arange(epoch-1), Errors, color="green", label='Error')
fig.legend()
plt.show()