import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch

# define a 3 dimensional function, it has 4 loal minima 
# f(3,2)=f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.584428, -1.848126) = 0
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)

X,Y = np.meshgrid(x, y)     # use this in 3 dimension,important
Z = himmelblau([X, Y])

plt.ion()

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')

ax.axis([-6,6, -6,6])       # fix the axis 
ax.set_zlim([0, 2700])
ax.plot_wireframe(X, Y, Z)  # just draw the fraw, otherwise you won't see the points

x = torch.tensor([-6.,6.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)

iters = 10000
for step in range(iters):
    pred = himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    # in order to recorde video 
    if step == 1:
        plt.pause(10)
    # add the points dynamicly
    if step % 200 == 0:
        ax.scatter(x.tolist()[0], x.tolist()[1], pred.item(),s=15, c='r')
        plt.show()
        plt.pause(0.1)

    # 
    if step == iters-1:
        print('final x = {}, f(x) = {}'.format(x.tolist(), pred.item()))
