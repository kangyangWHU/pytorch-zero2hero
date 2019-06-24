import numpy as np
import matplotlib.pyplot as plt
import time

def compute_error(b, w, points):
    '''
        compute the mse error of the points 
    '''
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += ( (w*x+b) - y)**2
    return total_error/float(len(points))

def step_gradient(b_current, w_current, points, learning_rate):
    '''
        single step gradient descent
    '''
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient = (2/N)*((w_current*x+b_current)-y)
        w_gradient = (2/N)*((w_current*x+b_current)-y)*x

    new_b = b_current - learning_rate*b_gradient        # update the b
    new_w = w_current - learning_rate*w_gradient        # update w

    return [new_b, new_w]


def gradient_descent(points, init_b, init_w, learning_rate, num_iters):
    '''
        perform gradient descent 
    '''
    b = init_b
    w = init_w

    plt.ion() # 开启interactive mode 成功的关键函数
    plt.figure()

    for i in range(num_iters):

        b,w = step_gradient(b, w, points, learning_rate)
        loss = compute_error(b, w, points)

        # plot the figure dynamicly
        plt.clf()               # clear the figure
        plt.axis([0,100,0,140])
        plt.scatter(points[:,0], points[:,1])
        x = np.arange(0,100,2)
        y = w*x+b
        plt.plot(x,y)
        plt.draw()              # after clear the figure, you must call the draw function
        plt.show()
        plt.pause(0.001)

    return [b,w]

def run():
    print()
    points = np.genfromtxt('data.csv', delimiter=',')


    learning_rate = 1e-4
    init_b = 0
    init_w = 0
    num_iters = 1000

    gradient_descent(points, init_b, init_w, learning_rate, num_iters)


if __name__ == '__main__':
    run()
