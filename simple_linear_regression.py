import numpy as np

def compute_error(b, w, points):
    '''
        compute the mse error of the points 
    '''
    total_error = 0
    for i in range(len(points))
        x = points[i, 0]
        y = points[i, 1]
        total_error += ( (w*x+b) - y)**2
    return total_error/float(len(points))

def step_gra
