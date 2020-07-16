# Import Libraries
import numpy as np

###### BUILD DATA SETS #####

def sinusoidal(x,noise = False):
    data = np.sin(2*x)
    if noise:
        noise = np.random.normal(0,0.1,data.shape)
        data = np.add(data,noise)   
    return data

def square(x,noise = False):
    data = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        if np.sin(2*x[i]) >= 0:
            data[i] = 1
        else:
            data[i] = -1       
    if noise:
        noise = np.random.normal(0,0.1,data.shape)
        data = np.add(data,noise)     
    return data
