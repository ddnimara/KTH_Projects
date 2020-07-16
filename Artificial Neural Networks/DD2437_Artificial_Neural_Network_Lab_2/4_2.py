import pandas as pd
import numpy as np
import csv
import part_2_functions as p2f
import matplotlib.pyplot as plt


#Reading data set from .dat file
filepath=r"cities.dat"
data = pd.read_csv(filepath,header=None,sep=',|;')
data = data.to_numpy()
data=data[:,:-1]

#initializing weights
mean = np.mean(data,axis = 0 )
sigma = 0.1
weights = p2f.initialize_weights(data.shape[0],data.shape[1],mean,sigma,True)

#Running algorithm
num_epochs = 20
eta = 0.2
neighbours = 2



for epoch in range(num_epochs):
    
    for i in range(data.shape[0]):
        distances = p2f.distance(data[i,:], weights)
        weights = p2f.update_weights_uniform(weights, distances, data[i,:], eta, neighbours,True)

    if epoch>0.25*num_epochs and epoch<0.5*num_epochs:
        neighbours = 1
    elif epoch>= 0.5*num_epochs:
        neighbours = 0

#Printing data
plt.figure()
plt.scatter(data[:,0],data[:,1],color = 'r')
plt.scatter(weights[:,0],weights[:,1],color = 'b')
plt.show()
    


    
    

        
        
        
            
