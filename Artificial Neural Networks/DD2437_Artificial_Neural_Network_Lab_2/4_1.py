import pandas as pd
import numpy as np
import csv
import part_2_functions as p2f


#Reading data set from .dat file
filepath=r"C:\Users\Adrian\OneDrive - Universidad Carlos III de Madrid\UNIVERSIDAD\KTH - Master Machine Learning\3rd Period\ANN\LABS\Lab2\CODE\data_lab2\animals.dat"
data = pd.read_csv(filepath,header=None)
data = data.to_numpy()
data = data.reshape(32,84)
filepath = r"C:\Users\Adrian\OneDrive - Universidad Carlos III de Madrid\UNIVERSIDAD\KTH - Master Machine Learning\3rd Period\ANN\LABS\Lab2\CODE\data_lab2\animalnames.txt"
names = pd.read_csv(filepath,header=None)
names = p2f.clean_data(names)



#initializing weights
weights = p2f.initialize_weights(100,84)

#Running algorithm
num_epochs = 20
eta = 0.2
neighbours = 20

for epoch in range(num_epochs):
    
    for i in range(data.shape[0]):
        distances = p2f.distance(data[i,:], weights)
        weights = p2f.update_weights_uniform(weights, distances, data[i,:], eta, neighbours)
    
    neighbours -=1

#Finding closest node
final_node = np.zeros((data.shape[0])) 
for i in range(data.shape[0]):
    distances = p2f.distance(data[i,:], weights)
    final_node[i] = np.argmin(distances)
    

p2f.print_min(final_node, names)
    


    
    

        
        
        
            
