import numpy as np
import pandas as pd


def euc_distance(x,w):
    subtraction = np.subtract(x,w)
    d = np.dot(subtraction.T,subtraction)
    return d

def distance(data,weights):
    distances = np.zeros((weights.shape[0]))
    for i in range(weights.shape[0]):
        distances[i] = euc_distance(data, weights[i,:])
        
    return distances


def initialize_weights(nRows,nColumns,mean = 0,sigma = 1,circular = False):
    weights = np.zeros((nRows,nColumns)) #dimension of weights: number of nodes X number of features
    if circular:
        weights = np.random.multivariate_normal(mean,sigma*np.identity(len(mean)),nRows)
        
    else:
        
        aux=np.arange(0,1,0.01)
        for i in range(nRows):
            weights[i,:] = aux[i]
    
    return weights

def update_weights(weights,distances,data_point,eta,neighbours,circular = False):
    maximum_node = np.argmin(distances)
    
    if circular:
        
        for i in range(maximum_node-neighbours,maximum_node+neighbours+1):
            proportion = 2**(int(-abs(maximum_node-i)))
            weights[i%len(weights.shape[0]),:] = eta*np.add(weights[i%len(weights.shape[0]),:],np.subtract(data_point,weights[i%len(weights.shape[0]),:]))*proportion
        
    else:
        if neighbours <= 0:
             weights[maximum_node,:] = eta*np.add(weights[maximum_node,:],np.subtract(data_point,weights[maximum_node,:]))
             return weights
        else:
            top_neighbour = maximum_node+neighbours+1
            bottom_neighbour = maximum_node-neighbours
            
            if top_neighbour>weights.shape[0]:
                top_neighbour = weights.shape[0]
                
            if bottom_neighbour<0:
                bottom_neighbour = 0
            
            
            for i in range(bottom_neighbour,top_neighbour):
                proportion = 2**(int(-abs(maximum_node-i)))
                weights[i,:] = eta*np.add(weights[i,:],np.subtract(data_point,weights[i,:]))*proportion
            
            
    return weights
    
def update_weights_uniform(weights,distances,data_point,eta,neighbours,circular = False):
    maximum_node = np.argmin(distances)
    if circular:
        for i in range(maximum_node-neighbours,maximum_node+neighbours+1):
            weights[i%(weights.shape[0]),:] = eta*np.add(weights[i%(weights.shape[0]),:],np.subtract(data_point,weights[i%(weights.shape[0]),:]))
    
    else: 
        
        if neighbours <= 0:
             weights[maximum_node,:] = eta*np.add(weights[maximum_node,:],np.subtract(data_point,weights[maximum_node,:]))
             return weights
        else:
            top_neighbour = maximum_node+neighbours+1
            bottom_neighbour = maximum_node-neighbours
            
            if top_neighbour>weights.shape[0]:
                top_neighbour = weights.shape[0]
                
            if bottom_neighbour<0:
                bottom_neighbour = 0
            
            
            for i in range(bottom_neighbour,top_neighbour):
                weights[i,:] = eta*np.add(weights[i,:],np.subtract(data_point,weights[i,:]))
            
                
    return weights

def print_min(final_node,names):
    copy = final_node.copy()
    order = np.arange(len(final_node))
    for i in range(len(final_node)):
        min_position = np.argmin(copy)
        min_node = min(copy)
        print('The node of animal '+str(names[min_position])+' is '+str(int(min_node)))
        copy[min_position] = 1000
        
def clean_data(data):
    data = pd.DataFrame(data).iloc[:,0].tolist()
    for i in range(len(data)):
        data[i]=data[i].replace("\t","")
        data[i]=data[i].replace("\'","")
        data[i]=data[i].replace("[","")
        data[i]=data[i].replace("]","")
        data[i]=data[i].replace(";","")
        
        
    return data
        
    