import numpy as np
import pandas as pd


def get_data(data='Training'):
    if (data=='Training'):
        x1 = np.array([-1,-1,1,-1,1,-1,-1,1])
        x2 = np.array([-1,-1,-1,-1,-1,1,-1,-1])
        x3 = np.array([-1,1,1,-1,-1,1,-1,1])

        x_res = np.vstack((x1,x2,x3))
    elif(data=='Testing'):
        x1d=np.array([1,-1,1,-1,1,-1,-1,1])
        x2d=np.array([1,1,-1,-1,-1,1,-1,-1])
        x3d=np.array([1,1,1,-1,1,1,-1,1])
        x_res = np.vstack((x1d,x2d,x3d))
    elif(data=='Dissimilar'):
        x_res = np.array([1, 1, -1, 1, -1, -1, 1, 1])
    return x_res

def get_pictures(filepath):
    pictures = pd.read_csv(filepath,header=None,sep=',')
    pictures = pictures.to_numpy()
    pictures_matrix = pictures.reshape(11,32,32)    
    return pictures_matrix
