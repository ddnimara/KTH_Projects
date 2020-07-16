# Import libraries
import numpy as np

##### COMPUTE METHODS ######

def gaussian_rbf(x, mu, sigma, n_nodes, n_samples):
    transfor_f = np.zeros((n_samples, n_nodes))
    for i in range(n_samples):
        for j in range(n_nodes):
            transfor_f[i,j] = np.exp(- pow((x[i] - mu[j]), 2) / (2 * pow(sigma, 2)))
    return transfor_f

def approx_function(weight, transfor_f):
    return np.dot(weight, transfor_f.T)

def error_function_minimizer(target, weight, transfor_f, n_samples, n_nodes, eta):
    error = approx_function(weight, transfor_f.reshape(1,-1)).T - target
    delta_weights = - eta * np.dot(error, transfor_f.reshape(1,-1))
    return delta_weights

def least_squares(phi,target):
    w = np.dot(np.linalg.inv(np.dot(phi.T,phi)),np.dot(phi.T,target))
    return w.reshape(1,w.shape[0])

def unshuffling(Z,indexes):
    newZ = np.zeros(len(indexes))
    for i in range(len(indexes)):
        newZ[indexes[i]] = Z[0,i]
        
    return newZ

def predict_values(y_test,y_train,weights,transfer_f_test,transfer_f_train):
    y_predict = np.zeros(len(y_test))
    y_train_predict = np.zeros(len(y_train))
    for i in range(len(y_predict)):
        y_predict[i] = approx_function(weights,  transfer_f_test[i])
    for i in range(len(y_train_predict)):
        y_train_predict[i] = approx_function(weights,  transfer_f_train[i])
        
    return y_train_predict,y_predict

def competitive_learning(rand_sample, mu, eta, update_index, method = "normal"):
    norm = np.zeros(len(mu)) 
    for i in range(len(norm)):
        norm[i] = np.linalg.norm(rand_sample-mu[i])
    mu[np.argmin(norm)] += eta*(rand_sample-mu[np.argmin(norm)])
    update_index.append(np.argmin(norm))
    if (method == "leaky_learning"):
        mu[np.argmax(norm)] += (eta**2)*(rand_sample-mu[np.argmax(norm)])
        update_index.append(np.argmax(norm))
    return mu, update_index





