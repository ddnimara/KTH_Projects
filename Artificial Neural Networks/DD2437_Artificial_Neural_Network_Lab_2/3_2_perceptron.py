import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import data_generation

seed = 42
np.random.seed(seed)

def generate_data(mean, cov, N):
    return np.random.multivariate_normal(mean, cov, N)

def delta_rule(eta, weights, X, target):
    delta_weights = -eta*np.dot((np.dot(weights, X) - target), X.T)
    return delta_weights    
    
def sigmoid_fun(x):
    return 2/(1+np.exp(-x)) - 1

def sigmoid_derivative(x_sig):
    return (1+x_sig)*(1-x_sig)/2

def forward_pass(x,W):
    results=[x]
    for i in range(len(W)):
        
        h =sigmoid_fun(np.dot(W[i],results[-1]))
        # print('h shape',h.shape)
        if(i<len(W)-1):
            h = np.vstack((h,np.ones((1,h.shape[1]))))
            # print('h shape',h.shape)
        results.append(h)
    #o =sigmoid_fun(np.dot(V,h))
    return results


def initialize_ml_weights(x,num_hidden_layers,width_of_h_layers):
    
    weights = []
    weights.append(np.zeros((width_of_h_layers[0], x.shape[0]))) # first weight : features x h[0]
    for i in range(num_hidden_layers-1):
        weights.append(np.zeros((width_of_h_layers[i+1],width_of_h_layers[i]+1))) #general: h[i+1] x h[i]
    weights.append(np.zeros((1, width_of_h_layers[-1]+1))) #at the end: output (=1) x h[final] 
    
    
    for i in range(len(weights)):
        for j in range(weights[i].shape[0]):            
            weights[i][j] = np.random.normal(0,0.1,weights[i].shape[1])
    return weights



def backwards_pass(targets,weights,outputs):
    last_error = np.multiply((outputs[-1] - targets),sigmoid_derivative(outputs[-1]))
    errors = [last_error]
    
    for i in list(reversed(range(1,len(outputs)))):
        # print("iteration i:",i)
        # print("weights:",weights[i-1].shape)
        # print("error first",errors[-1].shape)
        # print("sigmoid",sigmoid_derivative(outputs[i-1]).shape)
        last_error = np.multiply(np.dot(weights[i-1].T,errors[-1]) ,sigmoid_derivative(outputs[i-1]))
        last_error=last_error[:-1]
        errors.append(last_error)
    errors = errors[::-1]
    return errors
   
    
def ml_delta_rule(eta,x,errors,outputs):
    delta_w = []
    for i in range(len(errors)-1):
        er = - eta * np.dot(errors[i+1],outputs[i].T)
        delta_w.append(er)
    return delta_w

def ml_delta_rule_momentum(eta,d_w,alpha,x,errors,outputs):
    delta_w=[]
    for i in range(len(errors)-1):
        change = alpha*d_w[i] - np.dot(errors[i+1],outputs[i].T)*(1-alpha)
        delta_w.append(change)
    for j in range(len(delta_w)):
        delta_w[j] = delta_w[j] * eta
    return delta_w


def train(threshold,epochs,x,targets,h_num,h_width,eta,method='delta'):
    weights = initialize_ml_weights(x,h_num,h_width)
    d_w=[]
    for j in range(len(weights)):
        d_w.append(np.zeros((weights[j].shape[0],weights[j].shape[1])))
        for k in range(d_w[j].shape[0]):
            d_w[j][k] = np.random.normal(0,0.1,d_w[j].shape[1])
    error = 100
    old_error=1
    counter = 0
    while((error > threshold) and (counter< epochs)):
        #for i in range(epochs):
        #print("epoch number",i)
        outputs = forward_pass(x,weights)
        errors = backwards_pass(targets,weights,outputs)
        old_error = compute_Error(outputs[-1],targets)
        # print("Old error",old_error)
        if(method=='delta'):
            d_w = ml_delta_rule(eta,x,errors,outputs)
            for j in range(len(weights)):
                weights[j] = weights[j] + d_w[j]
        else:
            d_w = ml_delta_rule_momentum(eta,d_w,0.9,x,errors,outputs)
            for j in range(len(weights)):
                weights[j] = weights[j] + d_w[j]
        error = compute_Error(forward_pass(x,weights)[-1],targets)
        # print("New error",error)
        counter += 1
    print(counter)
    return weights

def compute_Error(output,target):
    return np.sum(np.square(output-target))/target.shape[0]

#FUNCTION APPROXIMATION

#Creating Gauss Bell function
def bell_function(x,y): 
    output = math.exp(-((pow(x,2)+pow(y,2)))/10)-0.5
    return output



threshold = 0.01
epochs = 50000
X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1,1)
y_train = data_generation.sinusoidal(X_train, noise = True).T
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
y_test = data_generation.sinusoidal(X_test, noise = True).T
X_train = np.vstack((X_train[:].T, np.ones(len(X_train[:]))))
X_test = np.vstack((X_test[:].T, np.ones(len(X_test[:]))))
h_num = 1
h_width = [8]
eta = 0.01

function_weights = train(threshold,epochs,X_train,y_train,h_num,h_width,eta,method='delta')

y_pred = forward_pass(X_test,function_weights)[-1]

# Plot final predicted curve X true curve
plt.plot(X_test[0], y_pred.T, label = "Predicted")
plt.plot(X_test[0], y_test.T, label = "True")
plt.legend()
plt.show()