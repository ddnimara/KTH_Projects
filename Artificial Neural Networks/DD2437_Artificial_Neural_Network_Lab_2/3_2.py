# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import rbf_network
import data_generation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
##### ANALYSIS #####

# Generate data sets
X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1,1)
y_train = data_generation.sinusoidal(X_train, noise = True)
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
y_test = data_generation.sinusoidal(X_test, noise = True)

# Initialize variables
eta = 0.001
n_features = X_train.shape[1]
n_nodes = 8
n_samples = X_train.shape[0]
epochs = 1000

# Find mean and sigma (std) of the training samples per feature
sigma = 1
width = 0.05
mu = np.linspace(0, 2*np.pi, n_nodes)
# Initialize weights
weights = np.zeros((n_features, n_nodes))

for i in range(n_features):
    for j in range(n_nodes):
        weights[i, j] = np.random.normal(mu[j], sigma)


# Update weights and analyse the learning curve in each epoch
mse_test = np.zeros(epochs)
mse_train = np.zeros(epochs)
mse_test_abs = np.zeros(epochs)
mse_train_abs = np.zeros(epochs)
for epoch in range(epochs):
    #Shuffling data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    new_x_train = X_train[indices,:]
    new_y_train = y_train[indices,:]
    
    # Generate the transfer functions
    transfer_f_train = rbf_network.gaussian_rbf(new_x_train, mu, sigma, n_nodes, n_samples)
    transfer_f_test = rbf_network.gaussian_rbf(X_test, mu, sigma, n_nodes, n_samples)
    
    #Apply sequential learning (delta rule)
    for i in range(new_x_train.shape[0]):
        d_weights = rbf_network.error_function_minimizer(new_y_train[i,:], weights, transfer_f_train[i,:], n_samples, n_nodes, eta)
        weights += d_weights
    
    # Predict values
    y_train_predict,y_predict = rbf_network.predict_values(y_test,new_y_train,weights,transfer_f_test,transfer_f_train)   
    
    mse_test[epoch] = mean_squared_error(y_predict, y_test)
    mse_train[epoch] = mean_squared_error(y_train_predict, new_y_train)
    mse_test_abs[epoch] = mean_absolute_error(y_predict, y_test)
    mse_train_abs[epoch] = mean_absolute_error(y_train_predict, new_y_train)


y_train_predict,y_predict = rbf_network.predict_values(y_test,y_train,weights,transfer_f_test,transfer_f_train)
# Plot final predicted curve X true curve
plt.plot(X_test, y_predict, label = "Predicted")
plt.plot(X_test, y_test, label = "True")
plt.legend()
plt.show()

# Plot learning Curve
plt.plot(range(epochs), mse_test, label = "Validation Curve")
plt.plot(range(epochs), mse_train, label = "Learning Curve")
plt.plot(range(epochs), mse_test_abs, label = "Validation Curve Absolute")
plt.plot(range(epochs), mse_train_abs, label = "Learning Curve Absolute")
plt.legend()
plt.show()
print(mse_test[-1])
print(mse_test_abs[-1])