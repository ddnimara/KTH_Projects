# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import rbf_network
import data_generation

##### ANALYSIS #####
seed = 42
np.random.seed(seed)

# Generate data sets
X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1,1)
y_train = data_generation.sinoidal(X_train)
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
y_test = data_generation.sinoidal(X_test)

# Initialize variables
eta = 0.001
n_features = X_train.shape[1]
n_nodes = 64
n_samples = X_train.shape[0]
epochs = 1000

# Find mean and sigma (std) of the training samples per feature
sigma = 1
mu = np.linspace(0, 2*np.pi, n_nodes)

# Initialize weights
weights = np.zeros((n_features, n_nodes))
for i in range(n_features):
    for j in range(n_nodes):
        weights[i, j] = np.random.normal(mu[j], sigma)

# Generate the transfer functions
transfer_f_train = rbf_network.gaussian_rbf(X_train, mu, sigma, n_nodes, n_samples)
transfer_f_test = rbf_network.gaussian_rbf(X_test, mu, sigma, n_nodes, n_samples)

# Update weights and analyse the learning curve in each epoch
y_predict = np.zeros(len(y_test))
y_train_predict = np.zeros(len(y_train))
mse_test = np.zeros(epochs)
mse_train = np.zeros(epochs)
for epoch in range(epochs):
    
    d_weights = rbf_network.error_function_minimizer(y_train, weights, transfer_f_train, n_samples, n_nodes, eta)
    weights += d_weights
    # Predict values
    for i in range(len(y_predict)):
        y_predict[i] = rbf_network.approx_function(weights,  transfer_f_test[i])
    for i in range(len(y_train_predict)):
        y_train_predict[i] = rbf_network.approx_function(weights,  transfer_f_train[i])
    mse_test[epoch] = np.mean((np.square(pow(y_predict - y_test, 2))))
    mse_train[epoch] = np.mean((np.square(pow(y_train_predict - y_train, 2))))


# Plot final predicted curve X true curve
print(y_predict)
plt.plot(X_test, y_predict, label = "Predicted")
plt.plot(X_test, y_test, label = "True")
plt.legend()
plt.show()

# Plot learning Curve
plt.plot(range(epochs), mse_test, label = "Validation Curve")
plt.plot(range(epochs), mse_train, label = "Learning Curve")
plt.legend()
plt.show()