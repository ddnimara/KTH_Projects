# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import rbf_network
import data_generation
from sklearn.metrics import mean_absolute_error

##### ANALYSIS #####
seed = 42
np.random.seed(seed)

# Generate data sets
X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1,1)
y_train = data_generation.square(X_train)
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1,1)
y_test = data_generation.square(X_test)

# Initialize variables
n_features = X_train.shape[1]
n_nodes = 50

n_samples = X_train.shape[0]

# Find mean and sigma (std) of the training samples per feature
sigma = 1
width = 0.09
mu = np.linspace(0, 2*np.pi, n_nodes)
# Initialize weights
weights = np.zeros((n_features, n_nodes))

for i in range(n_features):
    for j in range(n_nodes):
        weights[i, j] = np.random.normal(mu[j], sigma)

# Generate the transfer functions
transfer_f_train = rbf_network.gaussian_rbf(X_train, mu, width, n_nodes, n_samples) 
transfer_f_test = rbf_network.gaussian_rbf(X_test, mu, width, n_nodes, n_samples)

# Update weights and analyse the learning curve in each epoch
y_predict = np.zeros(len(y_test))
y_train_predict = np.zeros(len(y_train))


weights = rbf_network.least_squares(transfer_f_train,y_train)
y_predict = rbf_network.approx_function(weights,  transfer_f_test)

absolute_error = mean_absolute_error(y_test,y_predict.reshape(y_predict.shape[1],1))

# Plot final predicted curve X true curve
print(y_predict)
plt.plot(X_test, y_predict.reshape(y_predict.shape[1],1), label = "Predicted")
plt.plot(X_test, y_test, label = "True")
plt.legend()
plt.show()

print('The mean absolute error is: ',absolute_error)
