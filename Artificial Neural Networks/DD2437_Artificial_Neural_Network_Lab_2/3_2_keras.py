#Import libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from tqdm import tqdm
import data_generation

# Generate data sets
X_train = np.arange(0, 2*np.pi, 0.01).reshape(-1,1)
Y_train = data_generation.sinusoidal(X_train, noise = False)
X_val = np.arange(0.1, 2*np.pi, 0.05).reshape(-1,1)
Y_val = data_generation.sinusoidal(X_val, noise = False)
X_test = np.arange(0.05, 2*np.pi, 0.05).reshape(-1,1)
Y_test = data_generation.sinusoidal(X_test, noise = False)

# Initialize variables for the iterations
h_layers_width = [50] 
regularization_factor = [1e-3]
iterations = 1
epochs = 10000
predictions = np.zeros((len(h_layers_width), len(regularization_factor), iterations, X_test.shape[0]))
history_loss = np.zeros((len(h_layers_width), len(regularization_factor), iterations, epochs))
history_val_loss = np.zeros((len(h_layers_width), len(regularization_factor), iterations, epochs))
test_loss = np.zeros((len(h_layers_width), len(regularization_factor), iterations))
predictions_std = np.zeros((len(h_layers_width), len(regularization_factor), X_test.shape[0]))
predictions_mean = np.zeros((len(h_layers_width), len(regularization_factor), X_test.shape[0]))
loss_std = np.zeros((len(h_layers_width), len(regularization_factor), epochs))
loss_mean = np.zeros((len(h_layers_width), len(regularization_factor), epochs))
val_loss_std = np.zeros((len(h_layers_width), len(regularization_factor), epochs))
val_loss_mean = np.zeros((len(h_layers_width), len(regularization_factor), epochs))

for layer in tqdm(range(len(h_layers_width))): # Try different hidden layers
    for factor in tqdm(range(len(regularization_factor))): # Try different factor
        for i in tqdm(range(iterations)): # Iterate 10 times to take the average of the models
            # Set seed for each model accordingly to the iteration
            np.random.seed(i+2)
            
            # Generate model, fit and evaluate
            model = keras.Sequential() # Build Neural Network Layers
            #model.add(keras.layers.Dense(units = 8, activation = 'relu', input_dim = 5, kernel_regularizer = regularizers.l2(1e-3)))
            model.add(keras.layers.Dense(units = h_layers_width[layer], activation = 'relu', input_dim = 1, kernel_regularizer = regularizers.l2(regularization_factor[factor]))) # Use regularization in Hidden Layers (e.g. kernel_regularizer = regularizers.l1(0.1))
            model.add(keras.layers.Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam') # Define Neural Networking Learning Criteria and Method 
            #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) # Define Early Stopping (convergency criteria)
        
            # Fit model with defined parameters
            model.fit(X_train, Y_train, validation_data = (X_val,Y_val), epochs = epochs, verbose = 0)

            # Append evaluation, predictions, and validation errors
            test_loss[layer, factor, i] = (model.evaluate(X_test, Y_test))
            predictions[layer, factor, i] = (model.predict(X_test)).ravel()
            history_loss[layer, factor, i] = (model.history.history['loss'])
            history_val_loss[layer, factor, i] = (model.history.history['val_loss'])

        # Compute Std and Mean prediction and errors for each model
        predictions_std[layer, factor] = np.std(predictions[layer, factor], axis = 0)
        predictions_mean[layer, factor] = np.mean(predictions[layer, factor], axis = 0)
        loss_std[layer, factor] = np.std(history_loss[layer, factor], axis = 0)
        loss_mean[layer, factor] = np.mean(history_loss[layer, factor], axis = 0)
        val_loss_std[layer, factor] = np.std(history_val_loss[layer, factor], axis = 0)
        val_loss_mean[layer, factor] = np.mean(history_val_loss[layer, factor], axis = 0)

# Find best model
min_matrix = np.min(val_loss_mean, axis = 2)
min_index = np.argmin(min_matrix)
row = min_index//min_matrix.shape[1]
column = min_index%min_matrix.shape[1]

# Plot predictions
plt.plot(X_train, Y_train, label = 'Train Inputs')
plt.plot(X_test, predictions_mean[row, column], label = 'Predicted outputs')
plt.plot(X_test, Y_test, label = 'True testing outputs')
plt.xlim(0,6.2)
plt.fill_between(np.arange(0,predictions_mean[row, column].shape[0]),
                (predictions_mean[row, column] - predictions_std[row, column]).ravel(),
                (predictions_mean[row, column] + predictions_std[row, column]).ravel(),
                alpha=0.5,
                linestyle = '-' 
                )
plt.title("Prediction Curve X True Curve (width "+ str(h_layers_width[row]) + ", regularization factor "+ str(regularization_factor[column]) + ")")
plt.legend()
plt.show()

# Plot training & validation loss values
indexes = [0,30,60,90,120,150,180,210,240,270,299]
yerr_loss = np.zeros(loss_mean[row, column].shape[0])
yerr_val_loss = np.zeros(val_loss_mean[row, column].shape[0])
yerr_loss[indexes] = loss_std[row, column, indexes]
yerr_val_loss[indexes] = val_loss_std[row, column, indexes]
plt.errorbar(np.arange(0,loss_mean[row, column].shape[0]), loss_mean[row, column],  yerr = yerr_loss)
plt.errorbar(np.arange(0,val_loss_mean[row, column].shape[0]), val_loss_mean[row, column],  yerr = yerr_val_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


