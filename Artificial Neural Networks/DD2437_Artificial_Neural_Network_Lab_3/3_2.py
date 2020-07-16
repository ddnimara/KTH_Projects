#%%
import load_data as ld
import functions as fun
import numpy as np
import matplotlib.pyplot as plt
filepath = r"data/pict.dat"

pictures_matrix = ld.get_pictures(filepath)
# for i in range(pictures_matrix.shape[0]):
#     plt.imshow(pictures_matrix[i])
#     plt.show()

number_training = 3
training = np.zeros((number_training,pictures_matrix.shape[1]*pictures_matrix.shape[2]))
for i in range(number_training):
    training[i] = np.ravel(pictures_matrix[i])

weights = fun.fit(training)
#%%

#Checking stability of the three training patterns
for i in range(training.shape[0]):
    test_output = fun.predict(weights,training[i],100)
    if (np.array_equal(training[i],test_output)):
        print('Training pattern '+str(i)+" is stable.")
    else:
        print('Training pattern '+str(i)+" is NOT stable.")
    print('-------------')

#%%
#Trying degraded pattern 10
test_output = fun.predict(weights,pictures_matrix[9].reshape(-1))
if (np.array_equal(training[0],test_output)):
        print('Pattern 10 converged to pattern 1')
else:
    print('Pattern 10 did NOT converged to pattern 1')
    print('Converged attractor: ',test_output)

plt.imshow(test_output.reshape(32,32))
plt.title('Obtained output pattern')
plt.show()
plt.imshow(pictures_matrix[9])
plt.title('Original distorted pattern')
plt.show()
plt.imshow(pictures_matrix[0])
plt.title('Goal pattern')
plt.show()

#%%
#Trying degraded pattern 11
test_output = fun.predict(weights,np.ravel(pictures_matrix[10]))
if (np.array_equal(training[1],test_output)):
    print('Pattern 11 converged to pattern 2')
elif (np.array_equal(training[2],test_output)):
    print('Pattern 11 converged to pattern 3')
    
else:
    print('Pattern 11 converged to a different attractor.')
    print('Converged attractor: ',test_output)

plt.imshow(test_output.reshape(32,32))
plt.title('Obtained output pattern')
plt.show()
plt.imshow(pictures_matrix[10])
plt.title('Original distorted pattern')
plt.show()
plt.imshow(pictures_matrix[1])
plt.show()
plt.imshow(pictures_matrix[2])
plt.show()

#%%
#Selecting random units and checking convergence (question 3 of 3.2)
np.random.seed(42)
number_units = np.arange(weights.shape[0])
number_units = np.random.permutation(number_units)
convergence = False
i = 0
test = pictures_matrix[9].reshape(-1).copy()
previous_iter_100 = pictures_matrix[9].reshape(-1).copy()
for it in range(10):
    for i in range(len(number_units)):
        previous_test = test.copy()
        update_bit = fun.sign(np.dot(weights[number_units[i]],test))
        test[number_units[i]] = update_bit
        if (i%100 ==0) and (i!=0):
            if (np.array_equal(previous_iter_100,test)):
                number_iterations = it+1
                break
            plt.imshow(test.reshape(32,32))
            plt.title('Print after '+str(int(i+(it*len(number_units))))+' updates.')
            plt.show()
            previous_iter_100 = test.copy()
    if (np.array_equal(previous_iter_100,test)):
        break
    
plt.imshow(test.reshape(32,32))
plt.title("Final Plot. Number of iterations: "+str(number_iterations))
plt.show()



#%%
#Selecting random units and checking convergence (question 3 of 3.2)
np.random.seed(42)
number_units = np.arange(weights.shape[0])
number_units = np.random.permutation(number_units)
convergence = False
i = 0
test = pictures_matrix[10].reshape(-1).copy()
previous_iter_100 = pictures_matrix[10].reshape(-1).copy()
for it in range(10):
    for i in range(len(number_units)):
        previous_test = test.copy()
        update_bit = fun.sign(np.dot(weights[number_units[i]],test))
        test[number_units[i]] = update_bit
        if (i%100 ==0) and (i!=0):
            if (np.array_equal(previous_iter_100,test)):
                number_iterations = it+1
                break
            plt.imshow(test.reshape(32,32))
            plt.title('Print after '+str(int(i+(it*len(number_units))))+' updates.')
            plt.show()
            previous_iter_100 = test.copy()
    if (np.array_equal(previous_iter_100,test)):
        break
    
plt.imshow(test.reshape(32,32))
plt.title("Final Plot. Number of iterations: "+str(number_iterations))
plt.show()
#%%


