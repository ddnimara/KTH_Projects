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
number_of_distorted = 2
training = np.zeros((number_training,pictures_matrix.shape[1]*pictures_matrix.shape[2]))
for i in range(number_training):
    training[i] = np.ravel(pictures_matrix[i])

weights = fun.fit(training)

#Taking distorted patterns
distorted = np.zeros((number_of_distorted,pictures_matrix.shape[1]*pictures_matrix.shape[2]))
distorted[0] = np.ravel(pictures_matrix[9])
distorted[1] = np.ravel(pictures_matrix[10])

#Energy at attractors
print('Energy of Attractors')
for i in range(number_training):
    energy = fun.compute_lyapunov_energy(training[i].reshape(1,-1), weights)
    print(energy)
print('-----------')


#Energy at distorted patterns
print('Energy of distorted patterns')
for i in range(number_of_distorted):
    energy = fun.compute_lyapunov_energy(distorted[i].reshape(1,-1), weights)
    print(energy)
print('-----------')

#Follow energy changing while converging to an attractor
test_output,energy,energy_step = fun.predict(weights,np.ravel(pictures_matrix[0]),100,True)
plt.figure()
plt.plot(np.arange(len(energy)),energy)
plt.title('Energy change during approach to the attractor (pattern 1).')
plt.ylabel('Energy')
plt.show()

plt.figure()
plt.plot(np.arange(len(energy_step)),energy_step)
plt.title('Energy change during approach to the attractor (pattern 1).')
plt.ylabel('Energy')
plt.xlabel('Iterations')
plt.show()

test_output,energy,energy_step = fun.predict(weights,np.ravel(pictures_matrix[9]),100,True)
plt.figure()
plt.plot(np.arange(len(energy)),energy)
plt.title('Energy change during approach to the attractor (pattern 9).')
plt.ylabel('Energy')
plt.show()

plt.figure()
plt.plot(np.arange(len(energy_step)),energy_step)
plt.title('Energy change during approach to the attractor (pattern 9).')
plt.ylabel('Energy')
plt.xlabel('Iterations')
plt.show()

test_output,energy,energy_step = fun.predict(weights,np.ravel(pictures_matrix[10]),100,True)
plt.figure()
plt.plot(np.arange(len(energy)),energy)
plt.title('Energy change during approach to the attractor (pattern 11).')
plt.ylabel('Energy')
plt.show()

plt.figure()
plt.plot(np.arange(len(energy_step)),energy_step)
plt.title('Energy change during approach to the attractor (pattern 11).')
plt.ylabel('Energy')
plt.xlabel('Iterations')
plt.show()



#Generate a weight matrix with random values from G. distribution and iterate an arbitrary state
weights_random = np.random.normal(0,0.5,(weights.shape[0],weights.shape[1]))
figure_size = int(np.sqrt(len(test_output)))

test_output = fun.predict(weights_random,np.ravel(pictures_matrix[0]))
plt.figure()
plt.imshow(pictures_matrix[0])
plt.title('Original image. Pattern 1')
plt.show()

plt.figure()
plt.imshow(test_output.reshape(figure_size,figure_size))
plt.title('Image obtained after convergence.\n Pattern 1 (with randomly initialized W)')
plt.show()

test_output = fun.predict(weights_random,np.ravel(pictures_matrix[9]))
plt.figure()
plt.imshow(pictures_matrix[9])
plt.title('Original image. Pattern 10')
plt.show()

plt.figure()
plt.imshow(test_output.reshape(figure_size,figure_size))
plt.title('Image obtained after convergence.\n Pattern 10 (with randomly initialized W)')
plt.show()

test_output = fun.predict(weights_random,np.ravel(pictures_matrix[10]))
plt.figure()
plt.imshow(pictures_matrix[10])
plt.title('Original image. Pattern 11')
plt.show()

plt.figure()
plt.imshow(test_output.reshape(figure_size,figure_size))
plt.title('Image obtained after convergence.\n Pattern 11 (with randomly initialized W)')
plt.show()


#Making the weight matrix symmetric
weights_symmetric = 0.5*np.add(weights_random,weights_random.T)

test_output = fun.predict(weights_symmetric,np.ravel(pictures_matrix[0]))
plt.figure()
plt.imshow(pictures_matrix[0])
plt.title('Original image. Pattern 1')
plt.show()

plt.figure()
plt.imshow(test_output.reshape(figure_size,figure_size))
plt.title('Image obtained after convergence.\n Pattern 1 (with randomly initialized and symmetric W)')
plt.show()

test_output = fun.predict(weights_symmetric,np.ravel(pictures_matrix[9]))
plt.figure()
plt.imshow(pictures_matrix[9])
plt.title('Original image. Pattern 10')
plt.show()

plt.figure()
plt.imshow(test_output.reshape(figure_size,figure_size))
plt.title('Image obtained after convergence.\n Pattern 10 (with randomly initialized and symmetric W)')
plt.show()

test_output = fun.predict(weights_symmetric,np.ravel(pictures_matrix[10]))
plt.figure()
plt.imshow(pictures_matrix[10])
plt.title('Original image. Pattern 11')
plt.show()

plt.figure()
plt.imshow(test_output.reshape(figure_size,figure_size))
plt.title('Image obtained after convergence.\n Pattern 11 (with randomly initialized and symmetric W)')
plt.show()










