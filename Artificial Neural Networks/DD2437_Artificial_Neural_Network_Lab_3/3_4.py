import numpy as np
import functions as fun
import load_data as ld
import matplotlib.pyplot as plt

filepath = r"data/pict.dat"
pictures_matrix = ld.get_pictures(filepath)

number_training = 3
training = np.zeros((number_training,pictures_matrix.shape[1]*pictures_matrix.shape[2]))
for i in range(number_training):
    training[i] = np.ravel(pictures_matrix[i])

weights = fun.fit(training)
percentage_threshold = np.zeros(3)
accuracy = np.zeros((3,100))
for i in range(training.shape[0]):
    training_noise = training[i].copy()
    M = training_noise.shape[0]
    np.random.seed(42)
    plt.imshow(training_noise.reshape(32, 32))
    plt.title("Original image")
    plt.show()
    stable_points = 0
    shuffled_indeces = np.arange(M)
    np.random.shuffle(shuffled_indeces)
    for j in range(1,101): # j% noise
        training_noise = training[i].copy()
        percentage = j/100
        noise_magnitude = int(np.floor(percentage*M))
        flipped_indeces = shuffled_indeces[:noise_magnitude]#np.random.choice(M,noise_magnitude,replace=False)
        training_noise[flipped_indeces]=-training_noise[flipped_indeces]
        test_output = fun.predict_little_model(weights, training_noise)
        accuracy[i,j-1] = fun.compute_accuracy(training[i],test_output)
        if (np.array_equal(training[i], test_output)):
            percentage_threshold[i]=percentage
            print('Pattern ' + str(i+1) +' with noise ' + str(percentage) +' to its original pattern')
        else:
            print('Pattern ' + str(i+1) + ' did NOT converged to its original pattern')
            print('Converged attractor: ', test_output)
            #plt.imshow(test_output.reshape(32, 32))
            #plt.title("Erroneous attractor")
            #plt.show()
            #break

print(percentage_threshold)

print(accuracy[0,:])
print(accuracy[1,:])
print(accuracy[2,:])
max_iter=1
max_training_num=[3]
percentage_check = np.arange(1,101, 1)
#accuracy_mean = fun.accuracy_with_sample_images(percentage_check, max_training_num, max_iter, training)
plt.plot(percentage_check, accuracy[0,:],label='p1')
plt.plot(percentage_check, accuracy[1, :],label='p2')
plt.plot(percentage_check, accuracy[2, :],label='p2')
plt.title("Accuracy (pixel %) as a function of noise magnitude")
plt.xlabel("Error %")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
