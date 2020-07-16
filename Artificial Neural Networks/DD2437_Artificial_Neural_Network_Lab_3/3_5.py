# Import libraries
import numpy as np
import functions as fun
import load_data as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Load data
filepath = r"data/pict.dat"
pictures_matrix = ld.get_pictures(filepath)

#3_5 first bullet
#fun.three_point_five_first_bullet()

#3_5 second bullet
#fun.three_point_five_second_bullet()

#3_5 fourth bullet
#fun.three_point_five_fourth_bullet()

#3_5 fifth bullet
#fun.three_point_five_fifth_bullet()

#3_5 sixth bullet
fun.three_point_five_sixth_bullet()

# Learning


'''
number_of_random_samples = 300
max_training_num = np.arange(1,number_of_random_samples+1,1)#[3,4, 5, 6, 7]

np.random.seed(42)

max_iter = 1

percentage_check = np.arange(0,55,5)
training = np.zeros((number_of_random_samples, 10*10))

for i in range(number_of_random_samples):
    training[i] = fun.get_random_image(pixels=100,p=[0.3,0.7])

print(training.shape)

stable_patterns=np.zeros(number_of_random_samples)
percentage = 0.1
for i in range(number_of_random_samples):
    examined_trainining = training[:i+1].copy()

    weights = fun.fit(examined_trainining,method='asdfa')
    stable_points=0
    for j in range(examined_trainining.shape[0]):
        M = examined_trainining.shape[1]
        examined_trainining_noise = examined_trainining[j].copy()
        noise_magnitude = int(percentage* M)
        flipped_indeces = np.random.choice(M, noise_magnitude, replace=False)
        examined_trainining_noise[flipped_indeces] = -examined_trainining_noise[flipped_indeces]
        output = fun.predict(weights, examined_trainining_noise,max_iterations=100)
        if(np.array_equal(examined_trainining[j],output)):
            stable_points+=1
    stable_patterns[i] = stable_points/examined_trainining.shape[0]

print(stable_patterns)

plt.plot(np.arange(1/300,1+1/300,1/300),stable_patterns)
plt.show()
'''

'''fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(percentage_check, max_training_num)
ax.plot_surface(X,Y,accuracy_mean[:,:,0])'''




