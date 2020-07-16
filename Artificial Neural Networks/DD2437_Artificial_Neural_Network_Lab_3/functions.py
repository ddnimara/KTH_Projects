# Import libraries
import numpy as np
import itertools
import load_data as ld
import matplotlib.pyplot as plt
import seaborn as sns

"""
This lab is addressed to:
- Explain the principles underlying the operation and functionality of au-
toassociative networks;
- Train the Hopfield network;
- Explain the attractor dynamics of Hopfield networks the concept of energy
function;
- Demonstrate how autoassociative networks can do pattern completion and
noise reduction;
- Investgate the question of storage capacity and explain features that help
increase it in associative memories.
"""

def fit(training, method = 'self_connection'):
    M = training.shape[1]
    weights = np.zeros((M,M))
    N = training.shape[0]
    for i in range(M):
        for j in range(M):
            if method == 'self_connection':
                weights[i, j] = 1/M * np.dot(training[:,i],training[:,j])
            else:
                if i == j:
                    weights[i, j] = 0
                else:
                    weights[i, j] = 1/M * np.dot(training[:,i],training[:,j])

    return weights

def sparse_fit(training,method='self_connection'):
    M = training.shape[1]
    weights = np.zeros((M, M))
    N = training.shape[0]
    rho = training.mean()
    #print('rho',rho)
    for i in range(M):
        for j in range(M):
            if method == 'self_connection':
                weights[i, j] = np.dot(training[:, i]-rho, training[:, j]-rho)
            else:
                if i == j:
                    weights[i, j] = 0
                else:
                    weights[i, j] = np.dot(training[:, i]-rho, training[:, j]-rho)
    #print('weights', weights.mean())
    #print('training size ', N)
    #print('max weight ', np.max(weights))
    return weights

def sparse_activation(x,bias,weight_row):
    print("activation magnitude ",np.dot(weight_row,x))
    return 0.5 + 0.5*sign(np.dot(weight_row,x)-bias)


def sparse_predict(weights,bias,test_original,max_iterations=100):
    M = weights.shape[0]
    previous_test = np.zeros((1, M))
    counter = 0
    np.random.seed(42)
    test = test_original.copy()
    while(counter<max_iterations):
        previous_test = test.copy()
        #update_index = np.random.randint(M)
        #updated_bit = sign(np.dot(weights[update_index],test))
        for i in range(M):
            update_bit = sparse_activation(test,bias,weights[i])
            test[i] = update_bit
        if (np.array_equal(previous_test,test)):
            break
        counter+=1
        #print("Total number of iterations ",counter)
    return test
def sign(x):
    if x>=0:
        return 1
    else:
        return -1


def sign_vec(x):
    for i in range(x.shape[0]):
        if (x[i]>=0):
            x[i]=1
        else:
            x[i]=-1
    return x

def predict_little_model(weights,test_original,max_iterations=100):
    M = weights.shape[0]
    previous_test = np.zeros((1, M))
    counter = 0
    np.random.seed(42)
    test = test_original.copy()

    while (counter < max_iterations):
        previous_test = test.copy()
        # update_index = np.random.randint(M)
        # updated_bit = sign(np.dot(weights[update_index],test))
        test = sign_vec(np.dot(weights,previous_test))
        if (np.array_equal(previous_test, test)):
            break
        counter += 1
    # print("Total number of iterations ",counter)
    return test


def predict(weights,test_original,max_iterations=200,energy = False):
    M = weights.shape[0]
    previous_test = np.zeros((1,M))
    counter=0
    np.random.seed(42)
    test = test_original.copy()
    if (not energy):
        while(counter<max_iterations):
            previous_test = test.copy()
            #update_index = np.random.randint(M)
            #updated_bit = sign(np.dot(weights[update_index],test))
            for i in range(M):
                update_bit = sign(np.dot(weights[i],test))
                test[i] = update_bit
            if (np.array_equal(previous_test,test)):
                break
            counter+=1
        #print("Total number of iterations ",(counter+M))
        return test
    else:
        energy_vector = [] 
        energy_vector_step = [] 
        energy_vector_step.append(compute_lyapunov_energy(test,weights))
        energy_vector.append(compute_lyapunov_energy(test,weights))
        while(counter<max_iterations):
            previous_test = test.copy()
            for i in range(M):
                immediate_previous = test.copy()
                update_bit = sign(np.dot(weights[i],test))
                test[i] = update_bit
                energy_vector_step.append(compute_lyapunov_energy(test,weights)) #Every time we run through the for loop we compute the energy. If the
                #pattern does not change we will see a plateu and, therefore, a stepped plot until reaching the minimum
                if(not np.array_equal(immediate_previous,test)):
                    energy_vector.append(compute_lyapunov_energy(test,weights)) #In this case, the energy is stored everytime the pattern changes in order to
                    #get a more continuous plot. In the end, both lists will be returned and the one explaining the data in a better format can be plotted
            if (np.array_equal(previous_test,test)):
                break
            counter+=1
        print("Total number of iterations ",counter)
        return test, np.array(energy_vector),np.array(energy_vector_step)

def number_of_attractors(weights):
    lst = np.array(list(itertools.product([-1, 1], repeat=8)))
    results = []
    for i in range(lst.shape[0]):
        results.append(predict(weights,lst[i],100))

    results=np.array(results)
    return results

def compute_lyapunov_energy(x,w):
    return -x@w@x.T
    
def compute_accuracy(real,predicted):
    return real[real==predicted].shape[0]/real.shape[0]

def get_random_image(pixels=1024,p=[0.5,0.5]):
    return np.random.choice([-1,1],size=pixels,p=p)


def three_point_five_first_bullet():
    # Load data
    filepath = r"data/pict.dat"
    pictures_matrix = ld.get_pictures(filepath)

    # Learning
    max_training_num = [3,4, 5, 6, 7]
    training = np.zeros((max_training_num[-1], pictures_matrix.shape[1] * pictures_matrix.shape[2]))

    for i in range(max_training_num[-1]):
        training[i] = np.ravel(pictures_matrix[i])
    np.random.seed(42)

    max_iter = 5

    percentage_check = np.arange(15, 20, 5)
    accuracy_mean = accuracy_with_sample_images(percentage_check, max_training_num, max_iter, training)
    plt.plot(max_training_num, accuracy_mean[0, :, 0],label='p1')
    plt.plot(max_training_num, accuracy_mean[0, :, 1],label='p2')
    plt.plot(max_training_num, accuracy_mean[0, :, 2],label='p3')
    plt.title("Accuracy as a function of Number of Pictures Memorized")
    plt.legend()
    plt.xlabel("Number of Patterns memorized-trained on")
    plt.ylabel("Accuracy (% pixels)")
    plt.show()

def three_point_five_second_bullet():
    filepath = r"data/pict.dat"
    pictures_matrix = ld.get_pictures(filepath)

    # Learning
    number_of_random_samples = 15
    max_training_num = np.arange(3, 4 + number_of_random_samples, 1)  # [3,4, 5, 6, 7]

    np.random.seed(42)

    max_iter = 1

    percentage_check = np.arange(15, 20, 5)

    training = np.zeros((3 + number_of_random_samples, pictures_matrix.shape[1] * pictures_matrix.shape[2]))

    for i in range(3):
        training[i] = np.ravel(pictures_matrix[i])

    for i in range(number_of_random_samples):
        training[i + 3] = get_random_image()

    accuracy_mean = accuracy_with_sample_images(percentage_check, max_training_num, max_iter, training)
    plt.plot(max_training_num, accuracy_mean[0, :, 0],label='p1')
    plt.plot(max_training_num, accuracy_mean[0, :, 1],label='p2')
    plt.plot(max_training_num, accuracy_mean[0, :, 2],label='p3')
    plt.title("Accuracy as a function of Number of Pictures Memorized")
    plt.legend()
    plt.xlabel("Number of Patterns memorized-trained on")
    plt.ylabel("Accuracy (% pixels)")
    plt.show()



def three_point_five_fourth_bullet():
    number_of_random_samples = 300
    max_training_num = np.arange(1, number_of_random_samples + 1, 1)  # [3,4, 5, 6, 7]

    np.random.seed(1)

    max_iter = 1

    percentage_check = np.arange(0, 55, 5)
    training = np.zeros((number_of_random_samples, 10 * 10))

    for i in range(number_of_random_samples):
        training[i] = get_random_image(pixels=100)

    print(training.shape)

    stable_patterns = np.zeros(number_of_random_samples)

    for i in range(number_of_random_samples):
        examined_trainining = training[:i + 1].copy()
        weights = fit(examined_trainining, method='self_connection')
        if i == 0 or i == 99 or i == 199 or i == 299:
            sns.heatmap(weights, cmap = 'winter')
            plt.title('Heatmap after stack %i patterns' %(i+1))
            plt.show()
        stable_points = 0
        for j in range(examined_trainining.shape[0]):
            output = predict(weights, examined_trainining[j], max_iterations=1)
            if (np.array_equal(examined_trainining[j], output)):
                stable_points += 1
        stable_patterns[i] = stable_points / examined_trainining.shape[0]

    print(stable_patterns)

    plt.plot(np.arange(1, 301, 1), stable_patterns, label='Stable Points %')
    plt.vlines(13.8, 0, 1, linestyles='dashed', label='0.138*N')
    plt.title('Stable Pattern Percentage as a function of the training size')
    plt.xlabel('Number of Training Patterns')
    plt.ylabel('Stable Pattern Percentage')
    plt.legend()
    plt.show()

def three_point_five_fifth_bullet(method='self_connection'):
    number_of_random_samples = 300
    max_training_num = np.arange(1, number_of_random_samples + 1, 1)  # [3,4, 5, 6, 7]

    np.random.seed(42)

    max_iter = 1

    percentage_check = np.arange(0, 55, 5)
    training = np.zeros((number_of_random_samples, 10 * 10))

    for i in range(number_of_random_samples):
        training[i] = get_random_image(pixels=100)

    print(training.shape)

    stable_patterns = np.zeros(number_of_random_samples)
    percentage = 0.1
    for i in range(number_of_random_samples):
        examined_trainining = training[:i + 1].copy()

        weights = fit(examined_trainining, method)
        stable_points = 0
        for j in range(examined_trainining.shape[0]):
            M = examined_trainining.shape[1]
            examined_trainining_noise = examined_trainining[j].copy()
            noise_magnitude = int(percentage * M)
            flipped_indeces = np.random.choice(M, noise_magnitude, replace=False)
            examined_trainining_noise[flipped_indeces] = -examined_trainining_noise[flipped_indeces]
            output = predict(weights, examined_trainining_noise, max_iterations=100)
            if (np.array_equal(examined_trainining[j], output)):
                stable_points += 1
        stable_patterns[i] = stable_points / examined_trainining.shape[0]

    print(stable_patterns)

    plt.plot(np.arange(1,301,1), stable_patterns,label='Stable Points %')
    plt.vlines(13.8,0,1,linestyles='dashed',label='0.138*N')
    plt.legend()
    plt.title('Stable Pattern Percentage as a function of the training size')
    plt.xlabel('Number of Training Patterns')
    plt.ylabel('Stable Pattern Percentage')

    plt.show()

def three_point_five_sixth_bullet(method='self_connection',p=[0.1,0.9]):
    number_of_random_samples = 300
    max_training_num = np.arange(1, number_of_random_samples + 1, 1)  # [3,4, 5, 6, 7]

    np.random.seed(42)

    max_iter = 1

    percentage_check = np.arange(0, 55, 5)
    training = np.zeros((number_of_random_samples, 10 * 10))

    for i in range(number_of_random_samples):
        training[i] = get_random_image(pixels=100,p=p)

    print(training.shape)

    stable_patterns = np.zeros(number_of_random_samples)
    percentage = 0.1
    for i in range(number_of_random_samples):
        examined_trainining = training[:i + 1].copy()

        weights = fit(examined_trainining, method)
        stable_points = 0
        for j in range(examined_trainining.shape[0]):
            M = examined_trainining.shape[1]
            examined_trainining_noise = examined_trainining[j].copy()
            noise_magnitude = int(percentage * M)
            flipped_indeces = np.random.choice(M, noise_magnitude, replace=False)
            examined_trainining_noise[flipped_indeces] = -examined_trainining_noise[flipped_indeces]
            output = predict(weights, examined_trainining_noise, max_iterations=100)
            if (np.array_equal(examined_trainining[j], output)):
                stable_points += 1
        stable_patterns[i] = stable_points / examined_trainining.shape[0]

    print(stable_patterns)

    plt.plot(np.arange(1 / 300, 1 + 1 / 300, 1 / 300), stable_patterns)
    plt.show()


def accuracy_with_sample_images(percentage_check,max_training_num,max_iter,training):
    accuracy = np.zeros((percentage_check.shape[0], max_iter, len(max_training_num), max_training_num[-1]))
    for per in range(percentage_check.shape[0]):
        print('percentage:',percentage_check[per])
        for k in range(len(max_training_num)):
            print('model:',str(k+1))
            print('with size',max_training_num[k])
            for it in range(max_iter):
            #distort image
                sub_train = training[:max_training_num[k]]
                weights = fit(sub_train)
                for j in range(sub_train.shape[0]):
                    train = sub_train[j]
                    training_noise = train.copy()
                    M = training_noise.shape[0]
                    noise_magnitude = int(percentage_check[per]/100 * M)
                    if(percentage_check[per]>0):
                        print(noise_magnitude)
                    flipped_indeces = np.random.choice(M, noise_magnitude, replace=False)
                    training_noise[flipped_indeces] = -training_noise[flipped_indeces]
                    output = predict_little_model(weights,training_noise)
                    accuracy[per,it,k,j] = compute_accuracy(train,output)
                print('accuracies ', accuracy[per,it,k])

    print(accuracy)

    accuracy_mean = accuracy.mean(axis=1)
    return accuracy_mean

def three_point_six(activity=10,bias_low=0,bias_high=6,bias_step=0.5):
    number_of_random_samples = 300
    training = np.zeros((number_of_random_samples, 100))



    skeleton = np.zeros(100)
    skeleton[:activity] = 1
    np.random.seed(42)
    if(activity==1):
        training = np.zeros((100, 100))
        number_of_random_samples=100
        for i in range(100):
            training[i][i]=1
    else:
        for i in range(training.shape[0]):
            training[i] = np.random.permutation(skeleton)
    print("Unique samples ", np.unique(training,axis=0).shape[0])
    print('training size', training.shape)
    bias_range = np.arange(bias_low, bias_high, bias_step)

    capacity_size = np.zeros(bias_range.shape[0])
    count=0
    for bias in bias_range:
        stable_patterns = np.zeros(number_of_random_samples)

        for i in range(number_of_random_samples-1):
            examined_training = training[:i + 1].copy()

            weights = sparse_fit(examined_training, method='self_connection')
            stable_points = 0
            for j in range(examined_training.shape[0]):
                output = sparse_predict(weights, bias, examined_training[j], max_iterations=100)
                # print('outputs ratio',output[output==1].shape[0]/output.shape[0])
                if (np.array_equal(examined_training[j], output)):
                    stable_points += 1
            stable_patterns[i] = stable_points / examined_training.shape[0]
            if (stable_patterns[i]<1):
                break


        print(stable_patterns)
        capacity_size[count] = measure_capacity(stable_patterns)
        print('capacity size',capacity_size[count])
        '''
        plt.plot(np.arange(1, 301, 1), stable_patterns, label='Stable Points %')
        plt.vlines(13.8, 0, 1, linestyles='dashed', label='0.138*N')
        plt.legend()
        plt.title('Stable Pattern Percentage as a function of the training size')
        plt.xlabel('Number of Training Patterns')
        plt.ylabel('Stable Pattern Percentage')

        plt.show()
        '''
        count+=1
    plt.plot(bias_range, capacity_size, label = 'Capacity Size')
    position = bias_low + np.argmax(capacity_size)*bias_step
    plt.vlines(position,0,np.max(capacity_size),linestyles='dashed',label='At '+str(round(position,1)))
    plt.title("Capacity size as a function of the bias")
    plt.xlabel("Bias")
    plt.ylabel("Capacity Size")
    plt.legend()
    plt.show()


def measure_capacity(stable_patterns):
    counter=0
    for num in stable_patterns:
        if (num==1):
            counter+=1
        else:
            break
    return counter