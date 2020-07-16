import numpy as np
import load_data as ld
import functions as fun

training = ld.get_data('Training')
testing = ld.get_data('Testing')
biz_test = ld.get_data('Dissimilar')

weights = fun.fit(training)



for i in range(testing.shape[0]):

    print('initial', testing[i])
    test_output = fun.predict(weights,testing[i],100)
    print('target', training[i])
    print('ouput',test_output)
    print('-------------')


results = fun.number_of_attractors(weights)

print('attractors',np.unique(results,axis=0))

print('training', training)
print("number of attractors ", np.unique(results,axis=0).shape[0])

print('initial bizzare', biz_test)
biz_out=fun.predict(weights,biz_test,100)

print('final bizzare',biz_out)