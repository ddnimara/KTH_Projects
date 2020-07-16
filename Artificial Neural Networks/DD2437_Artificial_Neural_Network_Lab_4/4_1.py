from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet




image_size = [28,28]
train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

''' restricted boltzmann machine '''

print ("\nStarting a Restricted Boltzmann Machine..")
#%%
# QUESTIONs 1 and 2 (4.1) 

rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                 ndim_hidden=500,
                                 is_bottom=True,
                                 image_size=image_size,
                                 is_top=False,
                                 n_labels=10,
                                 batch_size=20
)
loss_err = np.empty(0)
previous_i = 0
for i in range(10,21):
    n_epochs = i
    loss_err_aux = rbm.cd1(visible_trainset=train_imgs, n_iterations = n_epochs*3000+1, bottom_n_iterations = previous_i*3000,n_nod  = str(rbm.ndim_hidden), n_ep = str(i) )
    loss_err = np.concatenate((loss_err,loss_err_aux))
    linspace = np.arange(0,len(loss_err)*rbm.print_period,rbm.print_period)
    plt.figure()
    plt.plot(linspace,loss_err)
    plt.ylabel("Reconstruction Loss")
    plt.xlabel('Iteration number')
    plt.title('Iteration number vs Reconstruction Loss ('+str(n_epochs)+' epochs)')
    plt.show()
    previous_i = i
    
#%%


#%%



