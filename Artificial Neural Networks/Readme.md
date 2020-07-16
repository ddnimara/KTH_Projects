### Artificial Neural Network Course DD2437

This course focuses on Neural Network Foundations, by implementing (from scratch, ocassionally using numpy) Neural Network Architectures:
* [Multi Layer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (trained via backprop). MLPs can be utilised for both
classification and regression problems.
* [Self Organising Maps](https://en.wikipedia.org/wiki/Self-organizing_map) (trained via competetive learning). SOMs are typically used for data visualisation,
by projecting multi dimensional data on the 2d plane or 3d space.
* [RBF Networks](https://en.wikipedia.org/wiki/Radial_basis_function_kernel). Similar to MLPs, they can be used in a plethora of applications, both classification and regression.
Usually prefered when the absolute magnitude/scale of features is not as relevant as their relative positions to some prototype vectors.
* [Hopfield Networks](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=A%20Hopfield%20network%20is%20a,systems%20with%20binary%20threshold%20nodes.). Useful for denoising
images (noisy inputs converge to originally stored attractors).
* [Deep Belief Networks](https://en.wikipedia.org/wiki/Deep_belief_network) (BDN), introduced by [Hinton et al.]{https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf). Utilised,
both as a discriminative model, as well as a generative one.
* [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder#:~:text=An%20autoencoder%20is%20a%20type,to%20ignore%20signal%20%E2%80%9Cnoise%E2%80%9D.). Undercomplete
autoencoders can be used for data compression, while overcomplete can be used to project the data into a higher and sparser dimensional space.
