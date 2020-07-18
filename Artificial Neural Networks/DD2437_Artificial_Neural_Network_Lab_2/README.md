# DD2437_Artificial_Neural_Network_Lab_2

This lab examines [Self Orginisining Maps](https://en.wikipedia.org/wiki/Self-organizing_map) and [RBF Networks](https://en.wikipedia.org/wiki/Radial_basis_function_network#:~:text=In%20the%20field%20of%20mathematical,the%20inputs%20and%20neuron%20parameters.).
More specifically, we analysed:
* Data visualisation via SOMs by projecting high dimensional dat on 1 and 2 dimensional grids.
* Utilised SOMs find a circular path between cities (can be seen as simpler variant of the Travelling Salesman Person TSP).
* Approximated basic functions via RBF networks.
* Examined the sensitivity of both architectures to initialisation.
* Analysed relationship between the number of Clusters in both models and the model's capacity. 
Generally, too many clusters lead to overfit, while too few to underfit.
* Importance of dynamic versus static neighbourhood in SOM output space.

## Examples

### RBF Networks

<img src="Explanatory Images/RBFInitialisation.png" title="RBF">

### SOMs

<img src="Explanatory Images/AnimalSom.png" title="Animal Ordering">

<img src="Explanatory Images/SOMsPoliticians.png" title="Politician Visualisation">

<img src="Explanatory Images/SOM_Cities.png" title="Cyclic Path">
