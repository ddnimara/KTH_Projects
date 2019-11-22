# Extra Readme File containing more detailed information about the project

The other readme file contains more practical inforomation on how to run the code. In this assignment, I implemented a Hiddden Markov Model
([HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model))based AI agent in order to predict the **species** and **movements** of birds.

To do so, it **observed** their movements, and based on that it had to deduce:
- The type (specie) of duck.
- The next movement of the duck.

Based on those two inferences, it then had to deduce whether or not to shoot the bird (there was an endagered specie that should be avoided)
and aim its shot taking into account the next movement of the bird. If the shot landed, I was granted a point.

In the end of each round, the agent also guesses the specie of each bird, gaining 1 point of every correct guess. 

To solve this assignment, I implemeneted an HMM class, covering the basic HMM such as Learning model Parameters and making inferences. The
class contains three main algorithms:
- [Forward Algorithm](https://en.wikipedia.org/wiki/Forward_algorithm)
- [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
- [Baum Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)

In the end, we created two lists of HMM models: 
- The first contained a movement predicting HMM for each bird in the instance
- The second contained an HMM for predicting species. The information from one environment carried over to the next.

In the end, my implementation managed to accumulate ~408 (there was some variance, depending on the initialization) points on the [Duck Hunt](https://kth.kattis.com/problems/kth.ai.duckhunt) minigame.
