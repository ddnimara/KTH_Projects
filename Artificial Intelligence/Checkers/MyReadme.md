# Extra readme describing the project in more detail

The other readme file was the one given to us in the AI Assignment. It shows how one can run the code in order to formulate an AI playing Checkers!

The task for this assignment was to create an AI agent that would be able to play Checkers. To do so, we had to implement a traditional AI technique called the [Minimax](https://en.wikipedia.org/wiki/Minimax) algorithm. In short, the minimax algorithm searches the game tree (looks at possible next game states) and picks the path that leads the agent to a state of maximum utility (according to a heuristic value), **assuming the opposing player plays optimally**.

However, the state space is far too large and so to save up memory and time I implement improvements:
- Opted into using a refined version of the algorithm, namely Minimax with [Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning).
- Checked for repeated states, storing them in [Transposition Tables](https://en.wikipedia.org/wiki/Transposition_table) utilizing corresponding hash functions.
- Lastly, it is possible that in certain game states (during the early-mid game), the branching factor is not prohibitedly large. It would desirable, in these occasions to be able to search at a lower depth, fully utilizing our time window (had 1 second to make each move). That is, the search depth is **dynamic**. To do so, I implemented iterative deepening.

In the end, my algorithm was capable of solving the assignment in [kattis](https://kth.kattis.com/problems/kth.ai.checkers), solving all 25 problem instances and achieving the best recorded time (thus far) in Java!
