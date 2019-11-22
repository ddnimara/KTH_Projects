import java.util.*;


public class Player {

    HashMap<String, int[]> maxMap = new HashMap<>();
    HashMap<String,int[]> minMap=new HashMap<>();
    HashMap<String,int[]> prevMax=new HashMap<>();
    HashMap<String,int[]> prevMin=new HashMap<>();
    int bestMove=-1;
    int depth=10;
    int maxPlayer=-1;
    int maxDepth=15;
    int minPlayer=-1;
    boolean isEndGame;


    /**
    * endGame Fucntion. In the end not needed.
    * Counts the pieces of the GameState and if they are below
    * a certain threshold, it returns true indicating we are in the end game.
    * Can be used to determine a switch in heuristic, since one can employ
    * a different one in the end game (a more aggressive one for the winning player)
    */
    public boolean endGame(GameState gameState)
    {
      int pieces=0;

      for (int i=0; i<32; i++)
      {
        if (gameState.get(i)!=0)
        {pieces++;}

      }

      return (pieces<5);
    }

    /**
     * Performs a move
     *
     * @param gameState
     *            the current state of the board
     * @param deadline
     *            time before which we must have returned
     * @return the next state the board is in after our move
     *
     * play method: the method that decides for the optimal move
     * Given our current GameState "gameState" and a Deadline "deadline"
     * we must find the optimal move within a certain time limit.
     * We do so using ab-pruning, using iterative deepening, move ordering
     * and repeated state checking.
     */
    public GameState play(final GameState gameState, final Deadline deadline) {
        Vector<GameState> nextStates = new Vector<GameState>();
        long timeStart=deadline.timeUntil();
        gameState.findPossibleMoves(nextStates); //find next possible moves
        //initialize!
        maxMap.clear();
        minMap.clear();
        if (nextStates.size() == 0) {
            // Must play "pass" move if there are no other moves possible.
            return new GameState(gameState, new Move());
        }
        /**
         * Here you should write your algorithms to get the best next move, i.e.
         * the best next state. This skeleton returns a random move instead.
         */
        //isEndGame=endGame(gameState); Uncomment if you want to implement a different heuristic
        bestMove=0; //by default, the first move is the best at the start
        int best=Integer.MIN_VALUE;
        maxPlayer=gameState.getNextPlayer(); // The player that needs to make a move is the max player
        minPlayer=nextStates.get(0).getNextPlayer(); //The one that follows after is the min Player

        int previousBest=0; //this was not needed in the final code.
                            //we used it to store bestMove[depth-1]
                            //in case our time ended at depth and
                            //needed to return the previous best

        this.depth=1; //we will start iterative search at depth=1
        //run ab-pruning once to initialize our hashes
        best = alphaBetaPruningHa(gameState,this.depth,Integer.MIN_VALUE,Integer.MAX_VALUE);
        long duration = timeStart - deadline.timeUntil();
        double seconds = (double) duration/Math.pow(10,9); //compute how much time has passed
        while(seconds<0.25 && this.depth<maxDepth)
        {
          //prevMax,prevMin: remeber hashes of the previous depth.
          //It will be used for ordering based on their previous evaluation
          prevMax=(HashMap<String,int[]>)maxMap.clone();
          prevMin=(HashMap<String,int[]>)minMap.clone();
          //clear the current max,min hashes, so that we reevaluate the state values
          maxMap.clear();
          minMap.clear();
          //increase the depth
          this.depth+=1;
          previousBest=bestMove; //remember the last one
          best = alphaBetaPruningHa(gameState,this.depth,Integer.MIN_VALUE,Integer.MAX_VALUE);
          //update the time!
          duration = timeStart - deadline.timeUntil();
          seconds = (double) duration/Math.pow(10,9);
        }
        previousBest=bestMove; //as stated earlier, in the end, because we end
                               //our while() preemtevily, we want to return bestMove
        return nextStates.elementAt(previousBest);
    }

    /**
    * The alphabeta pruning algorithm.
    * It gets as input the state, the depth, alpha and beta
    * and returns a value indicating value of the tree rooted at this node.
    * This implementations uses hash tables (transposition table) for state checking
    * since in checkers there are a lot of repeated states.
    * Furthermore, it uses a hash table for ordering which stores the previous
    * "best" node (the one who gave the "v" value) so that our search will start
    * from there in the next iteration of iterating deepening.
    */
    private int alphaBetaPruningHa(GameState gameState, int depth, int alpha, int beta){
      int v = 0; //this is the value which we will return
      int tempMove = 0; //this is the "best" node (the one that will give us the final "v")
      int currentPlayer=gameState.getNextPlayer(); //get the current player to check if he is
                                                   //the max or the min one
      int previousroute=0; //this is the previous best
      String s = representation(gameState); //get the unique string representation of the board
      //transposition table check
      if (currentPlayer==maxPlayer)
      {
        if (prevMax.containsKey(s)) //check for ordering
        {
          previousroute=prevMax.get(s)[2];
        }
        if (maxMap.containsKey(s))
        {
          int[] prevVal=maxMap.get(s);
          int prevDepth=prevVal[1];
          if (prevDepth>=depth) //we use the table if the previous instance
                                //appeared higher in the tree.
          {
            v = maxMap.get(s)[0];
            return v;
          }
        }
      }
      else if (currentPlayer != maxPlayer) //if we have the min player, do the same.
      {
        if (prevMin.containsKey(s))
        {
          previousroute=prevMin.get(s)[2];
        }
        if (minMap.containsKey(s))
        {

          int[] prevVal=minMap.get(s);
          int prevDepth=prevVal[1];
          if (prevDepth>=depth)
          {
            v = minMap.get(s)[0];
            return v;
          }
        }
      }
      //leaf node check
      if (gameState.isEOG() || depth == 0 )
      {
              v=heuristicValue(gameState);
      }
      //normal recursion
      else
      {
          Vector<GameState> nextStates = new Vector<GameState>();
          gameState.findPossibleMoves(nextStates);
          //time to use our memory!
          Vector<GameState> changedStates = (Vector<GameState>) nextStates.clone();
          if(nextStates.size()>1) //use memory only if we need it: i.e. we have more than one choice
          {
            GameState temp = nextStates.get(0);
            changedStates.set(0,nextStates.get(previousroute));
            changedStates.set(previousroute,temp);
          }
          if (currentPlayer == maxPlayer) //for the max player
          {
              v = Integer.MIN_VALUE; //v=-inf
              for (int child = 0; child < nextStates.size(); child++)
              {
                  GameState g = changedStates.get(child);
                  int value;
                  value = alphaBetaPruningHa(g,depth-1,alpha,beta);
                  if (value>v) //if we found a better value, update!
                  {
                    v=value;
                    tempMove=nextStates.indexOf(g); //the state that gave us the update
                  }
                  alpha = Math.max(alpha, v); //alpha update
                  if (beta <= alpha) //prune!
                  {
                      break;
                  }
              }
              bestMove = tempMove;
          }
          else //minplayer
          {
              v = Integer.MAX_VALUE;
              for (int child = 0; child < nextStates.size(); child++)
              {
                  GameState g = changedStates.get(child);
                  int value;
                  value=alphaBetaPruningHa(g,depth-1,alpha,beta);
                  if (value<v)
                  {
                    v=value;
                    tempMove=nextStates.indexOf(g);
                  }
                  beta = Math.min(beta, v); //beta update
                  if (beta <= alpha) //prune!
                  {
                      break;
                  }
              }
          }
      }
      //time to store the values!
      int[] val = {v,depth, tempMove};
      if (currentPlayer==maxPlayer)
      {
        maxMap.put(s,val);
      }
      else
      {
        minMap.put(s,val);
      }
      return v;
        }

    /**
    * Our heuristic function.
    * It is fairly simple: Simply: our pieces - opponent pieces
    * But! we give greater weight (3) to our kings since they are
    * more flexible and provide greater utility.
    */
    public int heuristicValue(GameState gameState)
    {
        //intialize
        int result=0;
        int redPieces=0;
        int whitePieces=0;
        int redKings=0;
        int whiteKings=0;
        int multiply=3;
        //check if someone has won
        if((gameState.isRedWin() && maxPlayer==Constants.CELL_RED ) || (gameState.isWhiteWin() && maxPlayer==Constants.CELL_WHITE) ){return 1000;}
        else if((gameState.isRedWin() && maxPlayer==Constants.CELL_WHITE ) || (gameState.isWhiteWin() && maxPlayer==Constants.CELL_RED)){return -1000;}
        //if not
        else
        {
          for (int i=0; i<32; i++) //scan the board and count the pieces
          {
              if ((gameState.get(i)&Constants.CELL_RED)>0)
              {
                if ((gameState.get(i)&Constants.CELL_KING)>0)
                {redKings++;}
                else
                {redPieces++;}
              }
              else if ((gameState.get(i)&Constants.CELL_WHITE)>0)
                {
                  if ((gameState.get(i)&Constants.CELL_KING)>0)
                  {whiteKings++;}
                  else
                  {whitePieces++;}
                }
          }
          result = redPieces + multiply*redKings - whitePieces - multiply*whiteKings;
          if (maxPlayer==Constants.CELL_WHITE) //Remember: ours vs theirs!
          {
            result=-result;
          }
          return result;
        }
    }
    /**
    * Simple function that gives us a unique string representation of the board.
    *
    */
    public String representation(GameState gameState)
    {
      StringBuilder str= new StringBuilder(); //use stringbuilder for speed
      for (int i=0; i<32; i++)
      {
          if ((gameState.get(i)&Constants.CELL_RED)>0)
          {
            if ((gameState.get(i)&Constants.CELL_KING)>0)
            {str.append("R");} //red king->R
            else
            {str.append("r");} //red piece->r
          }

          else if ((gameState.get(i)&Constants.CELL_WHITE)>0)
            {
              if ((gameState.get(i)&Constants.CELL_KING)>0)
              {str.append("W");} //white king->W
              else
              {str.append("w");} //white piece->w
            }
            else
            {
              str.append("."); //empty space->.
            }
      }
      return str.toString();
    }
}
