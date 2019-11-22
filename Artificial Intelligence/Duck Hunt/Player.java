import java.util.ArrayList;
import java.util.*;
class Player {
    HMM[] birds;
    public int timeElapsed=0;
    int timethreshold=80;
    int[] predictions;
    int round=-1; //-1 so that the "if" statement in Shoot works from round 0 (it initializes everything)
    double probThreshold = 0.6;
    HMM[] modelForSpecies=new HMM[Constants.COUNT_SPECIES];
    HMM[] modelsForSpeciesTemp;
    int shots=0;
    int hits=0;
    int statesForSpecies=1;
    int[][] successGuess=new int[Constants.COUNT_SPECIES][Constants.COUNT_MOVE];
    int[][] totalguess=new int[Constants.COUNT_SPECIES][Constants.COUNT_MOVE];
    int lastMovement=-1;
    int lastSpecie=-1;

    public Player() {

    }

    /**
     * Shoot!
     *
     * This is the function where you start your work.
     *
     * You will receive a variable pState, which contains information about all
     * birds, both dead and alive. Each bird contains all past moves.
     *
     * The state also contains the scores for all players and the number of
     * time steps elapsed since the last time this function was called.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return the prediction of a bird we want to shoot at, or cDontShoot to pass
     */
    public Action shoot(GameState pState, Deadline pDue) {
        /*
         * Here you should write your clever algorithms to get the best action.
         * This skeleton never shoots.
         */

         Action result=cDontShoot;
         //time is ticking!
         timeElapsed++;

         double weight; // parameters that changes depending on the number of players that are in the round.

         int numberOfBirds=pState.getNumBirds();
         //Each round we begin from scratch!
         if (round!=pState.getRound())
         {

           if (pState.getNumPlayers()>1) //if we are not alone, start shooting sooner!
           {weight=3;}
           else
           {weight=1.5;} // Else,take your time to build better models.

           round=pState.getRound(); // update round
           //intialize
           timeElapsed=0;
           birds=new HMM[numberOfBirds];
           modelsForSpeciesTemp=new HMM[numberOfBirds];
           timethreshold=100-(int)weight*numberOfBirds;
           predictions = new int[birds.length];
           for (int i=0; i<birds.length; i++)
           {
             birds[i]=new HMM();
             predictions[i]=Constants.SPECIES_UNKNOWN;
             modelsForSpeciesTemp[i]=new HMM(statesForSpecies);

           }


         }
         // at round 0 don't shoot, too risky! (fear the stork!)
         if(round==0)
         {
           return cDontShoot;
         }
         //int birdToShoot=-1;
         lastMovement=-1;
         double minProb=0;
         lastSpecie=-1;

         if(timeElapsed<timethreshold) //idle state. We do not do anything while we wait for enough observations
                                       //slow and steady wins the race!
         {

           result=cDontShoot;
         }
         else// shooting stage. We have our observations. Ready to fire!
         {

             for (int i=0;i<birds.length; i++)
             {

               if(pDue.remainingMs()<100) //failsafe. If we are lagging behind, just don't shoot and keep going!
               {return cDontShoot;}

                    Bird bird=pState.getBird(i); // We select the i-th bird
                    if (bird.isDead()==false)  // Let us not shoot a dead bird :P
                    {
                      int[] observations = storeObservations(bird);

                      HMM testForStork=new HMM(statesForSpecies); // we will train a single state specie models
                                                                  // to compare it with the rest
                      testForStork.scaledtraining(false,observations,50);
                      double[][] b = testForStork.getB(); // we will compare its B matrix with the rest.
                                                          // if it defers substantially, don't shoot! (likely a new specie)

                      int prediction=predictSpecie(bird); //make your prediction.


                      if(prediction==Constants.SPECIES_UNKNOWN || prediction==Constants.SPECIES_BLACK_STORK) //dont shoot unknown species or storks
                      {continue;}


                      if (l1Norm(b,modelForSpecies[prediction].getB())>0.55) //Heuristic approach. 0.55 was found after alot of experimentation as a good threshold.
                      {continue;}
                      if(pDue.remainingMs()<100) // second failsafe. If time is running out, abort!
                      {return cDontShoot;}

                      double bestModelProb=Double.NEGATIVE_INFINITY; // first we check which number of states
                                                                     // explain the observations best
                      int[] statePossibilities={3,4,5};              // we check for #states=3,4,5

                      for (int w : statePossibilities)
                      {
                        HMM temp = new HMM(w);
                        temp.scaledtraining(false,observations,100); //train for 100 iterations
                        double pr=temp.observationProbability(observations);

                        if (pr>bestModelProb) // if this model explains the data better, pick it!
                        {
                          bestModelProb=pr;
                          birds[i]=temp;
                        }
                      }

                      birds[i].scaledtraining(true,observations,100); //train it abit more.

                        HMM current = birds[i];
                        // time to calculate emission vector.
                        double[][] A=current.getA();
                        double[][] B=current.getB();
                        //first calculate a[T] = P[X_T|Observations]
                        Object[] params=current.scaledforwardAlgorithm(observations);
                        double[][] alpha = (double[][])params[0];
                        double[] stateVector=alpha[observations.length-1];
                        HMM temp = new HMM(stateVector,A,B); // create an hmm with pi=a[T]
                        double[] probs = temp.sequenceProb(); //calculate a[T]*A*B
                        //time to use our previous knowledge!
                        double sum=0;
                        for (int l=0; l<probs.length; l++)
                        {
                          //calculate a coefficient that takes into account the efficacy of previous models when predictign the "l" move for specie "prediction"
                          double coefficient = (totalguess[prediction][l]>0 ? (double)successGuess[prediction][l]/totalguess[prediction][l]:0);
                          //If we generally missed in the passed, notify the agent about this using the coefficient
                          probs[l]=probs[l]*((double)1/Constants.COUNT_MOVE)*(1+ coefficient); // we are stiring his confidence away from 1/9 (uniform)

                          sum+=probs[l]; //calculating sum for normalization
                        }
                        //normalize!
                        for (int l=0; l<probs.length; l++)
                        {
                          probs[l]=probs[l]/sum;
                        }

                        //fine the max probability!
                        for (int k=0; k<probs.length; k++)
                        {
                            if(minProb<probs[k])
                            {

                              minProb=probs[k];
                              lastMovement=k;
                              result=new Action(i,k);
                              lastSpecie=prediction;
                            }


                        }


                      }



                    }



          }


        if(minProb < probThreshold) { // if our max probability emission is not good enough, don't shoot!
           result = cDontShoot;

         }
         else if (timeElapsed>=timethreshold) // if we decide to shoot
         {
           //shots++ <- this was used for evaluating our model's accuracy.
           totalguess[lastSpecie][lastMovement]+=1; // we shot, remember that we shot the "predicted" specie at that movement.

         }

        return result;

    }
    /**
    * Max norm, ||A||_{inf}, also known as row norm.
    * As the name implies, returns the max row sum.
    */
    public double maxNorm(double[][] a, double[][] b)
    {

      double max=-1;
      for (int i=0; i<a[0].length; i++)
      {
        double sum=0;
        for (int j=0; j<a.length; j++)
        {
          sum+=Math.abs(a[j][i]-b[j][i]);

        }
        if (max<sum)
        {max=sum;}
      }

      return max;
    }
    /**
    * L1 norm, also known as column norm. Returns the max column sum.
    * After many experiments, we decided that this captures the differences
    * between species optimially.
    */
    public double l1Norm(double[][] a, double[][] b)
    {
      double result =0;

      for (int i=0; i<a.length; i++)
      {
        double sum=0;
        for (int j=0; j<a[0].length; j++)
        {
          sum+=Math.abs(a[i][j]-b[i][j]);
        }
        if(result<sum)
        {result=sum;}
      }

      return result;
    }
    /**
    * Frobenius matrix norm. Logical generalization of
    * euclidian vector distance. simply the square root of the sum of squares
    * of all the elements.
    */
    public double frobeniusNorm(double[][] a, double[][] b)
    {
      double result =0;

      for (int i=0; i<a.length; i++)
      {
        for (int j=0; j<a[0].length; j++)
        {
          result+=Math.pow(a[i][j]-b[i][j],2);
        }
      }
      result=Math.sqrt(result);
      return result;
    }
    /**
    * Method that stores the bird observations into an array.
    * Note that it ignores its observations when it was dead.
    */
    public int[] storeObservations(Bird b) //returns int array with bird observations
      {
          ArrayList<Integer> obs=new ArrayList();

          for (int i=0; i < b.getSeqLength(); i++)
          {
              if(b.wasDead(i)==true)
              {break;}

                obs.add(b.getObservation(i));

          }
          int[] observations= new int[obs.size()];
          for (int i=0; i < obs.size(); i++)
          {
              observations[i]=obs.get(i);

          }
          return observations;
      }

      /**
       * Guess the species!
       * This function will be called at the end of each round, to give you
       * a chance to identify the species of the birds for extra points.
       *
       * Fill the vector with guesses for the all birds.
       * Use SPECIES_UNKNOWN to avoid guessing.
       *
       * @param pState the GameState object with observations etc
       * @param pDue time before which we must have returned
       * @return a vector with guesses for all the birds
       */
      public int[] guess(GameState pState, Deadline pDue) {
          /*
           * Here you should write your clever algorithms to guess the species of
           * each bird. This skeleton makes no guesses, better safe than sorry!
           */

          int[] lGuess = new int[pState.getNumBirds()];

          if (round==0) //first round we do not know what to do.
          {

          	for (int i = 0; i < pState.getNumBirds(); ++i)
              {
                lGuess[i] = Constants.SPECIES_PIGEON; //guess randomly! We need to guess something
                                                      //otherwise nothing is revealed.

              }
          }
          else
          {

          	for (int i=0; i<pState.getNumBirds(); i++) // we are gonna guess each bird
          	{

              Bird b = pState.getBird(i);

              	lGuess[i]=predictSpecie(b); //guess the most likely specie
                predictions[i]=lGuess[i]; // store it (so we can compare it after the reveal)
                                          // mostly used to evaluate our model
          	}



          }


          return lGuess;
      }

      /**
      * Method for predicting the specie.
      */
      public int predictSpecie(Bird b)
      {
          int mostLikely=-1; //default is unknown

          double maxProb=Double.NEGATIVE_INFINITY;

          for (int j=0; j<Constants.COUNT_SPECIES; j++) // we need to find the most likely specie
                                  // i.e. the specie model that maximizes the
                                  // probability P(Observations)
          {
            if(modelForSpecies[j]!=null) // we do this for the spicies we have a model on

            {

              int[] obs=storeObservations(b);
              double[][] A = modelForSpecies[j].getA();

              double probability=modelForSpecies[j].observationProbability(obs); //get logProbability

              if (maxProb<probability)
              {
                maxProb=probability; // we found a more likely specie!
                mostLikely=j; // store it
              }

            }
          }
          //Note: we are not extra cautious with what we are guessing, because it is Generally
          //beneficial to us to guess (even if we are wrong), cause then we get more data
          //via the reveal stage.
          return mostLikely;
      }

      /**
       * If you hit the bird you were trying to shoot, you will be notified
       * through this function.
       *
       * @param pState the GameState object with observations etc
       * @param pBird the bird you hit
       * @param pDue time before which we must have returned
       */

      public void hit(GameState pState, int pBird, Deadline pDue) {
          //hits++; <- This was used for evaluating our accuracy
          successGuess[lastSpecie][lastMovement]+=1; //store that you hit it!
          //System.err.println("HIT BIRD!!!");
      }

      /**
       * If you made any guesses, you will find out the true species of those
       * birds through this function.
       *
       * @param pState the GameState object with observations etc
       * @param pSpecies the vector with species
       * @param pDue time before which we must have returned
       */
      public void reveal(GameState pState, int[] pSpecies, Deadline pDue) {

      	int numbOfBirds=pSpecies.length;
        //store the bird models in the follow list of lists
      	ArrayList<ArrayList<HMM>> storingModels=new ArrayList<ArrayList<HMM>>(Constants.COUNT_SPECIES);
        for (int i=0; i<Constants.COUNT_SPECIES; i++)
        {
          //initialization
          storingModels.add(new ArrayList<HMM>());

        }
      	for (int i=0; i<numbOfBirds; i++)
      	{
          //get the bird
          Bird b = pState.getBird(i);
          int[] observations = storeObservations(b);
          //train a specie model.
          modelsForSpeciesTemp[i].scaledtraining(false,observations,100);


      		if (pSpecies[i]!=Constants.SPECIES_UNKNOWN) //just in case pSpecies is unknown
      		{
            //if the true specie has been revealed to us.

            //store our specie model
            storingModels.get(pSpecies[i]).add(modelsForSpeciesTemp[i]);
          } //Store the bird models corresponding to i-th specie

      	}


      	for (int i=0; i<Constants.COUNT_SPECIES; i++) // time to update our species models!
      	{
          if (storingModels.get(i).isEmpty()==false) // only do this if we have atleast one model for it
          {
            if (modelForSpecies[i]!=null){storingModels.get(i).add(modelForSpecies[i]);} //add the existing model to the list

        		//time to average!

        		modelForSpecies[i] = HMM.getAverage(storingModels.get(i)); // this is a better estimate of our species models
        															//throughout this environment.
          }

      	}

      }

      public static final Action cDontShoot = new Action(-1, -1);
  }
