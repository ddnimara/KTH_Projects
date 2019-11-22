import java.util.Arrays;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
public class HMM {

	double[] pi;
	double[][] A;
	double[][] B;
	double stabilizer=10e-40; //the stabilizer is used in divisions so we avoid 0/0 cases.

	static int prec=10;
	static int states=1;

	/**
	 * Default constructor for HMM. Creates an empty HMM.
	 */
	public HMM() {
		pi=null;
		A=null;
		B=null;
	}
	/**
	 * Second (more sophisticated) constructor for HMM. Creates an HMM with
	 * pi=initial, A=transition, B=observations.
	 */
	public HMM(double[] initial, double[][] transition, double[][] observations) {
		pi=initial;
		A=transition;
		B=observations;
	}
	/**
	 * HMM constructor that creates a "random" HMM (close to uniform)
	 * with hidden states=nstates
	 */
public HMM(int nstates)
{
	pi=initializeMatrixr(1,nstates)[0];
	A=initializeMatrixr(nstates,nstates);
	B=initializeMatrixr(nstates,Constants.COUNT_MOVE);
}


/**
 * Returns the pi vector.
 */
	public double[] getPi() {
		return pi;
	}
	/**
	 * Sets the pi vector = initial
	 */
	private void setPi(double[] initial) {
		pi=initial;
	}
	/**
	 * Returns the A matrix.
	 */
	public double[][] getA() {
		return A;

	}


	/**
	 * Overides object toString() method. Used to get a succint representation of the hmm parameters.
	 */
	@Override
	public String toString() {
		return "HMM [pi=" + Arrays.toString(pi) + ", A=" + Arrays.deepToString(A) + ", B=" + Arrays.deepToString(B) + "]";
	}
	/**
	 * Sets transition matrix A=transition.
	 */
	private void setA(double[][] transition) {
		A=transition;
	}

	/**
	 * Returns emission matrix B
	 */
	public double[][] getB() {
		return B;
	}
	/**
	 * Sets B=observations
	 */
	private void setB(double[][] observations) {
		B=observations;
	}


	/**
	 * Random nxm matrix intialization. The matrices are almost uniform
	 * meaning ~1/m (m: collumn dimension). The resulting matrix is stohastic.
	 */
	public double[][] initializeMatrixr(int n,int m)
	{
		double[][] result=new double[n][m];

		for (int i=0; i<n; i++)
		{
			result[i][m-1]=1; // this is used to ensure that the matrix is row stohastic

			for (int j=0; j<m-1; j++)
			{


				result[i][j]=(double)1/m +(-1 +3*Math.random())/(20*m);
				result[i][m-1]-=result[i][j];
			}
		}
		return result;
	}
	/**
	 * Similar to the above method. Creates a trully random stohastic matrix.
	 * It serves as an alternative, though the almost uniform intialization
	 * showed more consistent results.
	 */
	public double[][] initializeMatrixra(int n,int m)
	{
		double[][] result=new double[n][m];
		Random r = new Random();
		for (int i=0; i<n; i++)
		{
			double sum=0;

			for (int j=0; j<m; j++)
			{


				result[i][j]=r.nextDouble();
				sum+=result[i][j];
			}
			for (int j=0; j<m; j++)
			{


				result[i][j]=result[i][j]/sum;

			}
		}
		//System.err.println("Initial matrix" + Arrays.deepToString(result));
		return result;
	}

	/**
	 * In case we want a more "readable" and "reasonable" representation of the matrices
	 * Generally good to avoid since round function uses BigDecimal which makes
	 * everything slower. Useful if we wanna round small numbers.
	 */
	private void roundMatrix(int precision)
	{
		for (int i=0; i<pi.length; i++)
		{
			pi[i]=round(pi[i],precision);
		}


		for (int i=0; i<A.length; i++)
		{
			for (int j=0; j<A[0].length; j++)
			{
				A[i][j]=round(A[i][j],precision);
			}
		}

		for (int i=0; i<B.length; i++)
		{
			for (int j=0; j<B[0].length; j++)
			{
				B[i][j]=round(B[i][j],precision);
			}
		}

	}

	/**
	 * Simple Matrix multiplication. Primarily used for computing pi*A*B
	 * to compute next emmision probability.
	 */
	private static double[][] matrixMultiply(double[][] a, double[][] b) {
	        int n = a.length;
	        int m = a[0].length;
	        int p = b[0].length;
	        double[][] result = new double[n][p];
	        double sum;

	        for(int i = 0; i < n; i++) {
	            for(int j = 0; j < p; j++) {
	                sum = 0;
	                for(int k = 0; k < m; k++) {
	                     sum += a[i][k] * b[k][j];
	                }
	                result[i][j] = sum;
	            }
	        }
	        return result;
	    }

			/**
			 * Computes P(O=observations| pi,A,B). We implemented the scaled implementation of the
			 * HMM algorithms and as such the probability is simply the sum of 1/c.
			 * To avoid underflow, we compute in log space.
			 */
	public double observationProbability(int[] observations)
	{
		Object[] forward=scaledforwardAlgorithm(observations);
		double[] c = (double[])forward[1];
		double[][] a = (double[][]) forward[0];
		double logProb=0;

		for (int i=0; i<c.length; i++)
		{
			logProb+=Math.log(c[i]);
		}

		logProb=-logProb;

		if(Double.isNaN(logProb)==true) // In case we get NaN.
		{

			logProb=Double.NEGATIVE_INFINITY;
		}
		return logProb;

	}



	/**
	 * Given our pi,A,B. the emission probability vector in T=1 is pi*A*B.
	 * We use this method in the Player class to compute alpha[T-1]*A*B=P(O_{t+1} | observations, pi, A. B)
	 */
	public double[] sequenceProb()
	{
		double[][] pi2=new double[1][pi.length];
		pi2[0]=pi;
		double[][] firstResult = matrixMultiply(pi2, A);
    double[][] secondResult = matrixMultiply(firstResult, B);
		return secondResult[0];
	}

	/**
	 * Scaled implementation of the forward algorith or alpha-pass.
	 * Along the way, we computed the scaling factors c.
	 */
	public Object[] scaledforwardAlgorithm(int[] observations)
	{
		//int precision=6;
		int time=observations.length;
		int n=A.length;
		double sum;
		double[] c=new double[time];



		double[][] alpha = new double[time][n];
		//computing alpha[0][i]
		c[0]=0;
		for (int i=0; i<n; i++)
		{
			alpha[0][i]=B[i][observations[0]]*pi[i];
			c[0]+=alpha[0][i];
		}

		c[0]=1/(c[0] + stabilizer); //stabilizer helps with underflow



		for (int i=0; i<n; i++)
		{
			alpha[0][i]=c[0]*alpha[0][i];
		}
		//computing alpha[t][i]
		for (int t=1; t<time; t++)
		{
			c[t]=0;
			for (int i=0; i<n; i++)
			{
				alpha[t][i]=0;
				for (int j=0; j<n; j++)
				{
					alpha[t][i]+=A[j][i]*alpha[t-1][j];
				}
				alpha[t][i]=B[i][observations[t]]*alpha[t][i];
				c[t]+=alpha[t][i];
			}
			c[t]=1/(c[t]+stabilizer);
			for (int i=0; i<n; i++)
			{
				alpha[t][i]=c[t]*alpha[t][i];
			}

		}

		Object[] result= new Object[2];

		result[0]=alpha;
		result[1]=c;
		return result;
	}



	/**
	 * Scaled implementation of backwards algorithm or betapass.
	 */
	public double[][] scaledBackwardsAlgorithm(int[] observations,double[] c)
	{
		int time=observations.length;
		int n=A.length;



		double[][] result = new double[time][n];
		//compute beta[0][i]
		for (int i=0; i<n; i++)
		{
			result[time-1][i]=c[time-1];
		}

		for (int t=time-2; t>=0; t--)
		{
			for (int i=0; i<n; i++)
			{
				result[t][i]=0;
				for (int j=0; j<n; j++)
				{
					result[t][i]+=A[i][j]*B[j][observations[t+1]]*result[t+1][j];
				}

				result[t][i]=result[t][i]*c[t];
			}

		}

		return result;
	}

	/**
	 * Round function. Uses Big Decimal for better precision.
	 * Rounding occurs the same way we learn in school.
	 * Costly (timewise) function due to BigDecimal.
	 * Rounds the number value, keeping "places" decimals.
	 */
	 public static double round(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();
	    if (value==Double.NEGATIVE_INFINITY)
	    {
	    	return value;
	    }
	    else
	    {
	    	BigDecimal bd = BigDecimal.valueOf(value);
	        bd = bd.setScale(places, RoundingMode.HALF_UP); //half up ensures that for example 1.5
																													//gets rounded to 2.0 and not 1.0
	        return bd.doubleValue();
	    }

	}
	/**
	 * Viterbi algorithm. Computes the most likely sequence of hidden states
	 * that produced this series of observations.
	 * It was implementeed in Log Space to avoid underflow.
	 */
	public int[] viterbi(int[] observations)
	{
		int time=observations.length;
		int n=A.length;
		double max;

		int[] result = new int[time]; // result[t]=k iff X_t=k

		double[][] delta= new double[time][n];
		int[][] delta_idx=new int[time][n];
		for (int i=0; i<n; i++)
		{

	        delta[0][i]=Math.log(B[i][observations[0]]) + Math.log(pi[i]);

		}



		for (int t=1; t<time; t++)
		{
			for (int i=0; i<n; i++)
			{

				max= Double.NEGATIVE_INFINITY; // we initialized max as -inf.
				for (int j=0; j<n; j++)
				{
					//searching for the max
					double candidate=delta[t-1][j] + Math.log(A[j][i]) + Math.log(B[i][observations[t]]);

					if (Double.compare(candidate, max) > 0) // if we found a larger number, keep it!
					{
	          max=candidate;
						delta_idx[t][i]=j; //keep the index for delta_idx (argmax)
					}
				}
				delta[t][i]=max;

			}

		}

		max = Double.NEGATIVE_INFINITY;
		//find he max again from delta[time-1]
		for (int j=0; j<n; j++)
		{
			if (delta[time-1][j]>max)
			{
				result[time-1]=j;
				max=delta[time-1][j]; //X*_T = argmax(delta[T])
			}
	    }
		//trace back
		for (int t=time-2; t>=0; t--)
		{
			result[t]=delta_idx[t+1][result[t+1]];
		}

		return result;
	}
	/**
	 * Learn parameters alpha,beta,c,gamma,digamma using scaled implementations,
	 * that correspond to the series of observations.
	 */
	public Object[] scaledlearnParameters(int[] observations)
	{
		int time = observations.length;
		int n=A.length;

		//retrieve alpha,beta,c
		Object[] a_pass= scaledforwardAlgorithm(observations);
		double[][] alpha = (double[][])a_pass[0];
		double[] c=(double[])a_pass[1];
		double[][] beta = scaledBackwardsAlgorithm(observations,c);

		//calculate gamma,digamma
		double[][][] di_gamma = new double[time-1][n][n];
		double[][] gamma= new double[time][n];

		for (int t=0; t<time-1; t++)
		{
			for (int i=0; i<n; i++)
			{
				gamma[t][i]=0;
				for (int j=0; j<n; j++)
				{
					//no need for scaling, we already scaled alpha,beta!
					di_gamma[t][i][j]=alpha[t][i]*A[i][j]*B[j][observations[t+1]]*beta[t+1][j];
					gamma[t][i]+=di_gamma[t][i][j];
				}


			}
		}
		// special case for last timestep.
		for (int i=0; i<n; i++)
		{
			gamma[time-1][i]=alpha[time-1][i];
		}


		Object[] result = new Object[5];
		result[0]=alpha;
		result[1]=beta;
		result[2]=di_gamma;
		result[3]=gamma;
		result[4]=c;
		return result;
	}
	/**
	 * HMM training using scaled imlementations.
	 * Parameter "usePrevious" is true iff we use our previous pi, A,B. Otherwise
	 * we intialized our parameteres with almost uniform distribution.
	 * The training runs for at most "iter" iterations, and corresponds
	 * "observations" series of observations.
	 */
	public void scaledtraining(boolean usePrevious,int[] observations, int iter)
	{
		int time = observations.length;
		int n=A.length;
		int k=B[0].length;
		// If we dont use existing parameters, Randomize!
		if (usePrevious==false)
		{
			A=initializeMatrixr(n,n);
			B=initializeMatrixr(n,k);
			pi=initializeMatrixr(1,n)[0];
		}



		int it=0;
		double oldLogProb=Double.NEGATIVE_INFINITY;
		double[] newpi=new double[n];
		double[][] newA=new double[n][n];
		double[][] newB=new double[n][k];
		while (it<iter) // Let the training commence! Stop the training after iter steps.
		{
			//get parameters
			Object[] parameters= scaledlearnParameters(observations);
			double[][] alpha = (double[][])parameters[0];
			double[][] beta = (double[][])parameters[1];
			double[][][] di_gamma = (double[][][])parameters[2];
			double[][] gamma = (double[][])parameters[3];
			double[] c= (double[])parameters[4];


			//pi
			newpi=gamma[0];

			double denom=0;
			double numer=0;
			for (int i=0; i<n; i++)
			{
				denom=0;
				for (int t=0; t<time-1; t++)
				{
					denom+=gamma[t][i];
				}

				for (int j=0; j<n; j++)
				{
					numer=0;
					for (int t=0; t<time-1; t++)
					{
						numer+=di_gamma[t][i][j];
					}

					newA[i][j]=numer/(denom+stabilizer); //stabilizer here is crucial to avoid 0/0!
				}


			}
			//B
			for (int i=0; i<n; i++)
			{
				denom=0;
				for (int t=0; t<time; t++)
				{
					denom+=gamma[t][i];


				}


				for (int j=0; j<k; j++)
				{
					numer=0;
					for (int t=0; t<time; t++)
					{
						if (observations[t]==j)
						{numer+=gamma[t][i];}
					}


					newB[i][j]=numer/(denom+stabilizer); //sstabilizer here is crucial to avoid 0/0!

				}
			}
			double logprob=0;
			//we calculate log P(O|pi,A,B) for stopping criteria. We prefer log to avoid underflow
			for (int t=0; t<time; t++)
			{
				logprob+=Math.log(c[t]);
			}
			logprob=-logprob;
			it++;
			//System.err.println("LogProd:" + logprob);
			if (logprob>oldLogProb) // if we have not yet found a local maxima, stop!
			{

				oldLogProb=logprob;
				A=newA;
				B=newB;
				pi=newpi;
			}

			else {break;}// if we found a (local) maximum, stop!
		}





	}
/**
* Average several HMM that belong on an hmmList. This method is used in Player class to
* average several HMM models used for predicting Species. The resulting HMM as an
* ensemble method. Useful as a regulirazation technique to avoid overfiting overfiting
* over limited observations.
*/
	public static HMM getAverage(ArrayList<HMM> hmmList)
	{
		double[] piRes=new double[states];
		double[][] ARes=new double[states][states];
		double[][] BRes=new double[states][Constants.COUNT_MOVE];

		for (int i=0; i<piRes.length; i++)
		{
			double res=0;
			for (int j=0; j<hmmList.size(); j++)
			{
				res+=hmmList.get(j).getPi()[i];
			}

			piRes[i]=res/hmmList.size();
		}

		for (int i=0; i<ARes.length; i++)
		{
			for (int j=0; j<ARes[0].length; j++)
			{
				double res=0;
				for (int k=0; k<hmmList.size(); k++)
				{
					res+=hmmList.get(k).getA()[i][j];
				}

				ARes[i][j]=res/hmmList.size();
			}

		}
		for (int i=0; i<BRes.length; i++)
		{
			for (int j=0; j<BRes[0].length; j++)
			{
				double res=0;
				for (int k=0; k<hmmList.size(); k++)
				{
					res+=hmmList.get(k).getB()[i][j];
				}

				BRes[i][j]=res/hmmList.size();
			}

		}

		HMM result = new HMM(piRes,ARes,BRes);

		return result;

	}



}
