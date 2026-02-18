/**
 * @author Aditya Ramanathan
 * @date February 3, 2025
 * This class represents a simple feedforward neural network with an A–B–1 architecture, 
 * consisting of an input layer with A neurons, one hidden layer with B neurons, and a 
 * single output neuron. There is functionality to simply just run the network on a set
 * of test cases or both train and then run the network. The training method used to 
 * minimize the error function in this network is gradient descent. 
 */
public class AB1Network
{

/**
 * Defining the network configuration parameters.
 */
   private static int numInputNodes;         // number of input activations
   private static int hiddenLayerNumNodes;   // number of activations in the hidden layer
   private static boolean isTraining;        // flag to show whether network should be trained
   private static boolean showInputTable;    // flag for showing the input table at the end of running
   private static boolean showTruthTable;    // flag for showing the truth table at the end of running
   private static boolean isRandomWeights;   // flag for random weights or manual weights
   private static double randomLowBound;     // random number generator low bound
   private static double randomHighBound;    // random number generator high bound
   private static int maximumIter;           // maximum number of iterations
   private static double idealErr;           // stop iterating after reaching this error value
   private static double lambda;             // learning factor
   private static int numTestCases;          // number of test cases
   private static String booleanAlgProblem;  // "AND", "OR", "XOR"
   private static String activationFunc;     // the activation function that is to be used

/**
 * Defining the arrays required within the network.
 */
   private static double[][] inputTable;
   private static double[] truthTable;
   private static double[] inputActivations;
   private static double[] hiddenActivations;
   private static double[][] weightsKJ;
   private static double[] weightsJ0;
   private static double[] thetaJ;
   private static double[] omegaJ;
   private static double[] psiJ;
   private static double[] deltaWJ0;
   private static double[][] deltaWKJ;
   private static double[] outputActivationsRun;

/**
 * Defining the constants required within the network.
 */
   private static double targetOutput;
   private static double outputActivation;
   private static double theta0;
   private static double omega0;
   private static double psi0;
   private static double averageError;
   private static int numIter;
   private static boolean maxIterationsComplete;
   private static boolean errorThresholdReached;

/**
 * Defines all the configuration parameters for the network.
 */
   public static void setConfigurationParams()
   {
      numInputNodes = 2;
      hiddenLayerNumNodes = 2;
      isTraining = true;
      showInputTable = true;
      showTruthTable = true;
      isRandomWeights = true;
      randomLowBound = -1.5;
      randomHighBound = 1.5;
      maximumIter = 100000;
      idealErr = 0.0002;
      lambda = 0.3;
      numTestCases = 4;
      booleanAlgProblem = "OR";
      activationFunc = "Sigmoid";
   } // public static void setConfigurationParams()

/**
 * Always outputs ALL the relevant user specified information PRIOR to training or running.
 * This includes the network configuration given the number of activations in the network 
 * layers and the activation function. If the network is training, it will output the random 
 * number range, the maximum number of iterations, the error threshold, and the value of the
 * learning rate, lambda.
 */
   public static void echoConfigurationParams()
   {
      System.out.println("Network Configuration: " + numInputNodes + "-" + hiddenLayerNumNodes + "-1 " + booleanAlgProblem);
      System.out.println("Activation Function: " + activationFunc);

      if (isTraining)
      {
         System.out.println("Random Weights Range: (" + randomLowBound + ", " + randomHighBound + ")");
         System.out.println("Learning Rate: " + lambda);
         System.out.println("Maximum Number of Iterations: " + maximumIter);
         System.out.println("Error Threshold: " + idealErr);
      }

      System.out.println("Number of Test Cases: " + numTestCases);
   } // public static void echoConfiguratonParams()

/**
 * All major network array memory allocations for the A-B-1 network.
 */
   public static void allocateMem()
   {
      inputTable = new double[numTestCases][numInputNodes];
      truthTable = new double[numTestCases];
      inputActivations = new double[numInputNodes];
      hiddenActivations = new double[hiddenLayerNumNodes];
      weightsKJ = new double[numInputNodes][hiddenLayerNumNodes];
      weightsJ0 = new double[hiddenLayerNumNodes];
      thetaJ = new double[hiddenLayerNumNodes];

      if (isTraining)
      {
         omegaJ = new double[hiddenLayerNumNodes];
         psiJ = new double[hiddenLayerNumNodes];
         deltaWJ0 = new double[hiddenLayerNumNodes];
         deltaWKJ = new double[numInputNodes][hiddenLayerNumNodes];
      }

      outputActivationsRun = new double[numTestCases];
   } // public static void allocateMem()

/**
 * Populates the input table for the network. 
 */
   public static void populateInputs()
   {
      inputTable[0][0] = 0.0;
      inputTable[0][1] = 0.0;
      inputTable[1][0] = 0.0;
      inputTable[1][1] = 1.0;
      inputTable[2][0] = 1.0;
      inputTable[2][1] = 0.0;
      inputTable[3][0] = 1.0;
      inputTable[3][1] = 1.0;
   } // public static void populateInputs()

/**
 * Populates the truth table if training the network is desired based on the configuration
 * of the network.
 */
   public static void populateTruthtable()
   {
      if (booleanAlgProblem.equals("OR"))
      {
         truthTable[0] = 0.0;
         truthTable[1] = 1.0;
         truthTable[2] = 1.0;
         truthTable[3] = 1.0;
      }
      else if (booleanAlgProblem.equals("AND"))
      {
         truthTable[0] = 0.0;
         truthTable[1] = 0.0;
         truthTable[2] = 0.0;
         truthTable[3] = 1.0;
      }
      else if (booleanAlgProblem.equals("XOR"))
      {
         truthTable[0] = 0.0;
         truthTable[1] = 1.0;
         truthTable[2] = 1.0;
         truthTable[3] = 0.0;
      }
      else // for a boolean function that is not in the above, manually enter outputs
      {
         truthTable[0] = 0.0;
         truthTable[1] = 0.0;
         truthTable[2] = 0.0;
         truthTable[3] = 0.0;
      }
   } // public static void populateTruthTable()

/**
 * Populates the weights either randomly or with manually entered weights based on the
 * configuration of the network.
 */
   public static void populateWeights()
   {
      if (isRandomWeights)
      {
         for (int k = 0; k < numInputNodes; k++)
         {
            for (int j = 0; j < hiddenLayerNumNodes; j++)
            {
               weightsKJ[k][j] = randomize(randomLowBound, randomHighBound);
            }
         }

         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            weightsJ0[j] = randomize(randomLowBound, randomHighBound);
         }
      } // if (isRandomWeights)
      else
      {
         weightsKJ[0][0] = 0.5;
         weightsKJ[1][0] = 0.5;
         weightsKJ[0][1] = 0.5;
         weightsKJ[1][1] = 0.5;
         weightsJ0[0] = 0.5;
         weightsJ0[1] = 0.5;
      }
   } // public static void populateWeights()

/**
 * Populates the input table, the truth table if training the network, and the specifies 
 * initial values for the weights within the network. 
 */
   public static void populateArrays()
   {
      populateInputs();

      if (isTraining)
      {
         populateTruthtable();
      }

      populateWeights();
   } // public static void populateArrays()

/**
 * Helper function for populateArrays() that generates random values for the weights of 
 * the network. In other words, it generates a random double value within a specified 
 * range.
 * @param low the lower bound of the range for the random number.
 * @param high the upper bound of the range for the random number.
 * @return the random number within the specified range.
 */
   public static double randomize(double low, double high)
   {
      return (low + (high - low) * Math.random());
   }

/**
 * Helper function to compute the error from the value of omega0.
 * @return the error, given the value of omega0.
 */
   public static double errorFunction()
   {
      return (0.5 * omega0 * omega0);
   }

/**
 * Defines the activation function for the network as a simple sigmoid function. 
 * @param theta the sum of the dot product of the activations and the weights in a specific
 * layer of the network. 
 * @return the output when theta is passed into the sigmoid function. 
 */
   public static double fActivation(double theta)
   {
      return sigmoid(theta);
   }

/**
 * Defines the derivative of the activation function for the network as the derivative of a
 * simple sigmoid function. 
 * @param theta the sum of the dot product of the activations and the weights in a specific
 * layer of the network. 
 * @return the output when theta is passed into the derivative of the sigmoid function.
 */
   public static double fPrimeActivation(double theta)
   {
      return sigmoidPrime(theta);
   }

/**
 * Outputs f(x) where f is the simple sigmoid function.
 * @param x the input to the sigmoid function.
 * @return the sigmoid value corresponding to x.
 */
   public static double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
 * Outputs f'(x) where f' is the derivative of the simple sigmoid function.
 * @param x the input to the sigmoid function.
 * @return the derivative of sigmoid value corresponding to x.
 */
   public static double sigmoidPrime(double x)
   {
      double sigmoidVal = sigmoid(x);
      return sigmoidVal * (1.0 - sigmoidVal);
   }

/**
 * Populates the inputActivations array for a specific test case. 
 * @param testCase the index of the test case that wants to be trained or run.
 */
   public static void defineInputActivations(int testCase)
   {
      for (int k = 0; k < numInputNodes; k++)
      {
         inputActivations[k] = inputTable[testCase][k];
      }
   }

/**
 * Runs the network once given a configuration to only run the network. Computes the output 
 * activation given a specific set of input activations.
 * @return the output activation after running the network.
 */
   public static double runForRunning()
   {
      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         double dotProductKJ = 0.0;
         for (int k = 0; k < numInputNodes; k++)
         {
            dotProductKJ += weightsKJ[k][j] * inputActivations[k];
         }

         thetaJ[j] = dotProductKJ;
         hiddenActivations[j] = fActivation(thetaJ[j]);
      }
      
      double dotProductJ0 = 0.0;
      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         dotProductJ0 += weightsJ0[j] * hiddenActivations[j];
      }

      theta0 = dotProductJ0;
      return (fActivation(theta0));
   } // public static double runForRunning()

/**
 * Runs the network once given a specific training case while training the network. Computes 
 * the output activation given a specific set of input activations.
 * @return the output activation after running the network.
 */
   public static double runForTraining()
   {
      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         double dotProductKJ = 0.0;
         for (int k = 0; k < numInputNodes; k++)
         {
            dotProductKJ += weightsKJ[k][j] * inputActivations[k];
         }

         thetaJ[j] = dotProductKJ;
         hiddenActivations[j] = fActivation(thetaJ[j]);
      }
      
      double dotProductJ0 = 0.0;
      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         dotProductJ0 += weightsJ0[j] * hiddenActivations[j];
      }

      theta0 = dotProductJ0;
      return (fActivation(theta0));
   } // public static double runForTraining()

/**
 * Trains the network, finding the error and computing the changes in weights that should
 * be made for the next case that is trained.
 * @param errorSum the running error total from all the cases trained so far.
 * @return the subsequent running error total after the next case is trained.
 */
   public static double train(double errorSum)
   {
      omega0 = targetOutput - outputActivation;
      psi0 = omega0 * fPrimeActivation(theta0);

      for (int j = 0; j < hiddenLayerNumNodes; j++) 
      {
         deltaWJ0[j] = lambda * hiddenActivations[j] * psi0;
         omegaJ[j] = psi0 * weightsJ0[j];
         psiJ[j] = omegaJ[j] * fPrimeActivation(thetaJ[j]);

         for (int k = 0; k < numInputNodes; k++)
         {
            deltaWKJ[k][j] = lambda * inputActivations[k] * psiJ[j];
         }
      }
      return (errorSum + errorFunction());
   } // public static double train(double errorSum)

/**
 * Updates the weights after the updates are computed in the train() method. 
 */
   public static void updateWeights()
   {
      for (int k = 0; k < numInputNodes; k++) 
      {
         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            weightsKJ[k][j] += deltaWKJ[k][j];
         }
      }

      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         weightsJ0[j] += deltaWJ0[j];
      }
   } // public static void updateWeights()

/**
 * Always outputs the reason for the end of training, the number of iterations reached, and
 * the average error reached at the end of the training.
 */
   public static void printTrainingExitInfo()
   {
      if (maxIterationsComplete)
      {
         System.out.println("The training ended because the maximum iterations was reached.");
      }
      
      if (errorThresholdReached)
      {
         System.out.println("The training ended because the error threshold was reached.");
      }

      System.out.println("The number of iterations was " + numIter + ".");
      System.out.println("The average error reached was " + averageError + ".");
   } // public static void printTrainingExitInfo()

/**
 * Trains the network given a set of training data: input table and truth table. 
 */
   public static void trainNetwork()
   {
      while (!maxIterationsComplete && !errorThresholdReached)
      {
         double errorTotal = 0.0;
         for (int t = 0; t < numTestCases; t++)
         {
            targetOutput = truthTable[t];
            defineInputActivations(t);
            outputActivation = runForTraining();
            errorTotal = train(errorTotal);
            updateWeights();
         }

         averageError = errorTotal/(double)numTestCases;
         numIter++;
         maxIterationsComplete = numIter >= maximumIter;
         errorThresholdReached = averageError < idealErr;
      }
   } // public static void trainNetwork()

/**
 * Runs the network given a set of test cases. 
 */
   public static void runNetwork()
   {
      for (int t = 0; t < numTestCases; t++)
      {
         defineInputActivations(t);
         outputActivation = runForRunning();
         outputActivationsRun[t] = outputActivation;
      }
   } // public static void runNetwork()

/**
 * Prints the results from training and or running the network. 
 */
   public static void printRunningResults()
   {
      if (showInputTable)
      {
         System.out.println("Input Table:");
         for (int t = 0; t < numTestCases; t++)
         {
            for (int k = 0; k < numInputNodes; k++)
            {
               System.out.print(inputTable[t][k] + " ");
            }
            System.out.println();
         }
      } // if (showInputTable)

      if (isTraining && showTruthTable) 
      {
         System.out.println("Truth Table:");
         for (int t = 0; t < numTestCases; t++)
         {
            System.out.println(truthTable[t]);
         }
      } // if (isTraining && showTruthTable)
      
      for (int t = 0; t < numTestCases; t++)
      {
         System.out.println("Test Case " + (t + 1) + " Output: " + outputActivationsRun[t]);
      }
   } // public static void printRunningResults()

/**
 * Executes the following methods in order: (1) Sets the Configuration Parameters, (2) Prints 
 * the relevant information regarding the configuration parameters, (3) Allocates memory for
 * the important arrays in the network, (4) Populates the arrays, (5) Trains the network if
 * that was requested and outputs training results, (6) Runs the network and outputs running
 * results. 
 * @param args arguments from the command line.
 */
   public static void main(String[] args)
   {
      setConfigurationParams();
      echoConfigurationParams();
      allocateMem();
      populateArrays();

      if (isTraining)
      {
         trainNetwork();
         printTrainingExitInfo();
      }

      runNetwork();
      printRunningResults();
   } // public static void main(String[] args)
} // public class AB1Network