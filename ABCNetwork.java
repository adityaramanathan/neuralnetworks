import java.io.*;

/**
 * @author Aditya Ramanathan
 * @date February 27, 2026
 * This class represents a simple feedforward neural network with an A–B–C architecture, 
 * consisting of an input layer with A neurons, one hidden layer with B neurons, and an
 * output layer with C neurons. There is functionality to simply just run the network on a 
 * set of test cases or both train and then run the network. The training method used to 
 * minimize the error function in this network is gradient descent. 
 */
public class ABCNetwork
{

/**
 * Defining the network configuration parameters.
 */
   private static int numInputNodes;         // number of input activations
   private static int hiddenLayerNumNodes;   // number of activations in the hidden layer
   private static int numOutputNodes;        // number of output activations
   private static boolean isTraining;        // flag to show whether network should be trained
   private static boolean showInputTable;    // flag for showing the input table at the end of running
   private static boolean showTruthTable;    // flag for showing the truth table at the end of running
   private static boolean saveFinalWeights;  // flag for whether weights should be saved at the end of running
   private static String getWeights;         // "Random" or "Load" or "Manual"
   private static String loadFileName;       // name of the file to load weights
   private static String saveFileName;       // name of the file to save weights
   private static double randomLowBound;     // random number generator low bound
   private static double randomHighBound;    // random number generator high bound
   private static int maximumIter;           // maximum number of iterations
   private static double idealErr;           // stop iterating after reaching this error value
   private static double lambda;             // learning factor
   private static int numTestCases;          // number of test cases
   private static String activationFunc;     // the activation function that is to be used

/**
 * Defining the arrays required within the network.
 */
   private static double[][] inputTable;
   private static double[][] truthTable;
   private static double[] inputActivations;
   private static double[] hiddenActivations;
   private static double[] outputActivations;
   private static double[] targetOutputs;
   private static double[][] weightsKJ;
   private static double[][] weightsJI;
   private static double[] thetaJ;
   private static double[] thetaI;
   private static double[] psiI;
   private static double[][] deltaWKJ;
   private static double[][] deltaWJI;
   private static double[][] outputActivationsRun;

/**
 * Defining the constants required within the network.
 */
   private static double averageError;
   private static int numIter;
   private static boolean maxIterationsComplete;
   private static boolean errorThresholdReached;
   private static long startTime;
   private static long endTime;

/**
 * Defines all the configuration parameters for the network.
 */
   public static void setConfigurationParams()
   {
      numInputNodes = 2;
      hiddenLayerNumNodes = 5;
      numOutputNodes = 3;
      isTraining = true;
      showInputTable = true;
      showTruthTable = true;
      saveFinalWeights = false;
      getWeights = "Random";
      loadFileName = "weights.bin";
      saveFileName = "weights.bin";
      randomLowBound = 0.1;
      randomHighBound = 1.5;
      maximumIter = 100000;
      idealErr = 0.0002;
      lambda = 0.3;
      numTestCases = 4;
      activationFunc = "Sigmoid";
   } // public static void setConfigurationParams()

/**
 * Populates the input table for the network to train or run. 
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
 * Populates the truth table for the network to train, if and only if training the network is 
 * desired based on the configuration of the network.
 */
   public static void populateTruthTable()
   {
      truthTable[0][0] = 0.0;
      truthTable[1][0] = 0.0;
      truthTable[2][0] = 0.0;
      truthTable[3][0] = 1.0;
      truthTable[0][1] = 0.0;
      truthTable[1][1] = 1.0;
      truthTable[2][1] = 1.0;
      truthTable[3][1] = 1.0;
      truthTable[0][2] = 0.0;
      truthTable[1][2] = 1.0;
      truthTable[2][2] = 1.0;
      truthTable[3][2] = 0.0;
   } // public static void populateTruthTable()

/**
 * Sets the weights for the network to specific manually entered values.
 */
   public static void setManualWeights()
   {
      weightsKJ[0][0] = 0.5;
      weightsKJ[1][0] = 0.5;
      weightsKJ[0][1] = 0.5;
      weightsKJ[1][1] = 0.5;
      weightsJI[0][0] = 0.5;
      weightsJI[1][0] = 0.5;
   } // public static void setManualWeights()

/**
 * Always outputs ALL the relevant user specified information PRIOR to training or running.
 * This includes the network configuration given the number of activations in the network 
 * layers, what will be printed (input table, truth table, etc...), from where the weights 
 * are being taken, etc... If the network is training, it will output the random number 
 * range, the maximum number of iterations, the error threshold, and the value of the 
 * learning rate, lambda.
 */
   public static void echoConfigurationParams()
   {
      System.out.println("Network Configuration: " + numInputNodes + "-" + hiddenLayerNumNodes + "-" + numOutputNodes);

      if (showInputTable)
      {
         System.out.println("The input table will be printed");
      }

      if (showTruthTable)
      {
         System.out.println("The truth table will be printed");
      }

      if (isTraining)
      {
         System.out.println("The network will train first and then run");
         System.out.println("Random Weights Range: (" + randomLowBound + ", " + randomHighBound + ")");
         System.out.println("Learning Rate: " + lambda);
         System.out.println("Maximum Number of Iterations: " + maximumIter);
         System.out.println("Error Threshold: " + idealErr);
      }

      if (saveFinalWeights)
      {
         System.out.println("The final weights will be saved to " + saveFileName);
      }

      System.out.println("The weights are obtained from " + getWeights);

      if (getWeights.equalsIgnoreCase("load"))
      {
         System.out.println("The weights will be loaded from " + loadFileName);
      }

      System.out.println("Activation Function: " + activationFunc);
      System.out.println("Number of Test Cases: " + numTestCases);
   } // public static void echoConfigurationParams()

/**
 * All major network array memory allocations for the network.
 */
   public static void allocateMem()
   {
      inputTable = new double[numTestCases][numInputNodes];
      truthTable = new double[numTestCases][numOutputNodes];
      inputActivations = new double[numInputNodes];
      hiddenActivations = new double[hiddenLayerNumNodes];
      outputActivations = new double[numOutputNodes];
      targetOutputs = new double[numOutputNodes];
      weightsKJ = new double[numInputNodes][hiddenLayerNumNodes];
      weightsJI = new double[hiddenLayerNumNodes][numOutputNodes];

      if (isTraining)
      {
         thetaJ = new double[hiddenLayerNumNodes];
         thetaI = new double[numOutputNodes];
         psiI = new double[numOutputNodes];
         deltaWKJ = new double[numInputNodes][hiddenLayerNumNodes];
         deltaWJI = new double[hiddenLayerNumNodes][numOutputNodes];
      }

      outputActivationsRun = new double[numTestCases][numOutputNodes];
   } // public static void allocateMem()

/**
 * Sets the weights of the network to randomly generated values within bounds provided
 * in the network configuration.
 */
   public static void randomlyGenerateWeights()
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
         for (int i = 0; i < numOutputNodes; i++)
         {
            weightsJI[j][i] = randomize(randomLowBound, randomHighBound);
         }
      }
   } // public static void randomlyGenerateWeights()

/**
 * Loads the weights for the network from the file whose name is provided in the network 
 * configuration.
 */
   public static void loadWeights() throws IOException
   {
      DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(loadFileName)));

      int loadK = in.readInt();
      int loadJ = in.readInt();
      int loadI = in.readInt();

      if (loadK != numInputNodes || loadJ != hiddenLayerNumNodes || loadI != numOutputNodes)
      {
         throw new IllegalArgumentException(String.format("Weight file configuration does not match network architecture of %d-%d-%d",
                                                          numInputNodes, hiddenLayerNumNodes, numOutputNodes));
      }

      for (int k = 0; k < numInputNodes; k++)
      {
         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            weightsKJ[k][j] = in.readDouble();
         }
      }

      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            weightsJI[j][i] = in.readDouble();
         }
      }

      in.close();
   } // public static void loadWeights() throws IOException

/**
 * Populates the weights either randomly, loading from a file, or manually based on the
 * configuration of the network.
 */
   public static void populateWeights() throws IOException
   {
      if (getWeights.equalsIgnoreCase("random"))
      {
         randomlyGenerateWeights();
      }
      else if (getWeights.equalsIgnoreCase("load"))
      {
         loadWeights();
      }
      else if (getWeights.equalsIgnoreCase("manual"))
      {
         setManualWeights();
      }
      else
      {
         throw new IllegalArgumentException("The configuration for populating weights is incorrect.");
      }
   } // public static void populateWeights() throws IOException

/**
 * Saves the weights to the given file name if training saving the weights is desired based
 * on the configuration of the network.
 */
   public static void saveWeights() throws IOException
   {
      DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(saveFileName)));

      out.writeInt(numInputNodes);
      out.writeInt(hiddenLayerNumNodes);
      out.writeInt(numOutputNodes);

      for (int k = 0; k < numInputNodes; k++)
      {
         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            out.writeDouble(weightsKJ[k][j]);
         }
      }

      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            out.writeDouble(weightsJI[j][i]);
         }
      }

      out.close();
   } // public void saveWeights() throws IOException

/**
 * Populates the input table, the truth table if training the network, and the specifies
 * initial values for the weights within the network.
 */
   public static void populateArrays() throws IOException
   {
      populateInputs();

      if (isTraining)
      {
         populateTruthTable();
      }

      populateWeights();
   } // public static void populateArrays() throws IOException

/**
 * Helper function for populateArrays() that generates a random double value within a
 * specified range.
 * @param low the lower bound of the range for the random number.
 * @param high the upper bound of the range for the random number.
 * @return the random number within the specified range.
 */
   public static double randomize(double low, double high)
   {
      return (low + (high - low) * Math.random());
   }

/**
 * Helper function to compute the error from the sum of omega_i values.
 * @return the error, given the sum of the omega_i values.
 */
   public static double errorFunction(double omega_i_sum)
   {
      return (0.5 * omega_i_sum);
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
 */
   public static void runForRunning()
   {
      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         double theta_j = 0.0;
         for (int k = 0; k < numInputNodes; k++)
         {
            theta_j += weightsKJ[k][j] * inputActivations[k];
         }

         hiddenActivations[j] = fActivation(theta_j);
      } // for (int j = 0; j < hiddenLayerNumNodes; j++)

      for (int i = 0; i < numOutputNodes; i++)
      {
         double theta_i = 0.0;
         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            theta_i += weightsJI[j][i] * hiddenActivations[j];
         }

         outputActivations[i] = fActivation(theta_i);
      } // for (int i = 0; i < numOutputNodes; i++)
   } // public static void runForRunning()

/**
 * Runs the network once given a specific training case while training the network. Computes 
 * the output activation given a specific set of input activations.
 */
   public static void runForTraining()
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
      } // for (int j = 0; j < hiddenLayerNumNodes; j++)

      for (int i = 0; i < numOutputNodes; i++)
      {
         double dotProductJI = 0.0;
         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            dotProductJI += weightsJI[j][i] * hiddenActivations[j];
         }

         thetaI[i] = dotProductJI;
         outputActivations[i] = fActivation(thetaI[i]);
      } // for (int i = 0; i < numOutputNodes; i++)
   } // public static void runForTraining()

/**
 * Trains the network, finding the error and computing the changes in weights that should
 * be made for the next case that is trained.
 * @param errorSum the running error total from all the cases trained so far.
 * @return the subsequent running error total after the next case is trained.
 */
   public static double train(double errorSum)
   {
      double omega_i_sum = 0.0;
      for (int i = 0; i < numOutputNodes; i++)
      {
         double omega_i = targetOutputs[i] - outputActivations[i];
         psiI[i] = omega_i * fPrimeActivation(thetaI[i]);

         for (int j = 0; j < hiddenLayerNumNodes; j++)
         {
            double grad_wji = -hiddenActivations[j] * psiI[i];
            deltaWJI[j][i] = -lambda * grad_wji;
         }

         omega_i_sum += (omega_i * omega_i);
      } // for (int i = 0; i < numOutputNodes; i++)

      for (int j = 0; j < hiddenLayerNumNodes; j++)
      {
         double omega_j = 0.0;
         for (int i = 0; i < numOutputNodes; i++)
         {
            omega_j += psiI[i] * weightsJI[j][i];
         }

         double psi_j = omega_j * fPrimeActivation(thetaJ[j]);

         for (int k = 0; k < numInputNodes; k++)
         {
            double grad_wkj = -inputActivations[k] * psi_j;
            deltaWKJ[k][j] = -lambda * grad_wkj;
         }
      } // for (int j = 0; j < hiddenLayerNumNodes; j++)

      return (errorSum + errorFunction(omega_i_sum));
   } // public static double train(double errorSum)

/**
 * Updates the weights after the change for each weight is computed in the train() method.
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
         for (int i = 0; i < numOutputNodes; i++) 
         {
            weightsJI[j][i] += deltaWJI[j][i];
         }
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
            for (int i = 0; i < numOutputNodes; i++)
            {
               targetOutputs[i] = truthTable[t][i];
            }

            defineInputActivations(t);
            runForTraining();
            errorTotal = train(errorTotal);
            updateWeights();
         } // for (int t = 0; t < numTestCases; t++)

         averageError = errorTotal/(double)numTestCases;
         numIter++;
         maxIterationsComplete = numIter >= maximumIter;
         errorThresholdReached = averageError < idealErr;
      } // while (!maxIterationsComplete && !errorThresholdReached)
   } // public static void trainNetwork()

/**
 * Runs the network given a set of test cases.
 */
   public static void runNetwork()
   {
      for (int t = 0; t < numTestCases; t++)
      {
         defineInputActivations(t);
         runForRunning();

         for (int i = 0; i < numOutputNodes; i++)
         {
            outputActivationsRun[t][i] = outputActivations[i];
         }
      } // for (int t = 0; t < numTestCases; t++)
   } // public static void runNetwork()

/**
 * Prints the results from training and or running the network.
 */
   public static void printRunningResults()
   {
      System.out.println("The time it took for training/running was " + (endTime - startTime) + "ms");

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
            for (int i = 0; i < numOutputNodes; i++)
            {
               System.out.print(truthTable[t][i] + " ");
            }
            System.out.println();
         }
      } // if (isTraining && showTruthTable)
      
      System.out.println("Results from Running: ");

      for (int t = 0; t < numTestCases; t++)
      {
         for (int k = 0; k < numInputNodes; k++)
         {
            System.out.print(inputTable[t][k] + " ");
         }

         for (int i = 0; i < numOutputNodes; i++)
         {
            System.out.print(outputActivationsRun[t][i] + " ");
         }
         System.out.println();
      } // for (int t = 0; t < numTestCases; t++)
   } // public static void printRunningResults()

/**
 * Executes the following methods in order: (1) Sets the Configuration Parameters, (2) Prints 
 * the relevant information regarding the configuration parameters, (3) Allocates memory for
 * the important arrays in the network, (4) Populates the arrays, (5) Trains the network if
 * that was set in the configuration parameters and outputs training results, (6) Runs the 
 * network, (7) Saves the final weights that were successful, (8) Outputs running results. 
 * @param args arguments from the command line.
 */
   public static void main(String[] args) throws IOException
   {
      setConfigurationParams();
      echoConfigurationParams();
      allocateMem();
      populateArrays();

      startTime = System.currentTimeMillis();

      if (isTraining)
      {
         trainNetwork();
         printTrainingExitInfo();
      }

      runNetwork();

      endTime = System.currentTimeMillis();

      if (saveFinalWeights)
      {
         saveWeights();
      }

      printRunningResults();
   } // public static void main(String[] args) throws IOException
} // public class ABCNetwork