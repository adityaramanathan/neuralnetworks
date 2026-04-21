import java.io.*;

/**
 * @author Aditya Ramanathan
 * @date April 17, 2026
 * This class represents a simple feedforward neural network that allows for any number
 * of layers with any number of nodes in each layer. There is functionality to simply
 * just run the network on a set of test cases. There is also functionality to both train
 * the network on a set of test cases and then run the network on a set of test cases. 
 * The training method used to minimize the error function in this network is gradient 
 * descent with backpropagation.
 * 
 * Table of Contents:
 * public static void loadConfigParams(String filename)
 * public static void loadInputTable()
 * public static void loadTruthTable()
 * public static void setDerivedQuantities()
 * public static void echoConfigurationParams()
 * public static void allocateMem()
 * public static void randomlyGenerateWeights()
 * public static void loadWeights() throws IOException
 * public static void populateWeights() throws IOException
 * public static void saveWeights() throws IOException
 * public static void populateArrays() throws IOException
 * public static double randomize(double low, double high)
 * public static double fActivation(double theta)
 * public static double fPrimeActivation(double theta)
 * public static double sigmoid(double x)
 * public static double sigmoidPrime(double x)
 * public static double hyperbolicTangent(double x)
 * public static double hyperbolicTangentPrime(double x)
 * public static void defineInputActivations(int testCase)
 * public static void runForRunning()
 * public static double runForTraining()
 * public static void train()
 * public static void printTrainingExitInfo()
 * public static void trainNetwork()
 * public static void runNetwork()
 * public static void printRunningResults()
 * public static void main(String[] args) throws IOException
 */
public class NLayerNetwork
{

/**
 * Defining the network configuration parameters.
 */
   private static int numActivationLayers;   // number of activation layers in the network
   private static int[] nLayers;             // number of nodes in each layer
   private static String inputFileName;      // name of the file containing the input table
   private static String truthFileName;      // name of the file containing the output table
   private static boolean isTraining;        // flag to show whether network should be trained
   private static boolean showInputTable;    // flag for showing the input table at the end of running
   private static boolean showTruthTable;    // flag for showing the truth table at the end of running
   private static boolean saveFinalWeights;  // flag for whether weights should be saved at the end of running
   private static String getWeights;         // "Random" or "Load"
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
   private static double[][] inputTable;           // the input table for the network to train or run
   private static double[][] truthTable;           // the truth table for the network to train
   private static double[] targetOutputs;          // the target output activations for a specific training case
   private static double[][] a;                    // the activations for all layers in the network
   private static double[][][] w;                  // the weights for all layers in the network
   private static double[][] theta;                // the theta values for the network
   private static double[][] psi;                  // the psi values for the network
   private static double[][] outputActivationsRun; // the output activations for all test cases when running the network

/**
 * Defining the constants required within the network.
 */
   private static double averageError;
   private static int numIter;
   private static boolean maxIterationsComplete;
   private static boolean errorThresholdReached;
   private static long startTime;
   private static long endTime;
   private static int outputLayerIndex;
   private static int lastHiddenLayerIndex;

   private static final String DEFAULT_CONFIG_FILE_NAME = "config.txt";
   private static final int INPUT_LAYER_INDEX = 0;
   private static final int FIRST_HIDDEN_LAYER_INDEX = 1;
   private static final int SECOND_HIDDEN_LAYER_INDEX = 2;


/**
 * Defines all the configuration parameters for the network by loading them from a file.
 * @param filename the name of the file from which the parameters will be loaded.
 */
   public static void loadConfigParams(String filename)
   {
      try (BufferedReader br = new BufferedReader(new FileReader(filename))) 
      {
         String line;

         line = br.readLine();
         int numConnectivityLayers = Integer.parseInt(line.substring(0, line.indexOf(';')).trim());
         numActivationLayers = numConnectivityLayers + 1;

         line = br.readLine();
         String configurationString = line.substring(0, line.indexOf(';')).trim();
         String[] numNodesPerLayer = configurationString.split("\\s+");

         nLayers = new int[numActivationLayers];

         for (int n = 0; n < numActivationLayers; n++)
         {
            nLayers[n] = Integer.parseInt(numNodesPerLayer[n]);
         }

         line = br.readLine();
         inputFileName = line.substring(0, line.indexOf(';')).trim();

         line = br.readLine();
         truthFileName = line.substring(0, line.indexOf(';')).trim();

         line = br.readLine();
         isTraining = Boolean.parseBoolean(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         showInputTable = Boolean.parseBoolean(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         showTruthTable = Boolean.parseBoolean(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         saveFinalWeights = Boolean.parseBoolean(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         getWeights = line.substring(0, line.indexOf(';')).trim();

         line = br.readLine();
         loadFileName = line.substring(0, line.indexOf(';')).trim();

         line = br.readLine();
         saveFileName = line.substring(0, line.indexOf(';')).trim();

         line = br.readLine();
         randomLowBound = Double.parseDouble(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         randomHighBound = Double.parseDouble(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         maximumIter = Integer.parseInt(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         idealErr = Double.parseDouble(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         lambda = Double.parseDouble(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         numTestCases = Integer.parseInt(line.substring(0, line.indexOf(';')).trim());

         line = br.readLine();
         activationFunc = line.substring(0, line.indexOf(';')).trim();
      } // try (BufferedReader br = new BufferedReader(new FileReader(filename)))
      catch (IOException | NumberFormatException e)
      {
         System.out.println("Configuration file not formatted as expected.");
         e.printStackTrace();
      } // catch (IOException | NumberFormatException e)
   } // public static void loadConfigParams(String filename)

/**
 * Populates the input table for the network to train or run by reading a binary input 
 * file. The file name is provided in the network configuration. 
 */
   public static void loadInputTable()
   {
      try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(inputFileName))))
      {
         for (int row = 0; row < numTestCases; row++)
         {
            for (int col = 0; col < nLayers[INPUT_LAYER_INDEX]; col++) 
            {
               inputTable[row][col] = in.readDouble();
            }
         }
      } // try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(inputFileName))))
      catch (IOException e)
      {
         System.out.println("Input table binary file not formatted as expected.");
         e.printStackTrace();
      } // catch (IOException e)
   } // public static void loadInputTable()

/**
 * Populates the truth table for the network to train, if and only if training the network is 
 * desired, by reading a text file with values separated by a space. The file name is provided 
 * in the network configuration. 
 */
   public static void loadTruthTable()
   {
      try (BufferedReader br = new BufferedReader(new FileReader(truthFileName))) 
      {
         String line;
         int row = 0;

         while ((line = br.readLine()) != null && row < numTestCases)
         {
            line = line.trim();
            String[] values = line.split(" ");

            for (int col = 0; col < nLayers[outputLayerIndex]; col++)
            {
               truthTable[row][col] = Double.parseDouble(values[col]);
            }

            row++;
         } // while ((line = br.readLine()) != null && row < numTestCases)

         br.close();
      } // try (BufferedReader br = new BufferedReader(new FileReader(truthFileName)))
      catch (IOException e)
      {
         System.out.println("Truth table file not formatted as expected.");
         e.printStackTrace();
      } // catch (IOException e)
   } // public static void loadTruthTable()

/**
 * Computes the derived quantities needed for the network, specifically the index of the
 * output layer and the index of the last hidden layer in the network. 
 */
   public static void setDerivedQuantities()
   {
      outputLayerIndex = numActivationLayers - 1;
      lastHiddenLayerIndex = outputLayerIndex - 1;
   }

/**
 * Always outputs all the relevant user specified information prior to training or running.
 * This includes the network configuration given the number of nodes in the network 
 * layers, the name of the file containing the input table/truth table, what will be printed 
 * (input table, truth table, etc...), from where the weights are being taken, etc... If the 
 * network is training, it will output the name of the file containing the truth table, the 
 * random number range, the maximum number of iterations for training, the error threshold, 
 * and the value of the learning rate, lambda.
 */
   public static void echoConfigurationParams()
   {
      String networkConfig = "";
      for (int n = 0; n < numActivationLayers; n++)
      {
         networkConfig += nLayers[n];

         if (n != outputLayerIndex)
         {
            networkConfig += "-";
         }
      } // for (int n = 0; n < numActivationLayers; n++)

      System.out.println("Network Configuration: " + networkConfig);
      System.out.println("Input table loaded from " + inputFileName);

      if (showInputTable)
      {
         System.out.println("The input table will be printed");
      }

      if (isTraining)
      {
         System.out.println("Truth table loaded from " + truthFileName);

         if (showTruthTable)
         {
            System.out.println("The truth table will be printed");
         }

         System.out.println("The network will train first and then run");
         System.out.println("Random Weights Range: (" + randomLowBound + ", " + randomHighBound + ")");
         System.out.println("Learning Rate: " + lambda);
         System.out.println("Maximum Number of Iterations: " + maximumIter);
         System.out.println("Error Threshold: " + idealErr);
      } // if (isTraining)

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
      inputTable = new double[numTestCases][nLayers[INPUT_LAYER_INDEX]];
      targetOutputs = new double[nLayers[outputLayerIndex]];

      a = new double[numActivationLayers][];
      w = new double[numActivationLayers - 1][][];

      for (int n = 0; n < numActivationLayers; n++)
      {
         a[n] = new double[nLayers[n]];
      }

      for (int n = 0; n < outputLayerIndex; n++)
      {
         w[n] = new double[nLayers[n]][nLayers[n + 1]];
      }

      if (isTraining)
      {
         truthTable = new double[numTestCases][nLayers[outputLayerIndex]];

         theta = new double[numActivationLayers - 1][];
         psi = new double[numActivationLayers][];

         for (int n = FIRST_HIDDEN_LAYER_INDEX; n < outputLayerIndex; n++)
         {
            theta[n] = new double[nLayers[n]];
         }

         for (int n = SECOND_HIDDEN_LAYER_INDEX; n < numActivationLayers; n++)
         {
            psi[n] = new double[nLayers[n]];
         }
      } // if (isTraining)

      outputActivationsRun = new double[numTestCases][nLayers[outputLayerIndex]];
   } // public static void allocateMem()

/**
 * Sets the weights of the network to randomly generated values within bounds provided
 * in the network configuration.
 */
   public static void randomlyGenerateWeights()
   {
      for (int n = INPUT_LAYER_INDEX; n < outputLayerIndex; n++)
      {
         for (int k = 0; k < nLayers[n]; k++)
         {
            for (int j = 0; j < nLayers[n + 1]; j++)
            {
               w[n][k][j] = randomize(randomLowBound, randomHighBound);
            }
         }
      } // for (int n = INPUT_LAYER_INDEX; n < outputLayerIndex; n++)
   } // public static void randomlyGenerateWeights()

/**
 * Loads the weights for the network from the file whose name is provided in the network 
 * configuration.
 */
   public static void loadWeights() throws IOException
   {
      DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(loadFileName)));

      for (int n = INPUT_LAYER_INDEX; n < numActivationLayers; n++)
      {
         int currLayerSize = in.readInt();
         if (currLayerSize != nLayers[n])
         {
            throw new IllegalArgumentException(String.format("Weight file configuration mismatch at layer %d. " +
                                                             "Expected %d but found %d.", n, nLayers[n], currLayerSize));
         }
      } // for (int n = INPUT_LAYER_INDEX; n < numActivationLayers; n++)

      for (int n = INPUT_LAYER_INDEX; n < outputLayerIndex; n++)
      {
         for (int k = 0; k < nLayers[n]; k++)
         {
            for (int j = 0; j < nLayers[n + 1]; j++)
            {
               w[n][k][j] = in.readDouble();
            }
         }
      } // for (int n = INPUT_LAYER_INDEX; n < outputLayerIndex; n++)

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
      else
      {
         throw new IllegalArgumentException("The configuration for populating weights is invalid.");
      }
   } // public static void populateWeights() throws IOException

/**
 * Saves the weights to the file (provided in the network's configuration) if the saving the 
 * weights is desired (also provided in the network's configuration).
 */
   public static void saveWeights() throws IOException
   {
      DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(saveFileName)));

      for (int n = INPUT_LAYER_INDEX; n < numActivationLayers; n++)
      {
         out.writeInt(nLayers[n]);
      }

      for (int n = INPUT_LAYER_INDEX; n < outputLayerIndex; n++)
      {
         for (int k = 0; k < nLayers[n]; k++)
         {
            for (int j = 0; j < nLayers[n + 1]; j++)
            {
               out.writeDouble(w[n][k][j]);
            }
         }
      } // for (int n = INPUT_LAYER_INDEX; n < outputLayerIndex; n++)

      out.close();
   } // public void saveWeights() throws IOException

/**
 * Populates the input table, the truth table if training the network, and populates
 * initial values for the weights within the network.
 */
   public static void populateArrays() throws IOException
   {
      loadInputTable();

      if (isTraining)
      {
         loadTruthTable();
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
 * Outputs f(x) where f is the hyperbolic tangent function.
 * @param x the input to the hyperbolic tangent function.
 * @return the hyperbolic tangent value corresponding to x.
 */
   public static double hyperbolicTangent(double x)
   {
      double epsilon = (x < 0) ? 1 : -1;
      double exponential = Math.exp(epsilon * 2.0 * x);
      return epsilon * ((exponential - 1.0) / (exponential + 1.0));
   }

/**
 * Outputs f'(x) where f' is the derivative of the hyperbolic tangent function.
 * @param x the input to the hyperbolic tangent function.
 * @return the derivative of hyperbolic tangent value corresponding to x.
 */
   public static double hyperbolicTangentPrime(double x)
   {
      double hyperbolicTangentVal = hyperbolicTangent(x);
      return (1.0 - hyperbolicTangentVal * hyperbolicTangentVal);
   }

/**
 * Populates the inputActivations array for a specific test case.
 * @param testCase the index of the test case that wants to be trained or run.
 */
   public static void defineInputActivations(int testCase)
   {
      int n = INPUT_LAYER_INDEX;
      for (int m = 0; m < nLayers[n]; m++)
      {
         a[n][m] = inputTable[testCase][m];
      }
   }

/**
 * Runs the network once given a configuration to only run the network. Computes the output 
 * activation given a specific set of input activations.
 */
   public static void runForRunning()
   {
      for (int n = FIRST_HIDDEN_LAYER_INDEX; n < numActivationLayers; n++)
      {
         for (int j = 0; j < nLayers[n]; j++)
         {
            double theta_j = 0.0;
            for (int k = 0; k < nLayers[n - 1]; k++)
            {
               theta_j += w[n - 1][k][j] * a[n - 1][k];
            }

            a[n][j] = fActivation(theta_j);
         } // for (int j = 0; j < nLayers[n]; j++)
      } // for (int n = FIRST_HIDDEN_LAYER_INDEX; n < numActivationLayers; n++)
   } // public static void runForRunning()

/**
 * Runs the network once given a specific training case while training the network. 
 * First, it computes the output activation given a specific set of input activations.
 * Then, it computes twice the error for efficiency, and it returns that value.
 * @return 2 * error for the particular training case.
 */
   public static double runForTraining()
   {
      for (int n = FIRST_HIDDEN_LAYER_INDEX; n < outputLayerIndex; n++)
      {
         for (int j = 0; j < nLayers[n]; j++)
         {
            double theta_j = 0.0;
            for (int k = 0; k < nLayers[n - 1]; k++)
            {
               theta_j += w[n - 1][k][j] * a[n - 1][k];
            }

            theta[n][j] = theta_j;
            a[n][j] = fActivation(theta[n][j]);
         } // for (int j = 0; j < nLayers[n]; j++)
      } // for (int n = FIRST_HIDDEN_LAYER_INDEX; n < outputLayerIndex; n++)

      double twiceError = 0.0;

      int n = outputLayerIndex;
      for (int i = 0; i < nLayers[n]; i++)
      {
         double theta_i = 0.0;
         for (int j = 0; j < nLayers[n - 1]; j++)
         {
            theta_i += w[n - 1][j][i] * a[n - 1][j];
         }

         a[n][i] = fActivation(theta_i);
         double omega_i = targetOutputs[i] - a[n][i];
         psi[n][i] = omega_i * fPrimeActivation(theta_i);
         twiceError += omega_i * omega_i;
      } // for (int i = 0; i < nLayers[n]; i++)
      
      return twiceError;
   } // public static double runForTraining()

/**
 * Trains the network by computing the changes in weights and updating the weights.
 */
   public static void train()
   {
      for (int n = lastHiddenLayerIndex; n >= SECOND_HIDDEN_LAYER_INDEX; n--)
      {
         for (int j = 0; j < nLayers[n]; j++)
         {
            double omega_j = 0.0;

            for (int i = 0; i < nLayers[n + 1]; i++)
            {
               omega_j += psi[n + 1][i] * w[n][j][i];
               w[n][j][i] += lambda * a[n][j] * psi[n + 1][i];
            }

            psi[n][j] = omega_j * fPrimeActivation(theta[n][j]);
         } // for (int j = 0; j < nLayers[n]; j++)
      } // for (int n = lastHiddenLayerIndex; n >= SECOND_HIDDEN_LAYER_INDEX; n--)

      int n = FIRST_HIDDEN_LAYER_INDEX;
      for (int k = 0; k < nLayers[n]; k++)
      {
         double omega_k = 0.0;

         for (int j = 0; j < nLayers[n + 1]; j++)
         {
            omega_k += psi[n + 1][j] * w[n][k][j];
            w[n][k][j] += lambda * a[n][k] * psi[n + 1][j];
         }

         double psi_k = omega_k * fPrimeActivation(theta[n][k]);

         for (int m = 0; m < nLayers[n - 1]; m++)
         {
            w[n - 1][m][k] += lambda * a[n - 1][m] * psi_k;
         }
      } // for (int k = 0; k < nLayers[n]; k++)
   } // public static void train()

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
      System.out.printf("The average error reached was %.4f.\n", averageError);
   } // public static void printTrainingExitInfo()

/**
 * Trains the network given a set of training data: input table and truth table.
 */
   public static void trainNetwork()
   {
      while (!maxIterationsComplete && !errorThresholdReached)
      {
         double twiceTotalError = 0.0;
         for (int t = 0; t < numTestCases; t++)
         {
            for (int i = 0; i < nLayers[outputLayerIndex]; i++)
            {
               targetOutputs[i] = truthTable[t][i];
            }

            defineInputActivations(t);
            twiceTotalError += runForTraining();
            train();
         } // for (int t = 0; t < numTestCases; t++)

         double totalError = 0.5 * twiceTotalError; 
         averageError = totalError / (double)numTestCases;

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

         int n = outputLayerIndex;
         for (int i = 0; i < nLayers[n]; i++)
         {
            outputActivationsRun[t][i] = a[n][i];
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
            for (int m = 0; m < nLayers[INPUT_LAYER_INDEX]; m++)
            {
               System.out.print(inputTable[t][m] + " ");
            }
            System.out.println();
         }
      } // if (showInputTable)

      if (isTraining && showTruthTable)
      {
         System.out.println("Truth Table:");
         for (int t = 0; t < numTestCases; t++)
         {
            for (int i = 0; i < nLayers[outputLayerIndex]; i++)
            {
               System.out.print(truthTable[t][i] + " ");
            }
            System.out.println();
         }
      } // if (isTraining && showTruthTable)
      
      System.out.println("Results from Running: ");

      for (int t = 0; t < numTestCases; t++)
      {
         for (int m = 0; m < nLayers[INPUT_LAYER_INDEX]; m++)
         {
            System.out.print(inputTable[t][m] + " ");
         }

         for (int i = 0; i < nLayers[outputLayerIndex]; i++)
         {
            System.out.printf("%.4f ", outputActivationsRun[t][i]);
         }
         System.out.println();
      } // for (int t = 0; t < numTestCases; t++)
   } // public static void printRunningResults()

/**
 * Does the following in order: (1) Sets and prints the name of the configuration file, 
 * (2) Loads the configuration parameters from that file, (3) Computes the derived quantities
 * for the network, (4) Prints the relevant information regarding the configuration parameters, 
 * (5) Allocates memory for the important arrays in the network, (6) Populates the network 
 * arrays, (7) Trains the network if that was set in the configuration parameters and outputs 
 * training results, (8) Runs the network, (9) If desired, saves the final weights that were 
 * successful, and (10) Outputs running results. 
 * @param args arguments from the command line.
 */
   public static void main(String[] args) throws IOException
   {
      String configFileName;
      if (args.length == 0)
      {
         configFileName = DEFAULT_CONFIG_FILE_NAME;
      }
      else
      {
         configFileName = args[0];
      }

      System.out.println("Configuration File Name: " + configFileName);

      loadConfigParams(configFileName);
      setDerivedQuantities();
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
} // public class NLayerNetwork