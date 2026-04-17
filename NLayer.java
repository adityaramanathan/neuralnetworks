import java.io.*;

/**
 * READ: This is the same code as ABCD Backprop currently. I've pushed to Github so I can
 * track the changes I'm making to NLayer and make sure I don't make errors.
 */

/**
 * @author Aditya Ramanathan
 * @date April 17, 2026
 * This class represents a simple feedforward N-layer neural network. There is functionality 
 * to simply just run the network on a set of test cases or both train and then run the 
 * network. The training method used to minimize the error function in this network is 
 * gradient descent with backpropagation.
 */
public class NLayer
{

/**
 * Defining the network configuration parameters.
 */
   private static int numActivationLayers;   // number of activation layers in the network
   private static int numInputNodes;         // number of input activations
   private static int numHiddenKNodes;       // number of activations in hidden layer K
   private static int numHiddenJNodes;       // number of activations in hidden layer J
   private static int numOutputNodes;        // number of output activations
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

   private static final String DEFAULT_CONFIG_FILE_NAME = "config.txt";
   private static final int INPUT_LAYER = 0;
   private static final int FIRST_HIDDEN_LAYER = 1;
   private static final int SECOND_HIDDEN_LAYER = 2;
   private static final int OUTPUT_LAYER = 3; // eventually need to delete this const.

/**
 * Defines all the configuration parameters for the network.
 */
   public static void setConfigurationParams()
   {
      numActivationLayers = 4;
      numInputNodes = 2;
      numHiddenKNodes = 5;
      numHiddenJNodes = 10;
      numOutputNodes = 3;
      inputFileName = "input.txt";
      truthFileName = "truth.txt";
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

         numInputNodes = Integer.parseInt(numNodesPerLayer[0]);
         numHiddenKNodes = Integer.parseInt(numNodesPerLayer[1]);
         numHiddenJNodes = Integer.parseInt(numNodesPerLayer[2]);
         numOutputNodes = Integer.parseInt(numNodesPerLayer[3]);

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
 * Populates the input table for the network to train or run by reading a binary input 
 * file. The file name is provided in the network configuration. 
 */
   public static void loadInputTable()
   {
      try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(inputFileName))))
      {
         for (int row = 0; row < numTestCases; row++) 
         {
            for (int col = 0; col < numInputNodes; col++) 
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

            for (int col = 0; col < numOutputNodes; col++)
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
 * Always outputs all the relevant user specified information prior to training or running.
 * This includes the network configuration given the number of activations in the network 
 * layers, the name of the file containing the input table/truth table, what will be printed 
 * (input table, truth table, etc...), from where the weights are being taken, etc... If the 
 * network is training, it will output the name of the file containing the truth table, the 
 * random number range, the maximum number of iterations for training, the error threshold, 
 * and the value of the learning rate, lambda.
 */
   public static void echoConfigurationParams()
   {
      System.out.println(String.format("Network Configuration: %d-%d-%d-%d", 
                                       numInputNodes, numHiddenKNodes, numHiddenJNodes, numOutputNodes));
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
      inputTable = new double[numTestCases][numInputNodes];
      targetOutputs = new double[numOutputNodes];

      a = new double[numActivationLayers][];
      w = new double[numActivationLayers - 1][][];
      
      int n = INPUT_LAYER;
      a[n] = new double[numInputNodes];
      w[n] = new double[numInputNodes][numHiddenKNodes];

      n = FIRST_HIDDEN_LAYER;
      a[n] = new double[numHiddenKNodes];
      w[n] = new double[numHiddenKNodes][numHiddenJNodes];

      n = SECOND_HIDDEN_LAYER;
      a[n] = new double[numHiddenJNodes];
      w[n] = new double[numHiddenJNodes][numOutputNodes];

      n = OUTPUT_LAYER;
      a[n] = new double[numOutputNodes];

      if (isTraining)
      {
         truthTable = new double[numTestCases][numOutputNodes];

         theta = new double[numActivationLayers - 1][];
         psi = new double[numActivationLayers][];

         n = FIRST_HIDDEN_LAYER;
         theta[n] = new double[numHiddenKNodes];

         n = SECOND_HIDDEN_LAYER;
         theta[n] = new double[numHiddenJNodes];
         psi[n] = new double[numHiddenJNodes];

         n = OUTPUT_LAYER;
         psi[n] = new double[numOutputNodes];
      } // if (isTraining)

      outputActivationsRun = new double[numTestCases][numOutputNodes];
   } // public static void allocateMem()

/**
 * Sets the weights of the network to randomly generated values within bounds provided
 * in the network configuration.
 */
   public static void randomlyGenerateWeights()
   {
      int n = INPUT_LAYER;
      for (int m = 0; m < numInputNodes; m++)
      {
         for (int k = 0; k < numHiddenKNodes; k++)
         {
            w[n][m][k] = randomize(randomLowBound, randomHighBound);
         }
      }

      n = FIRST_HIDDEN_LAYER;
      for (int k = 0; k < numHiddenKNodes; k++)
      {
         for (int j = 0; j < numHiddenJNodes; j++)
         {
            w[n][k][j] = randomize(randomLowBound, randomHighBound);
         }
      }

      n = SECOND_HIDDEN_LAYER;
      for (int j = 0; j < numHiddenJNodes; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            w[n][j][i] = randomize(randomLowBound, randomHighBound);
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

      int loadM = in.readInt();
      int loadK = in.readInt();
      int loadJ = in.readInt();
      int loadI = in.readInt();

      if (loadM != numInputNodes || loadK != numHiddenKNodes || loadJ != numHiddenJNodes || loadI != numOutputNodes)
      {
         throw new IllegalArgumentException(String.format("Weight file configuration does not match network architecture of %d-%d-%d-%d",
                                                          numInputNodes, numHiddenKNodes, numHiddenJNodes, numOutputNodes));
      }

      int n = INPUT_LAYER;
      for (int m = 0; m < numInputNodes; m++)
      {
         for (int k = 0; k < numHiddenKNodes; k++)
         {
            w[n][m][k] = in.readDouble();
         }
      }

      n = FIRST_HIDDEN_LAYER;
      for (int k = 0; k < numHiddenKNodes; k++)
      {
         for (int j = 0; j < numHiddenJNodes; j++)
         {
            w[n][k][j] = in.readDouble();
         }
      }

      n = SECOND_HIDDEN_LAYER;
      for (int j = 0; j < numHiddenJNodes; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            w[n][j][i] = in.readDouble();
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

      out.writeInt(numInputNodes);
      out.writeInt(numHiddenKNodes);
      out.writeInt(numHiddenJNodes);
      out.writeInt(numOutputNodes);

      int n = INPUT_LAYER;
      for (int m = 0; m < numInputNodes; m++)
      {
         for (int k = 0; k < numHiddenKNodes; k++)
         {
            out.writeDouble(w[n][m][k]);
         }
      }

      n = FIRST_HIDDEN_LAYER;
      for (int k = 0; k < numHiddenKNodes; k++)
      {
         for (int j = 0; j < numHiddenJNodes; j++)
         {
            out.writeDouble(w[n][k][j]);
         }
      }

      n = SECOND_HIDDEN_LAYER;
      for (int j = 0; j < numHiddenJNodes; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            out.writeDouble(w[n][j][i]);
         }
      }

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
      int n = INPUT_LAYER;
      for (int m = 0; m < numInputNodes; m++)
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
      int n = FIRST_HIDDEN_LAYER;
      for (int k = 0; k < numHiddenKNodes; k++)
      {
         double theta_k = 0.0;
         for (int m = 0; m < numInputNodes; m++)
         {
            theta_k += w[n - 1][m][k] * a[n - 1][m];
         }

         a[n][k] = fActivation(theta_k);
      } // for (int k = 0; k < numHiddenKNodes; k++) 

      n = SECOND_HIDDEN_LAYER;
      for (int j = 0; j < numHiddenJNodes; j++)
      {
         double theta_j = 0.0;
         for (int k = 0; k < numHiddenKNodes; k++)
         {
            theta_j += w[n - 1][k][j] * a[n - 1][k];
         }

         a[n][j] = fActivation(theta_j);
      } // for (int j = 0; j < numHiddenJNodes; j++)

      n = OUTPUT_LAYER;
      for (int i = 0; i < numOutputNodes; i++)
      {
         double theta_i = 0.0;
         for (int j = 0; j < numHiddenJNodes; j++)
         {
            theta_i += w[n - 1][j][i] * a[n - 1][j];
         }

         a[n][i] = fActivation(theta_i);
      } // for (int i = 0; i < numOutputNodes; i++)
   } // public static void runForRunning()

/**
 * Runs the network once given a specific training case while training the network. 
 * First, it computes the output activation given a specific set of input activations.
 * Then, it computes twice the error for efficiency, and it returns that value.
 * @return 2 * error for the particular training case.
 */
   public static double runForTraining()
   {
      int n = FIRST_HIDDEN_LAYER;
      for (int k = 0; k < numHiddenKNodes; k++)
      {
         double theta_k = 0.0;
         for (int m = 0; m < numInputNodes; m++)
         {
            theta_k += w[n - 1][m][k] * a[n - 1][m];
         }

         theta[n][k] = theta_k;
         a[n][k] = fActivation(theta[n][k]);
      } // for (int k = 0; k < numHiddenKNodes; k++)

      n = SECOND_HIDDEN_LAYER;
      for (int j = 0; j < numHiddenJNodes; j++)
      {
         double theta_j = 0.0;
         for (int k = 0; k < numHiddenKNodes; k++)
         {
            theta_j += w[n - 1][k][j] * a[n - 1][k];
         }

         theta[n][j] = theta_j;
         a[n][j] = fActivation(theta[n][j]);
      } // for (int j = 0; j < numHiddenJNodes; j++)

      double twiceError = 0.0;

      n = OUTPUT_LAYER;
      for (int i = 0; i < numOutputNodes; i++)
      {
         double theta_i = 0.0;
         for (int j = 0; j < numHiddenJNodes; j++)
         {
            theta_i += w[n - 1][j][i] * a[n - 1][j];
         }

         a[n][i] = fActivation(theta_i);
         double omega_i = targetOutputs[i] - a[n][i];
         psi[n][i] = omega_i * fPrimeActivation(theta_i);
         twiceError += omega_i * omega_i;
      } // for (int i = 0; i < numOutputNodes; i++)
      
      return twiceError;
   } // public static double runForTraining()

/**
 * Trains the network by computing the changes in weights and updating the weights.
 */
   public static void train()
   {
      int n = SECOND_HIDDEN_LAYER;
      for (int j = 0; j < numHiddenJNodes; j++)
      {
         double omega_j = 0.0;

         for (int i = 0; i < numOutputNodes; i++)
         {
            omega_j += psi[n + 1][i] * w[n][j][i];
            w[n][j][i] += lambda * a[n][j] * psi[n + 1][i];
         }

        psi[n][j] = omega_j * fPrimeActivation(theta[n][j]);
      } // for (int j = 0; j < numHiddenJNodes; j++)

      n = FIRST_HIDDEN_LAYER;
      for (int k = 0; k < numHiddenKNodes; k++)
      {
         double omega_k = 0.0;

         for (int j = 0; j < numHiddenJNodes; j++)
         {
            omega_k += psi[n + 1][j] * w[n][k][j];
            w[n][k][j] += lambda * a[n][k] * psi[n + 1][j];
         }

         double psi_k = omega_k * fPrimeActivation(theta[n][k]);

         for (int m = 0; m < numInputNodes; m++)
         {
            w[n - 1][m][k] += lambda * a[n - 1][m] * psi_k;
         }
      } // for (int k = 0; k < numHiddenKNodes; k++)
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
            for (int i = 0; i < numOutputNodes; i++)
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

         int n = OUTPUT_LAYER;
         for (int i = 0; i < numOutputNodes; i++)
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
            for (int m = 0; m < numInputNodes; m++)
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
         for (int m = 0; m < numInputNodes; m++)
         {
            System.out.print(inputTable[t][m] + " ");
         }

         for (int i = 0; i < numOutputNodes; i++)
         {
            System.out.printf("%.4f ", outputActivationsRun[t][i]);
         }
         System.out.println();
      } // for (int t = 0; t < numTestCases; t++)
   } // public static void printRunningResults()

/**
 * Does the following in order: (1) Prints the name of the configuration file, (2) Loads
 * the configuration parameters from that file, (3) Prints the relevant information regarding 
 * the configuration parameters, (4) Allocates memory for the important arrays in the 
 * network, (5) Populates the arrays, (6) Trains the network if that was set in the 
 * configuration parameters and outputs training results, (7) Runs the network, (8) If 
 * desired, saves the final weights that were successful, (9) Outputs running results. 
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
} // public class NLayer