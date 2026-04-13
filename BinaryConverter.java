import java.io.*;
import java.util.*;

/**
 * @author Aditya Ramanathan
 * @version 4/12/26
 * Converts a .txt file to a binary file.
 */
public class BinaryConverter 
{

   private static final String DEFAULT_INPUT_FILE_NAME = "input.txt";
   private static final String DEFAULT_BIN_FILE_NAME = "input.bin";

   /**
    * Given a specific input file, converts the content and writes it to a binary output file.
    * @param inputFileName the name of the input .txt file
    * @param outputFileName the name of the binary output file
    */
   public static void convertToBinary(String inputFileName, String outputFileName) throws IOException 
   {
      try (Scanner scanner = new Scanner(new File(inputFileName));
           DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(outputFileName))))
      {
         while (scanner.hasNext()) 
         {
            if (scanner.hasNextDouble()) 
            {
               double val = scanner.nextDouble();
               out.writeDouble(val);
            }
            else 
            {
               scanner.next(); 
            }
         } // while (scanner.hasNext()) 
      } // try (Scanner scanner = new Scanner(new File(inputFileName)); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(outputFileName))))
   } // public static void convertToBinary(String inputFileName, String outputFileName) throws IOException 

   /**
    * Converts the file to binary by calling the convertToBinary method. 
    * @param args arguments from the command line.
    */
   public static void main(String[] args) 
   {
      String inputFileName;
      String binFileName;
      if (args.length == 0)
      {
         inputFileName = DEFAULT_INPUT_FILE_NAME;
         binFileName = DEFAULT_BIN_FILE_NAME;
      }
      else if (args.length == 1)
      {
         inputFileName = args[0];
         binFileName = DEFAULT_BIN_FILE_NAME;
      }
      else
      {
         inputFileName = args[0];
         binFileName = args[1];
      }

      try 
      {
         convertToBinary(inputFileName, binFileName);
      } 
      catch (IOException e) 
      {
         e.printStackTrace();
      }
   } // public static void main(String[] args)
}