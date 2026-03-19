import java.io.*;
import java.util.*;

/**
 * @author Aditya Ramanathan
 * @version 3/17/26
 * Converts a .txt file to a binary file.
 */
public class BinaryConverter 
{
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
      try 
      {
         convertToBinary("input.txt", "input.bin");
      } 
      catch (IOException e) 
      {
         e.printStackTrace();
      }
   } // public static void main(String[] args)
}