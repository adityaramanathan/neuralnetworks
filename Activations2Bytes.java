/**
**
** Activations2Bytes.java
**
** @author Aditya Ramanathan
** May 8, 2026
**
** Inverse of Bytes2Activations
**
** Reads double activations (typically 0.0 to 1.0)
** and converts them back into unsigned byte values (0-255)
**
** Usage:
** java Activations2Bytes input_activation_file output_byte_file
**
**/
import java.io.*;

public class Activations2Bytes
{
   static final double SCALE  = 255.0;
   static final double OFFSET = 0.0;

   public static void main(String[] args)
   {
      double activationVal;
      int byteVal;

      if (args.length != 2)
      {
         System.out.println("Usage: java Activations2Bytes input_activation_file output_byte_file");
      }
      else
      {
         String inFileName  = args[0];
         String outFileName = args[1];

         System.out.printf("Reading file '%s' and writing out '%s'.\n", inFileName, outFileName);

         try
         {
            FileInputStream fInStream = new FileInputStream(inFileName);
            DataInputStream in = new DataInputStream(fInStream);

            try
            {
               FileOutputStream fOutStream = new FileOutputStream(outFileName);
               DataOutputStream out = new DataOutputStream(fOutStream);

               boolean EOF = false;

               while (!EOF)
               {
                  try
                  {
                     activationVal = in.readDouble();

                     
                     byteVal =
                        (int)Math.round((activationVal - OFFSET) * SCALE);

                     if (byteVal < 0)   byteVal = 0;
                     if (byteVal > 255) byteVal = 255;

                     // System.out.printf("%f\t%d\t\t",activationVal,byteVal);

                     /*
                     * Invert so background becomes white again
                     */
                     byteVal = ~byteVal;

                     out.writeByte(byteVal);
                  }
                  catch (EOFException e)
                  {
                     EOF = true;
                  }
               }

               out.close();
               fOutStream.close();
            }
            catch (Exception e)
            {
               System.err.println("File output error " + e);
            }

            in.close();
            fInStream.close();
         }
         catch (Exception e)
         {
            System.err.println("File input error " + e);
         }
      } // if (args.length != 2)
   } // public static void main(String[] args)
} // public class Activations2Bytes
