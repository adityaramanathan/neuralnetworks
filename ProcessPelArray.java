import java.io.*;
import java.nio.file.*;

/**
 * @author Aditya Ramanathan
 * May 5, 2026
 * 
 * This code follows the following pipeline:
 * 1. Load ORIG_WIDTH x ORIG_HEIGHT grayscale byte image (binImageFiles), 
 *    ORIG_WIDTH and ORIG_HEIGHT are constants
 * 2. Ones-complement since the background of photos is white
 * 3. Compute center of mass (COM)
 * 4. Fixed-size crop around (COM)
 * 5. Write processed grayscale bytes to output file
 */
public class ProcessPelArray 
{
   private static final int ORIG_WIDTH  = 3088;
   private static final int ORIG_HEIGHT = 2316;

   private static final int FINAL_WIDTH  = 1800;
   private static final int FINAL_HEIGHT = 1800;

   public static void main(String[] args) throws IOException 
   {
      if (args.length != 2) 
      {
         System.out.println("Usage: java ProcessPelArray <input.bin> <output.bin>");
         return;
      }

      String inputFile  = args[0];
      String outputFile = args[1];

      byte[] rawBytes = Files.readAllBytes(Paths.get(inputFile));

      if (rawBytes.length != ORIG_WIDTH * ORIG_HEIGHT) 
      {
         System.err.println("ERROR: File size does not match expected dimensions.");
         return;
      }

/*
 * Load byte data into PelArray (grayscale → RGB pels)
 */
      int[][] pels = new int[ORIG_HEIGHT][ORIG_WIDTH];

      for (int y = 0; y < ORIG_HEIGHT; y++) 
      {
         for (int x = 0; x < ORIG_WIDTH; x++) 
         {
            int v = rawBytes[y * ORIG_WIDTH + x] & 0xFF;
            pels[y][x] = (v << 16) | (v << 8) | v;
         }
      }

      PelArray img = new PelArray(pels);

/*
 * Ones Complement since the background is white
 */
      img = img.onesComplimentImage();

/*
 * Computer Center of Mass (COM)
 */
      int comX = img.getXcom();
      int comY = img.getYcom();

/*
 * Fixed-size crop around COM
 */
      int halfW = FINAL_WIDTH  / 2;
      int halfH = FINAL_HEIGHT / 2;

      int x0 = comX - halfW;
      int y0 = comY - halfH;

      x0 = Math.max(0, Math.min(x0, ORIG_WIDTH - FINAL_WIDTH));
      y0 = Math.max(0, Math.min(y0, ORIG_HEIGHT - FINAL_HEIGHT));

      int x1 = x0 + FINAL_WIDTH - 1;
      int y1 = y0 + FINAL_HEIGHT - 1;

      img = img.crop(x0, y0, x1, y1);

      int[][] out = img.getPelArray();

      int widthF = x1 - x0;
      int heightF = y1 - y0;

      byte[] outBytes = new byte[widthF * heightF];

      for (int y = 0; y < heightF; y++) 
      {
         for (int x = 0; x < widthF; x++) 
         {
            outBytes[y * widthF + x] = (byte)(out[y][x] & 0xFF);
         }
      }

      Files.write(Paths.get(outputFile), outBytes);
   } // public static void main(String[] args) throws IOException
} // public class ProcessPelArray