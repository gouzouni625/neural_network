package org.nn.distorters;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.Random;

/**
 * @class ImageDistorter
 * @brief Implementation of a Distorter of images.
 * <p>
 * Applies affine transformations on the given images.
 */
public class ImageDistorter extends Distorter {
  /**
   * @brief Default Constructor.
   */
  public ImageDistorter () {
    super();
  }

  /**
   * @param distortFrequency The value for the distort frequency.
   * @brief Constructor.
   */
  public ImageDistorter (int distortFrequency) {
    super(distortFrequency);
  }

  /**
   * @param image The set of data on which to apply the transformations.
   * @return Returns the distorted image.
   * @brief The method that will be called to apply a distortion on an image.
   */
  public BufferedImage distort (BufferedImage image) {
    Random random = new Random();
    double distortionType = random.nextDouble();

    BufferedImage transformedImage = new BufferedImage(image.getWidth(), image.getHeight(),
        image.getType());

    if (distortionType < 0.25) { // Rotating. [-pi/12, pi/12).
      double parameter = ((2 * random.nextDouble() - 1) / 12) * Math.PI; // Angle.

      new AffineTransformOp(AffineTransform.getRotateInstance(parameter, image.getWidth() / 2,
          image.getHeight() / 2), AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    } else if (distortionType < 0.5) {  // Scaling. [0.85, 1.15).
      // Volume for horizontal axis.
      double parameter = ((2 * random.nextDouble() - 1) * 15 / 100) + 1;

      new AffineTransformOp(AffineTransform.getScaleInstance(parameter, parameter),
          AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    } else if (distortionType < 0.75) {  // Shearing. [-0.15, 0.15).
      double parameter = ((2 * random.nextDouble() - 1) * 15 / 100);

      new AffineTransformOp(AffineTransform.getShearInstance(parameter, parameter),
          AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    } else { // translating [-5, 5).
      double parameterX = (2 * random.nextDouble() - 1) * 5;
      double parameterY = (2 * random.nextDouble() - 1) * 5;

      new AffineTransformOp(AffineTransform.getTranslateInstance(parameterX, parameterY),
          AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    }

    return transformedImage;
  }

  public double[][] distort (double[][] data) {
    int numberOfImages = data.length;

    for (int i = 0; i < numberOfImages; i++) {
      BufferedImage image = vectorToBufferedImage(data[i], 64, 64, - 1, 1);
      image = distort(image);
      data[i] = bufferedImageToVector(image, - 1, 1);
    }

    return data;
  }

  public static BufferedImage vectorToBufferedImage (double[] vector, int width, int height,
                                                     double minValue, double maxValue) {
    BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
    byte[] pixels = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();

    int numberOfPixels = pixels.length;

    for (int i = 0; i < numberOfPixels; i++) {
      pixels[i] = (byte) (((byte) ((vector[i] - minValue) * 255 / (maxValue - minValue))) & 0xFF);
    }

    return bufferedImage;
  }

  /**
   * @param image The OpenCV Mat object that represents the image.
   * @param min   The minimum value that the vector should have.
   * @param max   The maximum value that the vector should have.
   * @return Returns the image conversion to a vector.
   * @brief Converts an image to a vector of doubles.
   */
  public static double[] bufferedImageToVector (BufferedImage image, double min, double max) {
    byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

    int numberOfPixels = pixels.length;

    double[] vector = new double[numberOfPixels];

    for (int i = 0; i < numberOfPixels; i++) {
      vector[i] = (pixels[i] & 0xFF) * (max - min) / 255 + min;
    }

    /*****/
    /*for (int i = 0; i < 4096; i++) {
      if (vector[i] == - 1) {
        System.out.print(0 + " ");
      } else {
        System.out.print(1 + " ");
      }
      if (i % 64 == 0) {
        System.out.println();
      }
    }
    System.out.println();*/
    /*****/

    return vector;
  }

  /**
   * @param sampleRows The number of rows of each sample.
   * @brief Setter method for the number of rows of each sample.
   */
  public void setSampleRows (int sampleRows) {
    sampleRows_ = sampleRows;
  }

  /**
   * @return Returns the number of rows of each sample.
   * @brief Getter method for the number of rows of each sample.
   */
  public int getSampleRows () {
    return sampleRows_;
  }

  /**
   * @param sampleColumns The number of rows of each sample.
   * @brief Setter method for the number of columns of each sample.
   */
  public void setSampleColumns (int sampleColumns) {
    sampleColumns_ = sampleColumns;
  }

  /**
   * @return Returnes the number of columns of each sample.
   * @brief Getter method for the number of columns of each sample.
   */
  public int getSampleColumns () {
    return sampleColumns_;
  }

  private int sampleRows_; //!< The number of rows of each sample.
  private int sampleColumns_; //!< The number of columns of each sample.

}
