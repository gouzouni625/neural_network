package org.nn.distorters;

import org.improc.core.Core;
import org.improc.image.Image;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.util.Random;

/** @class ImageDistorter
 *
 *  @brief Implementation of a Distorter of images.
 *
 *  Applies affine transformations on the given images.
 */
public class ImageDistorter extends Distorter{
  /**
   *  @brief Default Constructor.
   */
  public ImageDistorter(){
    super();
  }

  /**
   *  @brief Constructor.
   *
   *  @param distortFrequency The value for the distort frequency.
   */
  public ImageDistorter(int distortFrequency){
    super(distortFrequency);
  }

  /**
   *  @brief The method that will be called to apply a distortion on an image.
   *
   *  @param image The set of data on which to apply the transformations.
   *
   *  @return Returns the distorted image.
   */
  public BufferedImage distort(BufferedImage image){
    Random random = new Random();
    double distortionType = random.nextDouble();

    BufferedImage transformedImage = new BufferedImage(image.getWidth(), image.getHeight(),
        image.getType());

    if(distortionType < 0.25){ // Rotating. [-pi/12, pi/12).
      double parameter = ((2 * random.nextDouble() - 1) / 12) * Math.PI; // Angle.

      new AffineTransformOp(AffineTransform.getRotateInstance(parameter, image.getWidth() / 2,
          image.getHeight() / 2), AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    }
    else if(distortionType < 0.5){  // Scaling. [0.85, 1.15).
      // Volume for horizontal axis.
      double parameter = ((2 * random.nextDouble() - 1) * 15 / 100) + 1;

      new AffineTransformOp(AffineTransform.getScaleInstance(parameter, parameter),
          AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    }
    else if(distortionType < 0.75){  // Shearing. [-0.15, 0.15).
      double parameter = ((2 * random.nextDouble() - 1) * 15 / 100);

      new AffineTransformOp(AffineTransform.getShearInstance(parameter, parameter),
          AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    }
    else{ // translating [-5, 5).
      double parameterX = (2 * random.nextDouble() - 1) * 5;
      double parameterY = (2 * random.nextDouble() - 1) * 5;

      new AffineTransformOp(AffineTransform.getTranslateInstance(parameterX, parameterY),
          AffineTransformOp.TYPE_BILINEAR).filter(image, transformedImage);
    }

    return transformedImage;
  }

  /**
   *  @brief Setter method for the number of rows of each sample.
   *
   *  @param sampleRows The number of rows of each sample.
   */
  public void setSampleRows(int sampleRows){
    sampleRows_ = sampleRows;
  }

  /**
   *  @brief Getter method for the number of rows of each sample.
   *
   *  @return Returns the number of rows of each sample.
   */
  public int getSampleRows(){
    return sampleRows_;
  }

  /**
   *  @brief Setter method for the number of columns of each sample.
   *
   *  @param sampleColumns The number of rows of each sample.
   */
  public void setSampleColumns(int sampleColumns){
    sampleColumns_ = sampleColumns;
  }

  /**
   *  @brief Getter method for the number of columns of each sample.
   *
   *  @return Returnes the number of columns of each sample.
   */
  public int getSampleColumns(){
    return sampleColumns_;
  }

  private int sampleRows_; //!< The number of rows of each sample.
  private int sampleColumns_; //!< The number of columns of each sample.

}
