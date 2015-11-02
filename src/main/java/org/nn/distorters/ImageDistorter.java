package org.nn.distorters;

import org.improc.core.Core;
import org.improc.image.Image;

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
   *  @param data The set of data on which to apply the transformations.
   *
   *  @return Returns the distorted image.
   */
  public double[][] distort(double[][] data){
    Random random = new Random();
    double destortionType, parameter;
    double[][] affineMatrix = new double[2][2];

    parameter = ((2 * random.nextDouble() - 1) / 12) * Math.PI; // Angle.
    affineMatrix[0][0] = Math.cos(parameter);
    affineMatrix[0][1] = Math.sin(parameter);
    affineMatrix[1][0] = -Math.sin(parameter);
    affineMatrix[1][1] = Math.cos(parameter);

    Image image = new Image(sampleColumns_, sampleRows_);
    for(int x = 0;x < sampleColumns_;x++){
      for(int y = 0;y < sampleRows_;x++){
        image.setPixel(x, y, (byte)(data[y][x] * 100));
      }
    }
    image = Core.affineTransformRotate(image, affineMatrix);
    for(int x = 0;x < sampleColumns_;x++){
      for(int y = 0;y < sampleRows_;x++){
        data[y][x] = (double)image.getPixel(x, y) / 100;
      }
    }
    return data;
    /*for(int i = 0;i < data.length;i++){
      destortionType = random.nextDouble();

      if(destortionType < 0.25){ // Rotating. [-pi/12, pi/12).
        parameter = ((2 * random.nextDouble() - 1) / 12) * Math.PI; // Angle.

      }
      else if(destortionType < 0.5){  // Scaling. [0.85, 1.15).
        // Volume for horizontal axis.
        parameter = ((2 * random.nextDouble() - 1) * 15 / 100) + 1;
        trfMtx.put(0, 0, parameter); trfMtx.put(0, 1, 0); trfMtx.put(0, 2, 0);

        // Volume for vertical axis.
        parameter = ((2 * random.nextDouble() - 1) * 15 / 100) + 1;
        trfMtx.put(1, 0, 0); trfMtx.put(1, 1, parameter); trfMtx.put(1, 2, 0);
      }
      else if(destortionType < 0.75){  // Shearing. [-0.15, 0.15).
        parameter = ((2 * random.nextDouble() - 1) * 15 / 100);
        trfMtx.put(0, 0, 1); trfMtx.put(0, 1, parameter); trfMtx.put(0, 2, 0);
        trfMtx.put(1, 0, 0); trfMtx.put(1, 1, 1); trfMtx.put(1, 2, 0);
      }
      else{ // translating [-5, 5).
        parameter = (2 * random.nextDouble() - 1) * 5;
        trfMtx.put(0, 0, 1); trfMtx.put(0, 1, 0); trfMtx.put(0, 2, parameter);

        parameter = (2 * random.nextDouble() - 1) * 5;
        trfMtx.put(1, 0, 0); trfMtx.put(1, 1, 1); trfMtx.put(1, 2, parameter);
      }

      for(int j = 0;j < sampleRows_;j++){
        for(int k = 0;k < sampleColumns_;k++){
          image.put(j, k, data[i][j * sampleColumns_ + k]);
        }
      }

      Imgproc.warpAffine(image, image, trfMtx, image.size());

      for(int j = 0;j < sampleRows_;j++){
        for(int k = 0;k < sampleColumns_;k++){
          data[i][j * sampleColumns_ + k] = image.get(j, k)[0];
        }
      }

    }*/
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
