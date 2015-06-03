package main.distorters;

import java.util.Random;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/** \class ImageDistorter
 *  \brief Class that contains methods for distorting images by applying affine
 *         transformations on them.
 */
public class ImageDistorter extends Distorter{

  public ImageDistorter(int distortFrequency){
    super(distortFrequency);
  }
  
  /** \brief Applies random affine transformations on a set of data. 
   * 
   *  \param data The set of data on which to apply the transformations.
   *  \param sampleRows The number of rows of each sample if it is mapped on
   *                    a grid.
   *  \param sampleColumns The number of columns of each sample if it is mapped
   *                       on a grid.
   *  \return The transformed data.
   */
  public double[][] distort(double[][] data){
    Random random = new Random();
    double destortionType, parameter;
    Mat trfMtx = new Mat(2, 3, CvType.CV_64F);
    Mat image = new Mat(sampleRows_, sampleColumns_, CvType.CV_64F);
    
    for(int i = 0;i < data.length;i++){
      destortionType = random.nextDouble();

      if(destortionType < 0.25){ // Rotating. [-pi/12, pi/12).
        parameter = ((2 * random.nextDouble() - 1) / 12) * Math.PI; // Angle.
        trfMtx.put(0, 0, Math.cos(parameter));
        trfMtx.put(0, 1, Math.sin(parameter));
        trfMtx.put(0, 2, 0);
        trfMtx.put(1, 0, -Math.sin(parameter));
        trfMtx.put(1, 1, Math.cos(parameter));
        trfMtx.put(1, 2, 0);
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
      
    }

    return data;
  }

  public void setSampleRows(int sampleRows){
    sampleRows_ = sampleRows;
  }

  public int getSampleRows(){
    return sampleRows_;
  }

  public void setSampleColumns(int sampleColumns){
    sampleColumns_ = sampleColumns;
  }

  public int getSampleColumns(){
    return sampleColumns_;
  }

  private int sampleRows_;
  private int sampleColumns_;
}