package data_managers;

import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class MNISTDataManager{
  
  public MNISTDataManager(String trainingDataPath,
                          String trainingLabelsPath,
                          String testingDataPath,
                          String testingLabelsPath){
    trainingDataPath_ = trainingDataPath;
    trainingLabelsPath_ = trainingLabelsPath;
    testingDataPath_ = testingDataPath;
    testingLabelsPath_ = testingLabelsPath;
  }
  
  public void load() throws IOException{
    // Load the training data. =================================================
    FileInputStream fileInputStream = new FileInputStream(trainingDataPath_);
    fileInputStream.skip(16);
    
    trainingData_ = new int[numberOfTrainingSamples_][sampleLength_];
    byte[] sampleBuffer = new byte[sampleLength_];
    
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      fileInputStream.read(sampleBuffer);
      
      for(int j = 0;j < sampleLength_;j++){
        trainingData_[i][j] = ((int)(sampleBuffer[j] & 0xFF));    
      }
    }
    fileInputStream.close();
    
    // Load the training labels. ===============================================
    fileInputStream = new FileInputStream(trainingLabelsPath_);
    fileInputStream.skip(8);
    
    trainingLabels_ = new int[numberOfTrainingSamples_];
    byte[] labelBuffer = new byte[1];
    
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      fileInputStream.read(labelBuffer);
      trainingLabels_[i] = (int)(labelBuffer[0] & 0xFF);
    }
    fileInputStream.close();
    
    // Load the testing data. ==================================================
    fileInputStream = new FileInputStream(testingDataPath_);
    fileInputStream.skip(16);
    
    testingData_ = new int[numberOfTestingSamples_][sampleLength_];
    
    for(int i = 0;i < numberOfTestingSamples_;i++){
      fileInputStream.read(sampleBuffer);
      
      for(int j = 0;j < sampleLength_;j++){
        testingData_[i][j] = ((int)(sampleBuffer[j] & 0xFF));
      }
    }
    fileInputStream.close();
    
    // Load the testing labels. ================================================
    fileInputStream = new FileInputStream(testingLabelsPath_);
    fileInputStream.skip(8);
    
    testingLabels_ = new int[numberOfTestingSamples_];
    
    for(int i = 0;i < numberOfTestingSamples_;i++){
      fileInputStream.read(labelBuffer);
      testingLabels_[i] = (int)(labelBuffer[0] & 0xFF);
    }
    fileInputStream.close();
  }
  
  public void saveImage(String path){
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    
    Mat image = Mat.zeros(sampleRows_, sampleColumns_, CvType.CV_64F);
    
    // Save training data as images. ===========================================
    for(int sample = 0;sample < trainingData_.length;sample++){
      for(int i = 0;i < sampleRows_;i++){
        for(int j = 0;j < sampleColumns_;j++){
          image.put(i, j, trainingData_[sample][i * sampleColumns_ + j]);
        }
      }
      
      Highgui.imwrite(path + "/training_data/image" + sample + ".tiff", image);
    }
    
    // Save testing data as images. ============================================
    for(int sample = 0;sample < testingData_.length;sample++){
      for(int i = 0;i < sampleRows_;i++){
        for(int j = 0;j < sampleColumns_;j++){
          image.put(i, j, testingData_[sample][i * sampleColumns_ + j]);
        }
      }
      
      Highgui.imwrite(path + "/testing_data/image" + sample + ".tiff", image);
    }
  }
  
  public void toBinaryFormat(){
    for(int i = 0;i < trainingData_.length;i++){
      for(int j = 0;j < sampleLength_;j++){
        if(trainingData_[i][j] > 0){
          trainingData_[i][j] = 255;
        }
        else{
          trainingData_[i][j] = 0;
        }
      }
    }
    
    for(int i = 0;i < testingData_.length;i++){
      for(int j = 0;j < sampleLength_;j++){
        if(testingData_[i][j] > 0){
          testingData_[i][j] = 255;
        }
        else{
          testingData_[i][j] = 0;
        }
      }
    }
  }
  
  public void save() throws IOException{
    FileOutputStream fileOutputStream = null;
    DataOutputStream dataOutputStream = null;
    
    // Save training data. =====================================================
    fileOutputStream = new FileOutputStream(trainingDataOutputPath_);
    dataOutputStream = new DataOutputStream(fileOutputStream);
      
    dataOutputStream.writeInt(trainingData_.length);
    for(int i = 0;i < trainingData_.length;i++){
      for(int j = 0;j < sampleLength_;j++){
        dataOutputStream.writeInt((int)trainingData_[i][j]);
      }
    } 
    dataOutputStream.close();
   
    // Save training labels. ===================================================
    fileOutputStream = new FileOutputStream(trainingLabelsOutputPath_);
    dataOutputStream = new DataOutputStream(fileOutputStream);
      
    dataOutputStream.writeInt(trainingLabels_.length);
    for(int i = 0;i < trainingLabels_.length;i++){
      dataOutputStream.writeInt(trainingLabels_[i]);
    }
    dataOutputStream.close();
    
    // Save testing data. ======================================================
    fileOutputStream = new FileOutputStream(testingDataOutputPath_);
    dataOutputStream = new DataOutputStream(fileOutputStream);
      
    dataOutputStream.writeInt(testingData_.length);
    for(int i = 0;i < testingData_.length;i++){
      for(int j = 0;j < sampleLength_;j++){
        dataOutputStream.writeInt((int)testingData_[i][j]);
      }
    } 
    dataOutputStream.close();
    
    // Save testing labels. ====================================================
    fileOutputStream = new FileOutputStream(testingLabelsOutputPath_);
    dataOutputStream = new DataOutputStream(fileOutputStream);
      
    dataOutputStream.writeInt(testingLabels_.length);
    for(int i = 0;i < testingLabels_.length;i++){
      dataOutputStream.writeInt(testingLabels_[i]);
    }
    dataOutputStream.close();
  }

  public void setTrainingDataOutputPath(String trainingDataOutputPath){
    trainingDataOutputPath_ = trainingDataOutputPath;
  }

  public String getTrainingDataOutputPath(){
    return trainingDataOutputPath_;
  }

  public void setTrainingLabelsOutputPath(String trainingLabelsOutputPath){
    trainingLabelsOutputPath_ = trainingLabelsOutputPath;
  }

  public String getTrainingLabelsOutputPath(){
    return trainingLabelsOutputPath_;
  }

  public void setTestingDataOutputPath(String testingDataOutputPath){
    testingDataOutputPath_ = testingDataOutputPath;
  }

  public String getTestingDataOutputPath(){
    return testingDataOutputPath_;
  }

  public void setTestingLabelsOutputPath(String testingLabelsOutputPath){
    testingLabelsOutputPath_ = testingLabelsOutputPath;
  }

  public String getTestingLabelsOutputPath(){
    return testingLabelsOutputPath_;
  }

  public void setNumberOfTrainingSamples(int numberOfTrainingSamples){
    numberOfTrainingSamples_ = numberOfTrainingSamples;
  }

  public int getNumberOfTrainingSamples(){
    return numberOfTrainingSamples_;
  }

  public void setNumberOfTestingSamples(int numberOfTestingSamples){
    numberOfTestingSamples_ = numberOfTestingSamples;
  }

  public int getNumberOfTestingSamples(){
    return numberOfTestingSamples_;
  }

  public void setSampleLength(int sampleLength){
    sampleLength_ = sampleLength;
  }

  public int getSampleLength(){
    return sampleLength_;
  }

  public void setNumberOfLabels(int numberOfLabels){
    numberOfLabels_ = numberOfLabels;
  }

  public int getNumberOfLabels(){
    return numberOfLabels_;
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

  private String trainingDataPath_;
  private String trainingLabelsPath_;
  private String testingDataPath_;
  private String testingLabelsPath_;
  
  private String trainingDataOutputPath_;
  private String trainingLabelsOutputPath_;
  private String testingDataOutputPath_;
  private String testingLabelsOutputPath_;
  
  private int numberOfTrainingSamples_;
  private int numberOfTestingSamples_;
  private int sampleLength_;
  private int numberOfLabels_;
  private int sampleRows_;
  private int sampleColumns_;
  
  private int[][] trainingData_;
  private int[] trainingLabels_;
  private int[][] testingData_;
  private int[] testingLabels_;
}
