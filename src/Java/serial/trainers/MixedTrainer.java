package trainers;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/** \class MixedTrainer class. Trains a neural network on a mixed data set.
 *         The data set comprises of data from the MNIST database and data
 *         collected using GeoGebra applet. The MNIST database can be found
 *         at
 *         http://yann.lecun.com/exdb/mnist/
 *         and the GeoGebra software at
 *         https://www.geogebra.org/
 *         
 */
public class MixedTrainer extends Trainer{
  
  public MixedTrainer(int[] sizesOfLayers){
    super(sizesOfLayers);
    
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  }
  
  public void load() throws IOException{
    numberOfDatabaseTrainingSamples_ = 255;
    numberOfMNISTTrainingSamples_ = 345;
    numberOfTrainingSamples_ = numberOfDatabaseTrainingSamples_ +
                               numberOfMNISTTrainingSamples_;
    sampleRows_ = 28;
    sampleColumns_ = 28;
    sampleLength_ = sizesOfLayers_[0];
    numberOfLabels_ = sizesOfLayers_[sizesOfLayers_.length - 1];
    
    FileInputStream fileInputStream = null;
    DataInputStream dataInputStream = null;
    
    trainingSet_ = new double[numberOfTrainingSamples_][sampleLength_];
    trainingLabels_ = new double[numberOfTrainingSamples_][numberOfLabels_];
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      for(int j = 0;j < numberOfLabels_;j++){
        trainingLabels_[i][j] = 0;
      }
    }
    
    // Load the database's training set data. =================================
    fileInputStream = new FileInputStream(
                           "data/training_data/my_data/database_training_set");
    dataInputStream = new DataInputStream(fileInputStream);
    
    int numberOfSamples = dataInputStream.readInt();
    for(int i = 0;i < numberOfSamples;i++){
      
      for(int j = 0;j < sampleRows_;j++){
        for(int k = 0;k < sampleColumns_;k++){
          trainingSet_[i][j * sampleColumns_ + k] =
                                           ((double)dataInputStream.readInt());
        }
      }
    }
    dataInputStream.close();
    
    // Load the database's training labels data. ==============================
    fileInputStream = new FileInputStream(
                        "data/training_data/my_data/database_training_labels");
    dataInputStream = new DataInputStream(fileInputStream);
    
    numberOfSamples = dataInputStream.readInt();
    for(int i = 0;i < numberOfSamples;i++){
      trainingLabels_[i][dataInputStream.readInt()] = 1;
    }
    dataInputStream.close();
    
    // Load the MNIST training set data. ======================================
    fileInputStream = new FileInputStream(
                      "data/training_data/MNIST_data/train-images.idx3-ubyte");
    fileInputStream.skip(16);
    
    byte[] sampleBuffer = new byte[sampleLength_];
    
    for(int i = 0;i < numberOfMNISTTrainingSamples_;i++){
      fileInputStream.read(sampleBuffer);
      
      for(int j = 0;j < sampleLength_;j++){
        trainingSet_[i + numberOfDatabaseTrainingSamples_][j] =
                                            ((double)(sampleBuffer[j] & 0xFF));
      }
    }
    fileInputStream.close();
    
    // Load the MNIST training labels data. ===================================
    fileInputStream = new FileInputStream(
                      "data/training_data/MNIST_data/train-labels.idx1-ubyte");
    fileInputStream.skip(8);
    
    byte[] labelBuffer = new byte[1];
    
    for(int i = 0;i < numberOfMNISTTrainingSamples_;i++){
      fileInputStream.read(labelBuffer);
      trainingLabels_[i + numberOfDatabaseTrainingSamples_]
                     [(int)(labelBuffer[0] & 0xFF)] = 1;
    }
    fileInputStream.close();
    
    // Load the MNIST testing set data. =======================================
    numberOfTestingSamples_ = 10000;
    testingSet_ = new double[numberOfTestingSamples_][sampleLength_];
    
    fileInputStream = new FileInputStream(
                        "data/testing_data/MNIST_data/t10k-images.idx3-ubyte");
    fileInputStream.skip(16);

    for(int i = 0;i < numberOfTestingSamples_;i++){
      fileInputStream.read(sampleBuffer);
      
      for(int j = 0;j < sampleLength_;j++){
        testingSet_[i][j] = ((double)(sampleBuffer[j] & 0xFF));
      }
    }
    fileInputStream.close();
    
    // Load the MNIST testing labels data. ====================================
    testingLabels_ = new int[numberOfTestingSamples_];
    
    fileInputStream = new FileInputStream(
                        "data/testing_data/MNIST_data/t10k-labels.idx1-ubyte");
    fileInputStream.skip(8);
    
    for(int i = 0;i < numberOfTestingSamples_;i++){
      fileInputStream.read(labelBuffer);
      
      testingLabels_[i] = (int)(labelBuffer[0] & 0xFF);
    }
    fileInputStream.close();
    
    this.prepareData();
  }
  
  public void train() throws IOException{
    numberOfEpochs_ = 200;
    batchSize_ = 60;
    gamma_ = 0.25;
    
    double[][] trainingSetBuffer = new double[numberOfTrainingSamples_]
                                             [sampleLength_];
    int distortFrequency = 10;
    
    for(int epoch = 0;epoch < numberOfEpochs_;epoch++){
      System.out.println("Epoch: " + epoch);
      
      // Rotate, scale and translate each training sample, to virtually
      // increase the training set's size. The distortions are applied on the
      // initial training set and not on the already distorted.
      if(epoch % distortFrequency == 0 && epoch != 0){
        System.out.println("Distorting the training set...");
        // On the first distortion, create the training set buffer.
        // On every other distortion, load the training set from the buffer.
        if(epoch == distortFrequency){
          for(int i = 0;i < numberOfTrainingSamples_;i++){
            for(int j = 0;j < sampleLength_;j++){
              trainingSetBuffer[i][j] = trainingSet_[i][j];
            }
          }
        }
        else{
          for(int i = 0;i < numberOfTrainingSamples_;i++){
            for(int j = 0;j < sampleLength_;j++){
              trainingSet_[i][j] = trainingSetBuffer[i][j];
            }
          }
        }
        
        distort(trainingSet_, sampleRows_, sampleColumns_);
        System.out.println("Distortion is done!");
      }
      
      // Actually train the neural network.
      for(int batch = 0;batch < numberOfTrainingSamples_ / batchSize_;batch++){
        neuralNetwork_.train(Arrays.copyOfRange(trainingSet_,
                                 batch * batchSize_, numberOfTrainingSamples_),
                             Arrays.copyOfRange(trainingLabels_,
                                 batch * batchSize_, numberOfTrainingSamples_),
                            batchSize_, 1, gamma_);
      }
      
      // Test the result on each epoch.
      int correctAnswerCounter = 0;
      double[] output = new double[10];
      for(int i = 0;i < numberOfTestingSamples_;i++){
        output = neuralNetwork_.feedForward(testingSet_[i]);

        double max = output[0];
        int index = 0;
        for(int j = 0;j < 10;j++){
          if(output[j] > max){
            max = output[j];
            index = j;
          }
        }

        if(index == testingLabels_[i]){
          correctAnswerCounter++;
        }
      }
      System.out.println(correctAnswerCounter + " correct answers!");
      
    }
    neuralNetwork_.saveNetwork(
        "data/trained_networks/network_tr" + numberOfTrainingSamples_ + "_e" +
         numberOfEpochs_ + "_b" + batchSize_ + "_g" + gamma_ + "_mixed" +
         numberOfDatabaseTrainingSamples_ + "-" +
         numberOfMNISTTrainingSamples_ + "_distort");
  }
  
  /** \brief Training and testing data are converted so that they only have
   *         binary values(-1 or 1).
   */
  public void prepareData(){
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      for(int j = 0;j < sampleLength_;j++){
        if(trainingSet_[i][j] > 0){
          trainingSet_[i][j] = 1;
        }
        else{
          trainingSet_[i][j] = -1;
        }
      }
    }
    
    for(int i = 0;i < numberOfTestingSamples_;i++){
      for(int j = 0;j < sampleLength_;j++){
        if(testingSet_[i][j] > 0){
          testingSet_[i][j] = 1;
        }
        else{
          testingSet_[i][j] = -1;
        }
      }
    }
  }
  
  /** \brief Applies random affine transformations on a set of data. 
   * 
   *  \param data The set of data on which to apply the transformations.
   *  \param sampleRows The number of rows of each sample if it is mapped on
   *                    a grid.
   *  \param sampleColumns The number of columns of each sample if it is mapped
   *                       on a grid.
   */
  public void distort(double[][] data, int sampleRows, int sampleColumns){
    Random random = new Random();
    double destortionType, parameter;
    Mat trfMtx = new Mat(2, 3, CvType.CV_64F);
    Mat image = new Mat(sampleRows, sampleColumns, CvType.CV_64F);
    
    for(int i = 0;i < data.length;i++){
      destortionType = random.nextDouble();

      if(destortionType < 0.25){ // Rotating. [-pi/12, pi/12).
        parameter = ((2 * random.nextDouble() - 1) / 12) * Math.PI; // Angle.
        trfMtx.put(0, 0, Math.cos(parameter)); trfMtx.put(0, 1, Math.sin(parameter)); trfMtx.put(0, 2, 0);
        trfMtx.put(1, 0, -Math.sin(parameter)); trfMtx.put(1, 1, Math.cos(parameter)); trfMtx.put(1, 2, 0);
      }
      else if(destortionType < 0.5){  // Scaling. [0.85, 1.15).
        parameter = ((2 * random.nextDouble() - 1) * 15 / 100) + 1; // Volume for horizontal axis.
        trfMtx.put(0, 0, parameter); trfMtx.put(0, 1, 0); trfMtx.put(0, 2, 0);
        
        parameter = ((2 * random.nextDouble() - 1) * 15 / 100) + 1; // Volume for vertical axis.
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
      
      for(int j = 0;j < sampleRows;j++){
        for(int k = 0;k < sampleColumns;k++){
          image.put(j, k, data[i][j * sampleColumns + k]);
        }
      }
      
      Imgproc.warpAffine(image, image, trfMtx, image.size());
      
      for(int j = 0;j < sampleRows;j++){
        for(int k = 0;k < sampleColumns;k++){
          data[i][j * sampleColumns + k] = image.get(j, k)[0];
        }
      }
      
    }
  }

  private int numberOfDatabaseTrainingSamples_;
  private int numberOfMNISTTrainingSamples_;
  private int numberOfTrainingSamples_;
  
  private int sampleRows_;
  private int sampleColumns_;
  private int sampleLength_;
  private int numberOfLabels_;
  
  private int numberOfTestingSamples_;
};
