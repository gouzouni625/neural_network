package trainers;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.Arrays;

import org.opencv.core.Core;

import distorters.Distorter;

public class SimpleTrainer extends Trainer{

  public SimpleTrainer(int[] sizesOfLayers, String trainingSetPath,
                                            String trainingLabelsPath,
                                            String testingSetPath,
                                            String testingLabelsPath,
                                            Distorter distorter){
    super(sizesOfLayers, trainingSetPath, trainingLabelsPath,
          testingSetPath, testingLabelsPath, distorter);
    
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  }
  
  @Override
  public void load() throws Exception{
    numberOfLabels_ = sizesOfLayers_[sizesOfLayers_.length - 1];
    
    FileInputStream fileInputStream = null;
    DataInputStream dataInputStream = null;
    
    // Initialize training set and testing labels.
    trainingSet_ = new double[numberOfTrainingSamples_][sampleLength_];
    trainingLabels_ = new double[numberOfTrainingSamples_][numberOfLabels_];
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      for(int j = 0;j < numberOfLabels_;j++){
        trainingLabels_[i][j] = 0;
      }
    }
    
    // Initialize testing set and testing labels.
    testingSet_ = new double[numberOfTestingSamples_][sampleLength_];
    testingLabels_ = new int[numberOfTestingSamples_];
    
    // Load the training set data. =============================================
    fileInputStream = new FileInputStream(trainingSetPath_);
    dataInputStream = new DataInputStream(fileInputStream);
    
    // Just read the first int to discard it. It will not be used. Instead,
    // numberOfTrainingSamples_ will be used to give the ability to use less
    // than total number of samples. It numberOfSamples_ > numberOfSamples, then
    // an exception will be thrown.
    int numberOfSamples = dataInputStream.readInt();
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      for(int j = 0;j < sampleLength_;j++){
        trainingSet_[i][j] = ((double)dataInputStream.readInt());
      }
    }
    dataInputStream.close();
  
    // Load the training labels data. ==========================================
    fileInputStream = new FileInputStream(trainingLabelsPath_);
    dataInputStream = new DataInputStream(fileInputStream);
  
    numberOfSamples = dataInputStream.readInt();
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      trainingLabels_[i][dataInputStream.readInt()] = 1;
    }
    dataInputStream.close();
  
    // Load the testing set data. ==============================================
    fileInputStream = new FileInputStream(testingSetPath_);
    dataInputStream = new DataInputStream(fileInputStream);
    
    // Just read the first int to discard it. It will not be used. Instead,
    // numberOfTrainingSamples_ will be used to give the ability to use less
    // than total number of samples. It numberOfSamples_ > numberOfSamples, then
    // an exception will be thrown.
    numberOfSamples = dataInputStream.readInt();
    for(int i = 0;i < numberOfTestingSamples_;i++){
      for(int j = 0;j < sampleLength_;j++){
        testingSet_[i][j] = ((double)dataInputStream.readInt());
      }
    }
    dataInputStream.close();
    
    // Load the testing labels data. ===========================================
    fileInputStream = new FileInputStream(testingLabelsPath_);
    dataInputStream = new DataInputStream(fileInputStream);
    
    numberOfSamples = dataInputStream.readInt();
    for(int i = 0;i < numberOfTestingSamples_;i++){
      testingLabels_[i] = dataInputStream.readInt();
    }
    fileInputStream.close();
    
    this.prepareData();
  }

  @Override
  public void train() throws Exception{    
    double[][] trainingSetBuffer = new double[numberOfTrainingSamples_]
                                             [sampleLength_];
    
    double bestAccuracy = 0;
    for(int epoch = 0;epoch < numberOfEpochs_;epoch++){
      if(!quiet_){
        System.out.println("Epoch: " + epoch);
      }
      
      // Rotate, scale and translate each training sample, to virtually
      // increase the training set's size. The distortions are applied on the
      // initial training set and not on the already distorted.
      if(epoch % distorter_.getDistortFrequency() == 0 && epoch != 0){
        if(!quiet_){
          System.out.println("Distorting the training set...");
        }
        // On the first distortion, create the training set buffer.
        // On every other distortion, load the training set from the buffer.
        if(epoch == distorter_.getDistortFrequency()){
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
        
        trainingSet_ = distorter_.distort(trainingSet_);
        if(!quiet_){
          System.out.println("Distortion is done!");
        }
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
      
      double accuracy = (double)correctAnswerCounter / numberOfTestingSamples_;
      accuracy *= 100;
      if(accuracy > bestAccuracy){
        bestAccuracy = accuracy;
      }
      
      if(!quiet_){
        System.out.println(correctAnswerCounter + "/" +
                                 numberOfTestingSamples_ + " correct answers!");

        System.out.println(accuracy + "%");
        System.out.println("Best so far: " + bestAccuracy + "%");
      }
      
    }
  } 
}