package trainers;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

/** \class MNISTTrainer class. Trains a neural network on the MNIST database
 *         of handwritten equation. The database can be found at
 *         http://yann.lecun.com/exdb/mnist/  
 */
public class MNISTTrainer extends Trainer{
  
  public MNISTTrainer(int[] sizesOfLayers){
    super(sizesOfLayers);
  }
  
  public void load() throws IOException{
    numberOfTrainingSamples_ = 60000;
    numberOfTestingSamples_ = 10000;
    sampleLength_ = sizesOfLayers_[0];
    numberOfLabels_ = sizesOfLayers_[sizesOfLayers_.length - 1];
    
    // Load the training set. =================================================
    FileInputStream fileInputStream = new FileInputStream(
                      "data/training_data/MNIST_data/train-images.idx3-ubyte");
    fileInputStream.skip(16);
    
    trainingSet_ = new double[numberOfTrainingSamples_][sampleLength_];
    byte[] sampleBuffer = new byte[sampleLength_];
    
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      fileInputStream.read(sampleBuffer);
      
      for(int j = 0;j < sampleLength_;j++){
        trainingSet_[i][j] = ((double)(sampleBuffer[j] & 0xFF)) / 127.5 - 1;
      }
    }
    fileInputStream.close();
    
    // Load the training labels. ==============================================
    fileInputStream = new FileInputStream(
                      "data/training_data/MNIST_data/train-labels.idx1-ubyte");
    fileInputStream.skip(8);
    
    trainingLabels_ = new double[numberOfTrainingSamples_][numberOfLabels_];
    byte[] labelBuffer = new byte[1];
    
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      fileInputStream.read(labelBuffer);
      
      for(int j = 0;j < numberOfLabels_;j++){
        trainingLabels_[i][j] = 0;
      }
      
      trainingLabels_[i][(int)(labelBuffer[0] & 0xFF)] = 1;
    }
    fileInputStream.close();
    
    // Load the testing set. ==================================================
    fileInputStream = new FileInputStream(
                        "data/testing_data/MNIST_data/t10k-images.idx3-ubyte");
    fileInputStream.skip(16);
    
    testingSet_ = new double[numberOfTestingSamples_][sampleLength_];
    
    for(int i = 0;i < numberOfTestingSamples_;i++){
      fileInputStream.read(sampleBuffer);
      
      for(int j = 0;j < sampleLength_;j++){
        testingSet_[i][j] = ((double)(sampleBuffer[j] & 0xFF)) / 127.5 - 1;
      }
    }
    fileInputStream.close();
    
    // Load the testing labels. ===============================================
    fileInputStream = new FileInputStream(
                        "data/testing_data/MNIST_data/t10k-labels.idx1-ubyte");
    fileInputStream.skip(8);
    
    testingLabels_ = new int[numberOfTestingSamples_];
    
    for(int i = 0;i < numberOfTestingSamples_;i++){
      fileInputStream.read(labelBuffer);
      testingLabels_[i] = (int)(labelBuffer[0] & 0xFF);
    }
    fileInputStream.close();
  }
  
  public void train() throws IOException{
    numberOfEpochs_ = 10;
    batchSize_ = 1000;
    gamma_ = 0.25;
    
    for(int epoch = 0;epoch < numberOfEpochs_;epoch++){
      System.out.println("Epoch: " + epoch);
      
      // Train the neural network. ============================================
      for(int batch = 0;batch < numberOfTrainingSamples_ / batchSize_;batch++){
        neuralNetwork_.train(
                           Arrays.copyOfRange(trainingSet_, batch * batchSize_,
                                                     numberOfTrainingSamples_),
                           Arrays.copyOfRange(trainingLabels_,
                                 batch * batchSize_, numberOfTrainingSamples_),
                           batchSize_, 1, gamma_);
      }
      
      // Test the neural network. =============================================
      int correctAnswerCounter = 0;
      double[] output = new double[numberOfLabels_];
      for(int i = 0;i < numberOfTestingSamples_;i++){
        output = neuralNetwork_.feedForward(testingSet_[i]);
        
        double max = output[0];
        int index = 0;
        for(int j = 0;j < output.length;j++){
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
         numberOfEpochs_ + "_b" + batchSize_ + "_g" + gamma_);
  }

  private int numberOfTrainingSamples_;
  private int numberOfTestingSamples_;
  
  private int sampleLength_;
  private int numberOfLabels_;
};
