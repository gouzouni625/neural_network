package main.trainers;

import java.util.Arrays;

import main.distorters.Distorter;
import main.utilities.data.DataSet;

import org.opencv.core.Core;

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
    DataSet trainingSet = DataSet.loadIDXFormat(trainingSetPath_, trainingLabelsPath_);
    DataSet testingSet = DataSet.loadIDXFormat(testingSetPath_, testingLabelsPath_);

    trainingSet_ = new double[trainingSet.size()][sampleLength_];
    trainingLabels_ = new double[trainingSet.size()][numberOfLabels_];
    for(int i = 0;i < trainingSet.size();i++){
      for(int j = 0;j < sampleLength_;j++){
        trainingSet_[i][j] = ((double)(trainingSet.get(i).data_[j] & 0xFF)) / 127.5 - 1;
      }

      for(int j = 0;j < numberOfLabels_;j++){
        trainingLabels_[i][j] = 0;
      }

      trainingLabels_[i][trainingSet.get(i).label_ & 0xFF] = 1;
    }

    testingSet_ = new double[testingSet.size()][sampleLength_];
    testingLabels_ = new int[testingSet.size()];
    for(int i = 0;i < testingSet.size();i++){
      for(int j = 0;j < sampleLength_;j++){
        testingSet_[i][j] = ((double)(testingSet.get(i).data_[j] & 0xFF)) / 127.5 - 1;
      }

      testingLabels_[i] = testingSet.get(i).label_ & 0xFF;
    }

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
      double[] output = new double[numberOfLabels_];
      for(int i = 0;i < numberOfTestingSamples_;i++){
        output = neuralNetwork_.feedForward(testingSet_[i]);

        double max = output[0];
        int index = 0;
        for(int j = 0;j < numberOfLabels_;j++){
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

        System.out.println("Found best accuracy, saving the neural network!");
        neuralNetwork_.saveNetwork(neuralNetworkSavePath_);
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
