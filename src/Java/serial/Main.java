import java.io.*;
import java.util.Arrays;

public class Main{ 

  public static void main(String[] args) throws Exception{

    // Set some parameters.
    int numberOfTrainingSamples = 60000;
    int numberOfTestingSamples = 10000;
    int numberOfEpochs = 200;
    int batchSize = 1000;
    double gamma = 0.25;

    // Create the neural network.
    int[] sizesOfLayers = {784, 30, 10};
    NeuralNetwork net = new NeuralNetwork(sizesOfLayers);


    // Load the training set.
    FileInputStream trainingSetStream = new FileInputStream("../../../data/train-images.idx3-ubyte");
    FileInputStream trainingLabelsStream = new FileInputStream("../../../data/train-labels.idx1-ubyte");

    trainingSetStream.skip(16);
    trainingLabelsStream.skip(8);

    double[][] trainingSet = new double[numberOfTrainingSamples][784];
    double[][] trainingLabels = new double[numberOfTrainingSamples][10];
    byte[] sampleBuffer = new byte[784];
    byte[] labelBuffer = new byte[1];

    for(int i = 0;i < numberOfTrainingSamples;i++){
      trainingSetStream.read(sampleBuffer);

      for(int j = 0;j < 784;j++){
        trainingSet[i][j] = ((double)(sampleBuffer[j] & 0xFF)) / 127.5 - 1;
      }

      for(int j = 0;j < 10;j++){
        trainingLabels[i][j] = 0;
      }
      trainingLabelsStream.read(labelBuffer);
      trainingLabels[i][(int)(labelBuffer[0] & 0xFF)] = 1;
    }

    trainingSetStream.close();
    trainingLabelsStream.close();

    // Load the testing set.
    FileInputStream testingSetStream = new FileInputStream("../../../data/t10k-images.idx3-ubyte");
    FileInputStream testingLabelsStream = new FileInputStream("../../../data/t10k-labels.idx1-ubyte");

    testingSetStream.skip(16);
    testingLabelsStream.skip(8);

    double[][] testingSet = new double[numberOfTestingSamples][784];
    int[] testingLabels = new int[numberOfTestingSamples];
    for(int i = 0;i < numberOfTestingSamples;i++){
      testingSetStream.read(sampleBuffer);

      for(int j = 0;j < 784;j++){
        testingSet[i][j] = ((double)(sampleBuffer[j] & 0xFF)) / 127.5 - 1;
      }

      testingLabelsStream.read(labelBuffer);
      testingLabels[i] = (int)(labelBuffer[0] & 0xFF);
    }

    testingSetStream.close();
    testingLabelsStream.close();

    // Main loop.
    for(int epoch = 0; epoch < numberOfEpochs;epoch++){

      System.out.println("Epoch: " + epoch);

      // Train.
      for(int batch = 0;batch < numberOfTrainingSamples / batchSize;batch++){
        net.train(Arrays.copyOfRange(trainingSet, batch * batchSize, numberOfTrainingSamples),
                  Arrays.copyOfRange(trainingLabels, batch * batchSize, numberOfTrainingSamples),
                  batchSize, 1, gamma);
      }

      // Test.
      int counter = 0;
      double[] output = new double[10];
      for(int i = 0;i < numberOfTestingSamples;i++){
        output = net.feedForward(testingSet[i]);

        double max = output[0];
        int guess = 0;
        for(int j = 0;j < 10;j++){
          if(output[j] > max){
            max = output[j];
            guess = j;
          }
        }

        if(guess == testingLabels[i]){
          counter++;
        }

      }

      System.out.println(counter + " correct answers!");
    }
  }

};
