#include <iostream>
#include <fstream>
#include <cstdlib>
#include "neural_network.h"

#define NUMBER_OF_SAMPLES 60000
#define NUMBER_OF_TESTING_SAMPLES 10000

#define NUMBER_OF_EPOCHS 200
#define BATCH_SIZE 1000
#define GAMMA 0.25

int main()
{
  // Network's form.
  std::vector<int> sizes;
  sizes.push_back(784);
  sizes.push_back(30);
  sizes.push_back(10);
  neural_network::NeuralNetwork net(sizes);

  // Loading the training set on the CPU.
  std::ifstream trainingSetFile, trainingLabelsFile;
  trainingSetFile.open("../../../data/train-images.idx3-ubyte", std::ios::in);
  if(!trainingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TRAINING SET'S FILE..." << std::endl;
    exit(1);
  }
  trainingLabelsFile.open("../../../data/train-labels.idx1-ubyte", std::ios::in);
  if(!trainingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TRAINING LABELS' FILE..." << std::endl;
    exit(1);
  }
  trainingSetFile.seekg(16, std::ios::beg);
  trainingLabelsFile.seekg(8, std::ios::beg);

  double **trainingSet = new double*[NUMBER_OF_SAMPLES];
  double **trainingLabels = new double*[NUMBER_OF_SAMPLES];
  unsigned char trainingSetBuffer[784];
  unsigned char buffer;
  for(int i = 0;i < NUMBER_OF_SAMPLES;i++)
  {
    trainingSet[i] = new double[784];
    trainingSetFile.read((char*)trainingSetBuffer, 784);

    for(int j = 0;j < 784;j++)
      trainingSet[i][j] = (double)trainingSetBuffer[j] / 127.5 - 1;

    trainingLabels[i] = new double[10];
    for(int j = 0;j < 10;j++)
      trainingLabels[i][j] = 0;
    trainingLabelsFile.read((char*)&buffer, 1);
    trainingLabels[i][(unsigned int)(buffer)] = 1;
  }

  // Testing files and variables.
  std::ifstream testingSetFile, testingLabelsFile;
  testingSetFile.open("../../../data/t10k-images.idx3-ubyte", std::ios::in);
  if(!testingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TESTING SET'S FILE..." << std::endl;
    exit(1);
  }
  testingLabelsFile.open("../../../data/t10k-labels.idx1-ubyte", std::ios::in);
  if(!testingLabelsFile.is_open())
  {
    std::cout << "CANNOT OPEN TESTING LABELS' FILE..." << std::endl;
    exit(1);
  }
  double *testingSample = new double[784];
  unsigned char testingSampleBuffer[784];
  //unsigned char buffer; already declared above.
  double *output = new double[10];

  // Program's main loop.
  for(int epoch = 0;epoch < NUMBER_OF_EPOCHS;epoch++)
  {
    std::cout << "EPOCH: " << epoch << std::endl;

    for(int batch = 0;batch < NUMBER_OF_SAMPLES / BATCH_SIZE;batch++)
    {
      // Training the Neural Network.
      net.train(&trainingSet[batch * BATCH_SIZE], &trainingLabels[batch * BATCH_SIZE], BATCH_SIZE, 1, GAMMA);
    }
 
    // Testing.
    testingSetFile.seekg(16, std::ios::beg);
    testingLabelsFile.seekg(8, std::ios::beg);

    int counter = 0;

    for(int i = 0;i < NUMBER_OF_TESTING_SAMPLES;i++)
    {
      int guess;
      int correct;
      testingSetFile.read((char*)testingSampleBuffer, 784);
      for(int j = 0;j < 784;j++)
        testingSample[j] = (double)testingSampleBuffer[j] / 127.5 - 1;

      net.feedForward(testingSample, output);

      double max = output[0];
      guess = 0;
      for(int j = 1;j < 10;j++)
      {
        if(output[j] > max)
        {
          max = output[j];
          guess = j;
        }
      }

      testingLabelsFile.read((char*)&buffer, 1);
      correct = (unsigned int)buffer;

      //std::cout << "Test " << i << " guess = " << guess << " correct = " << correct << std::endl;
      if(guess == correct)
        counter++;
    }

    std::cout << counter << " correct answers!" << std::endl;
    std::cout << (double)counter / 100 << "\% accuracy!" << std::endl;
  }

  net.saveNetwork();

  // Cleaning up.
  trainingSetFile.close();
  trainingLabelsFile.close();
  testingSetFile.close();
  testingLabelsFile.close();

  delete[] output;
  delete[] testingSample;
  for(int i = 0;i < NUMBER_OF_SAMPLES;i++)
  {
    delete[] trainingSet[i];
    delete[] trainingLabels[i];
  }
  delete[] trainingSet;
  delete[] trainingLabels;

}
