#include <iostream>
#include <fstream>
#include <cstdlib> /* srand, rand */
#include <ctime> /* time */
#include <cmath> /* exp */
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "neural_network.h"
#include "parameters.h"

using namespace neural_network;

void initializeNablaTheta(int numberOfThetas, float *deviceNablaTheta);
void backPropagation(float *deviceTrainingSet, float *deviceTrainingLabels, int numberOfSamples,
                     float* deviceWeights, float *deviceBiases, int numberOfLayers, int totalNeurons,
                     float *deviceNablaTheta, float *deviceActivations, float *deviceDeltas, int *sizesOfLayers,
                     int concurrentSamples);
void updateThetas(int trainingSetSize, int *sizesOfLayers, int numberOfLayers, int totalNeurons,
                  float *deviceWeights, float *deviceBiases, float *deviceNablaTheta, float gamma, float lambda);

void constantsInitialize(int *sizesOfLayers);

void initializeThetas(float *weights, int numberOfWeights, float *biases, int numberOfBiases);
void saveThetas(float *weights, int numberOfWeights, float *biases, int numberOfBiases);
void loadThetas(float *weights, int numberOfWeights, float *biases, int numberOfBiases);
void reloadTrainingSet(float* deviceTrainingSet, int rank, int processLoad, int sizeOfSample);

void distort(curandState *deviceStates, float *deviceTrainingSet, int processLoad, bool setupKernel);

int main(int argc, char **argv)
{
  // Program's input.
  // First input: 0 don't evaluate on each epoch.
  //              1 evaluate on each epoch.
  // Second input: 0 Initialize network's variables.
  //               1 Load network's variables.
  bool evaluateOnEachEpoch = true, loadNetworksVariables = false;
  if(argc == 2)
  {
    evaluateOnEachEpoch = (atoi(argv[1]) != 0);
    loadNetworksVariables = false;
  }
  else if(argc == 3)
  {
    evaluateOnEachEpoch = (atoi(argv[1]) != 0);
    loadNetworksVariables = (atoi(argv[2]) != 0);
  }

  // Defining network's variables.
  // Initializing weights and biases.
  int sizesOfLayers[] = {784, 30, 10};
  int numberOfLayers = NUMBER_OF_LAYERS;
  Network net(numberOfLayers, sizesOfLayers);
  int numberOfWeights = 0, numberOfBiases = 0;
  for(int i = 1;i < numberOfLayers;i++)
  {
    numberOfWeights += sizesOfLayers[i] * sizesOfLayers[i - 1];
    numberOfBiases += sizesOfLayers[i]; 
  }
  float *weights, *biases;
  weights = new float[numberOfWeights];
  biases = new float[numberOfBiases];

  if(loadNetworksVariables)
  {
    std::cout << "LOADING THETAS..." << std::endl;
    loadThetas(weights, numberOfWeights, biases, numberOfBiases);
  }
  else
  {
    std::cout << "INITIALIZING THETAS..." << std::endl;
    initializeThetas(weights, numberOfWeights, biases, numberOfBiases);
  }

  // Setting up MPI.
  int rank, numberOfProcesses;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

  // Quatro's GPU with id 2 doesn't work. Avoid using it.
  if(rank != 2)
    cudaSetDevice(rank);
  else
    cudaSetDevice(rank + 1);

  int processLoad = NUMBER_OF_SAMPLES / numberOfProcesses;

  // Loading training set and labels on CPU.
  std::ifstream trainingSetFile, trainingLabelsFile;
  trainingSetFile.open("../data/train-images.idx3-ubyte", std::ios::in);
  if(!trainingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TRAINING SET FILE..." << std::endl;
    exit(1);
  }
  trainingLabelsFile.open("../data/train-labels.idx1-ubyte", std::ios::in);
  if(!trainingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TRAINING LABELS' FILE..." << std::endl;
    exit(1);
  }
  trainingSetFile.seekg(16 + rank * processLoad * sizesOfLayers[0], std::ios::beg);
  trainingLabelsFile.seekg(8 + rank * processLoad, std::ios::beg);

  float *trainingSet = new float[processLoad * sizesOfLayers[0]];
  float *trainingLabels = new float[processLoad * sizesOfLayers[numberOfLayers - 1]];
  unsigned char *trainingSetBuffer = new unsigned char[sizesOfLayers[0]];
  unsigned char buffer;
  for(int i = 0;i < processLoad;i++)
  {
    trainingSetFile.read((char*)trainingSetBuffer, sizesOfLayers[0]);
    for(int j = 0;j < sizesOfLayers[0];j++)
      trainingSet[i * sizesOfLayers[0] + j] = (float)(trainingSetBuffer[j]) / 127.5 - 1;

    for(int j = 0;j < sizesOfLayers[numberOfLayers - 1];j++)
      trainingLabels[i * sizesOfLayers[numberOfLayers - 1] + j] = 0;
    trainingLabelsFile.read((char*)&buffer, 1);
    trainingLabels[i * sizesOfLayers[numberOfLayers - 1] + (unsigned int)(buffer)] = 1;
  }

  // Loading training set and labels on GPU.
  float *deviceTrainingSet, *deviceTrainingLabels;
  if(cudaMalloc((void**) &deviceTrainingSet, processLoad * sizesOfLayers[0] * sizeof(float)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceTrainingSet..." << std::endl;
    exit(1);
  }
  if(cudaMalloc((void**) &deviceTrainingLabels, processLoad * sizesOfLayers[numberOfLayers - 1] * sizeof(float))
     != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceTrainingLabels..." << std::endl;
    exit(1);
  }
  if(cudaMemcpy(deviceTrainingSet, trainingSet, processLoad * sizesOfLayers[0] * sizeof(float),
                cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error in cudaMemcpy for deviceTrainingSet..." << std::endl;
    exit(1);
  }
  if(cudaMemcpy(deviceTrainingLabels, trainingLabels, processLoad * sizesOfLayers[numberOfLayers - 1] * sizeof(float),
                cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error in cudaMemcpy for deviceTrainingLabels..." << std::endl;
    exit(1);
  }

  // Loading network's variables on GPU.
  int totalNeurons = 0;
  for(int i = 0;i < numberOfLayers - 1;i++)
    totalNeurons += sizesOfLayers[i + 1];

  float *deviceWeights, *deviceBiases, *deviceNablaTheta, *deviceActivations, *deviceDeltas;
  if(cudaMalloc((void**) &deviceNablaTheta,
     (numberOfWeights + numberOfBiases) * sizeof(float)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceNablaTheta..." << std::endl;
    exit(1);
  }
  if(cudaMalloc((void**) &deviceWeights, numberOfWeights * sizeof(float)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceWeights..." << std::endl;
    exit(1);
  }
  if(cudaMalloc((void**) &deviceBiases, numberOfBiases * sizeof(float)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceBiases..." << std::endl;
    exit(1);
  }
  if(cudaMalloc((void**) &deviceActivations, CONCURRENT_SAMPLES * totalNeurons * sizeof(float)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceActivations..." << std::endl;
    exit(1);
  }
  if(cudaMalloc((void**) &deviceDeltas, CONCURRENT_SAMPLES * totalNeurons * sizeof(float)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceDeltas..." << std::endl;
    exit(1);
  }
  if(cudaMemcpy(deviceWeights, weights, numberOfWeights * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error in cudaMemcpy for deviceWeights..." << std::endl;
    exit(1);
  }
  if(cudaMemcpy(deviceBiases, biases, numberOfBiases * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error in cudaMemcpy for deviceBiases..." << std::endl;
    exit(1);
  }
  constantsInitialize(sizesOfLayers);

  // Cleaning up CPU.
  delete[] trainingSet;
  delete[] trainingLabels;
  delete[] trainingSetBuffer;
  trainingSetFile.close();
  trainingLabelsFile.close();

  // Variables used for MPI_Allreduce.
  float *nablaTheta = new float[numberOfWeights + numberOfBiases];
  float *nablaThetaBuffer = new float[numberOfWeights + numberOfBiases];

  // Testing files and variables.
  std::ifstream testingSetFile, testingLabelsFile;
  testingSetFile.open("../data/t10k-images.idx3-ubyte", std::ios::in);
  if(!testingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TESTING SET FILE..." << std::endl;
    exit(1);
  }
  testingLabelsFile.open("../data/t10k-labels.idx1-ubyte", std::ios::in);
  if(!testingLabelsFile.is_open())
  {
    std::cout << "CANNOT OPEN TESTING LABELS' FILE..." << std::endl;
    exit(1);
  }
  float *testingSample = new float[sizesOfLayers[0]];
  unsigned char *testingSampleBuffer = new unsigned char[sizesOfLayers[0]];
  float *output = new float[sizesOfLayers[numberOfLayers - 1]];
  int bestEpoch = 0;
  int bestScore = 0;
  bool distortingCounter = false;
  bool setupKernel = true;

  // Cuda random-number-generator's variables.
  curandState *deviceStates;
  if(cudaMalloc((void **)&deviceStates, processLoad * sizeof(curandState)) != cudaSuccess)
  {
    std::cout << "Error in cudaMalloc for deviceStates..." << std::endl;
    exit(1);
  }

  // Program's main loop.
  cudaError_t err;
  for(int epoch = 0;epoch < TOTAL_EPOCHS;epoch++)
  {
    if(rank == 0)
      std::cout << "EPOCH:" << epoch << std::endl;

    // Distorting the training set.
    if(!(epoch % DISTORT_FREQUENCY) && epoch > 0)
    {
      if(distortingCounter)
      {
        std::cout << "RANK: " << rank << " RELOADING TRAINING SET..." << std::endl; 
        reloadTrainingSet(deviceTrainingSet, rank, processLoad, sizesOfLayers[0]);
        distortingCounter = false;
      }
      std::cout << "RANK: " << rank << " DISTORTING TRAINING SET..." << std::endl;
      distort(deviceStates, deviceTrainingSet, processLoad, setupKernel);
      distortingCounter = true;
      setupKernel = false;
      std::cout << "RANK: " << rank << " DONE WITH DISTORTING..." << std::endl;
    }

    for(int batch = 0;batch < processLoad / BATCH_SIZE;batch++)
    {
      initializeNablaTheta(numberOfWeights + numberOfBiases, deviceNablaTheta);

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for initializeNablaTheta " << err << std::endl;
        exit(1);
      }

      // Calculating nablaTheta using the back-propagation algorithm.
      backPropagation(&deviceTrainingSet[batch * BATCH_SIZE * sizesOfLayers[0]],
                      &deviceTrainingLabels[batch * BATCH_SIZE * sizesOfLayers[numberOfLayers - 1]],
                      BATCH_SIZE, deviceWeights, deviceBiases, numberOfLayers, totalNeurons, deviceNablaTheta,
                      deviceActivations, deviceDeltas, sizesOfLayers, CONCURRENT_SAMPLES);

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for backPropagation " << err << std::endl;
        exit(1);
      }

      if(cudaMemcpy(nablaTheta, deviceNablaTheta, (numberOfWeights + numberOfBiases) * sizeof(float),
                    cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        std::cout << "Error in cudaMemcpy for nablaTheta..." << std::endl;
        exit(1);
      }

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for first memcpy " << err << std::endl;
        exit(1);
      }

      // Combining the results from every process.
      MPI_Allreduce(nablaTheta, nablaThetaBuffer, numberOfWeights + numberOfBiases, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);

      if(cudaMemcpy(deviceNablaTheta, nablaThetaBuffer, (numberOfWeights + numberOfBiases) * sizeof(float),
                    cudaMemcpyHostToDevice) != cudaSuccess)
      {
        std::cout << "Error in cudaMemcpy for deviceNablaTheta..." << std::endl;
        exit(1);
      }

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for second memcpy " << err << std::endl;
        exit(1);
      }

      // Updating network's variables using gradient descent.
      updateThetas(numberOfProcesses * BATCH_SIZE, sizesOfLayers, numberOfLayers, totalNeurons, deviceWeights,
                   deviceBiases, deviceNablaTheta, GAMMA, LAMBDA);

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for updateThetas " << err << std::endl;
        exit(1);
      }
    }

    // Evaluating on test samples.
    if(rank == 0 && (evaluateOnEachEpoch || !(epoch % EVALUATE_FREQUENCY)))
    {
      if(cudaMemcpy(weights, deviceWeights, numberOfWeights * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
      {
       std::cout << "Error in cudaMemcpy for weights 2" << std::endl;
        exit(1);
      }
      if(cudaMemcpy(biases, deviceBiases, numberOfBiases * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        std::cout << "Error in cudaMemcpy for biases 2" << std::endl;
        exit(1);
      }

      net.set(weights, biases);

      testingSetFile.seekg(16, std::ios::beg);
      testingLabelsFile.seekg(8, std::ios::beg);

      int counter = 0;

      for(int i = 0;i < NUMBER_OF_TESTING_SAMPLES;i++)
      {
        int guess;
        int correct;
        testingSetFile.read((char*)testingSampleBuffer, sizesOfLayers[0]);
        for(int j = 0;j < sizesOfLayers[0];j++)
          testingSample[j] = (float)(testingSampleBuffer[j]) / 127.5 - 1;

        net.feedForward(testingSample, output);

        float max = output[0];
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

      if(counter >= bestScore)
      {
        bestScore = counter;
        bestEpoch = epoch;
      }

      std::cout << "EPOCH:" << epoch << " ";
      std::cout << counter << " correct answers!!!";
      std::cout << " ***** " << "BEST SCORE:" << bestScore << " BEST EPOCH:" << bestEpoch << std::endl;
    }

    // Saving network's variables.
    if(rank == 0 && !(epoch % SAVE_FREQUENCY) && (epoch > 0))
    {
      if(cudaMemcpy(weights, deviceWeights, numberOfWeights * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
      {
       std::cout << "Error in cudaMemcpy for weights 3" << std::endl;
        exit(1);
      }
      if(cudaMemcpy(biases, deviceBiases, numberOfBiases * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        std::cout << "Error in cudaMemcpy for biases 3" << std::endl;
        exit(1);
      }
      std::cout << "SAVING THETAS..." << std::endl;
      saveThetas(weights, numberOfWeights, biases, numberOfBiases);
    }
  }

  // Cleaning up CPU
  delete[] nablaTheta;
  delete[] nablaThetaBuffer;
  delete[] weights;
  delete[] biases;
  delete[] testingSample;
  delete[] testingSampleBuffer;
  delete[] output;
  testingSetFile.close();
  testingLabelsFile.close();

  // Cleaning up GPU.
  cudaFree(deviceTrainingSet);
  cudaFree(deviceTrainingLabels);
  cudaFree(deviceWeights);
  cudaFree(deviceBiases);
  cudaFree(deviceNablaTheta);
  cudaFree(deviceActivations);
  cudaFree(deviceDeltas);
  cudaFree(deviceStates);

  // Closing MPI.
  MPI_Finalize();
}

// Randomly initializes network's variables.
void initializeThetas(float *weights, int numberOfWeights, float *biases, int numberOfBiases)
{
  srand(time(NULL));
  for(int i = 0;i < numberOfWeights;i++)
    weights[i] = ((float) rand() / (RAND_MAX)) * 0.5 - 0.25;
  for(int i = 0;i < numberOfBiases;i++)
    biases[i] = ((float) rand() / (RAND_MAX)) * 0.5 - 0.25;
}

// Saves network's variables.
void saveThetas(float *weights, int numberOfWeights, float *biases, int numberOfBiases)
{
  std::ofstream weightsFile("weights.txt", std::ios::out);
  if(!weightsFile.is_open())
  {
    std::cout << "CANNOT OPEN WEIGHTS' FILE..." << std::endl;
    exit(1);
  }
  std::ofstream biasesFile("biases.txt", std::ios::out);
  if(!biasesFile.is_open())
  {
    std::cout << "CANNOT OPEN BIASES' FILE..." << std::endl;
    exit(1);
  }

  weightsFile.write((char*)weights, numberOfWeights * sizeof(float) / sizeof(char));
  biasesFile.write((char*)biases, numberOfBiases * sizeof(float) / sizeof(char));

  weightsFile.close();
  biasesFile.close();
}

// Loads network's variables.
void loadThetas(float *weights, int numberOfWeights, float *biases, int numberOfBiases)
{
  std::ifstream weightsFile("weights.txt", std::ios::in);
  if(!weightsFile.is_open())
  {
    std::cout << "WEIGHTS' FILE DOESN'T EXIST..." << std::endl;
    exit(1);
  }
  std::ifstream biasesFile("biases.txt", std::ios::in);
  if(!biasesFile.is_open())
  {
    std::cout << "BIASES' FILE DOESN'T EXIST..." << std::endl;
    exit(1);
  }

  weightsFile.read((char*)weights, numberOfWeights * sizeof(float) / sizeof(char));
  biasesFile.read((char*)biases, numberOfBiases * sizeof(float) / sizeof(char));

  weightsFile.close();
  biasesFile.close();
}

void reloadTrainingSet(float* deviceTrainingSet, int rank, int processLoad, int sizeOfSample)
{
  std::ifstream trainingSetFile;
  trainingSetFile.open("../data/train-images.idx3-ubyte", std::ios::in);
  if(!trainingSetFile.is_open())
  {
    std::cout << "CANNOT OPEN TRAINING SET FILE..." << std::endl;
    exit(1);
  }
  trainingSetFile.seekg(16 + rank * processLoad * sizeOfSample, std::ios::beg);

  float *trainingSet = new float[processLoad * sizeOfSample];
  unsigned char *trainingSetBuffer = new unsigned char[sizeOfSample];
  for(int i = 0;i < processLoad;i++)
  {
    trainingSetFile.read((char*)trainingSetBuffer, sizeOfSample);
    for(int j = 0;j < sizeOfSample;j++)
      trainingSet[i * sizeOfSample + j] = (float)(trainingSetBuffer[j]) / 127.5 - 1;
  }

  if(cudaMemcpy(deviceTrainingSet, trainingSet, processLoad * sizeOfSample * sizeof(float),
                cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error in cudaMemcpy for deviceTrainingSet..." << std::endl;
    exit(1);
  }

  trainingSetFile.close();
  delete[] trainingSet;
  delete[] trainingSetBuffer;
}
