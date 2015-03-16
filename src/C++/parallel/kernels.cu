#include <iostream>
#include <curand_kernel.h>
#include "parameters.h"

__device__ __constant__ int deviceSizesOfLayers[NUMBER_OF_LAYERS];

// deviceSizesOfLayers has a constant value through the execution of the program.
// Load it on the constant memory of the GPU for faster reading.
void constantsInitialize(int *sizesOfLayers)
{
  if(cudaMemcpyToSymbol(deviceSizesOfLayers, sizesOfLayers, NUMBER_OF_LAYERS * sizeof(int)) != cudaSuccess)
  {
    std::cout << "Error in cudaMemcpyToSymbol for deviceSizesOfLayers..." << std::endl;
    exit(1);
  }
}

// Initializes nablaTheta to zero.
__global__
void initializeNablaThetaCU(int numberOfThetas, float *nablaTheta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= 0 && index < numberOfThetas)
    nablaTheta[index] = 0;
}

// initializeNablaThetaCU wrapper.
void initializeNablaTheta(int numberOfThetas, float *deviceNablaTheta)
{
  // Each cuda thread will work on one variable.
  dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1);
  dim3 dimGrid((int)(numberOfThetas / MAX_THREADS_PER_BLOCK) + 1, 1);
  initializeNablaThetaCU<<<dimGrid, dimBlock>>>(numberOfThetas, deviceNablaTheta);

  cudaError_t err = cudaDeviceSynchronize();  
  if(err != cudaSuccess)  
  {
    std::cout << "Error in synchronizing for initializeNablaThetaCU " << err << std::endl;
    exit(1);
  }
}

// Neurons' activation function.
__device__
float sigmoid(float z)
{
  return (1 / (1 + exp(-z)));
}

// Calculates the output of each Neuron in the network given it's input.
__global__
void activations(float *trainingSet, int sample, float *weights, float *biases, int numberOfLayers,
                 int totalNeurons, int currentLayer, float *a)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; // Which neuron am I working on?
  int currentLayerSize = deviceSizesOfLayers[currentLayer + 1];

  sample += blockIdx.y;

  int inputSize = deviceSizesOfLayers[currentLayer];
  int weightsIndex = index * inputSize;

  for(int i = 0;i < currentLayer;i++)
    index += deviceSizesOfLayers[i + 1];

  int aIndex = index + blockIdx.y * totalNeurons;

  int inputIndex = 0;
  float *input;
  if(currentLayer == 0)
  {
    input = trainingSet;
    inputIndex = sample * inputSize;
  }
  else
  {
    input = a;
    for(int i = 0;i < currentLayer - 1;i++)
      inputIndex += deviceSizesOfLayers[i + 1];
    inputIndex += blockIdx.y * totalNeurons;
  }

  // Each thread on this block is working on the same layer and thus using the same input.
  // Cooperatively load the input on the shared memory for faster reading.
  __shared__ float sharedInput[MAX_SIZE_OF_LAYER];
  float threadsAvailable = (MAX_THREADS_PER_BLOCK < currentLayerSize) ? MAX_THREADS_PER_BLOCK : currentLayerSize;
  for(int i = 0;i < (int)(inputSize / threadsAvailable) + 1;i++)
  {
    int tempIndex = threadIdx.x + i * threadsAvailable;
    if(tempIndex >= 0 && tempIndex < inputSize)
      sharedInput[tempIndex] = input[inputIndex + tempIndex];
  }

  __shared__ float sharedBiase;
  if(threadIdx.x == 0)
    sharedBiase = biases[index];

  for(int i = 0;i < currentLayer;i++)
    weightsIndex += deviceSizesOfLayers[i] * deviceSizesOfLayers[i + 1];

  __syncthreads();
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < 0 || index >= currentLayerSize)
    return;

  float sum = 0;
  for(int i = 0;i < inputSize;i++)
    sum += sharedInput[i] * weights[weightsIndex + i];

  a[aIndex] = sigmoid(sum + sharedBiase);
}

// Calculates the error of each neuron of the network.
__global__
void deltas(float *deltas, float *weights, float *a, int numberOfLayers, int currentLayer, int totalNeurons)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; // Which neuron am I working on?
  int currentLayerSize = deviceSizesOfLayers[currentLayer + 1];

  int weightsIndex = index;

  for(int i = 0;i < currentLayer;i++)
    index += deviceSizesOfLayers[i + 1];

  int adIndex = index + blockIdx.y * totalNeurons;

  int deltasIndex = 0;

  for(int i = 0;i < currentLayer + 1;i++)
  {
    deltasIndex += deviceSizesOfLayers[i + 1];
    weightsIndex += deviceSizesOfLayers[i] * deviceSizesOfLayers[i + 1];
  }

  deltasIndex += blockIdx.y * totalNeurons;

  // Each thread on this block is working on the same layer and thus using the same delta values.
  // Cooperatively load these values to shared memory for faster reading.
  __shared__ float sharedDeltas[MAX_SIZE_OF_LAYER];
  int nextLayerSize = deviceSizesOfLayers[currentLayer + 2]; // How many deltas I want to load to shared memory.
  float threadsAvailable = (MAX_THREADS_PER_BLOCK < currentLayerSize) ? MAX_THREADS_PER_BLOCK : currentLayerSize;
  for(int i = 0;i < (int)(nextLayerSize / threadsAvailable) + 1;i++)
  {
    int tempIndex = threadIdx.x + i * threadsAvailable;
    if(tempIndex >= 0 && tempIndex < nextLayerSize)
      sharedDeltas[tempIndex] = deltas[deltasIndex + tempIndex];
  }

  __syncthreads();
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < 0 || index >= currentLayerSize)
    return;

  float sum = 0, aTemp = a[adIndex];
  for(int i = 0;i < nextLayerSize;i++)
   sum += weights[weightsIndex + i * currentLayerSize] * sharedDeltas[i];

  sum *= aTemp * (1 - aTemp);

  deltas[adIndex] = sum;
}

__global__
void firstDeltas(float *trainingLabels, int sample, float *a, float *deltas, int numberOfLayers, int totalNeurons)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; // Which neuron am I working on?
  int currentLayerSize = deviceSizesOfLayers[numberOfLayers - 1];
  if(index < 0 || index >= currentLayerSize)
    return;

  sample += blockIdx.y; // Which sample am I working on? (There are several samples being fed concurrently to the
                        // network)

  int adIndex = index;
  for(int i = 0;i < numberOfLayers - 2;i++)
    adIndex += deviceSizesOfLayers[i + 1];

  adIndex += blockIdx.y * totalNeurons;

  deltas[adIndex] = a[adIndex] - trainingLabels[sample * currentLayerSize + index];
}

// Calculates the derivative of the const function with respect to each network's variable.
__global__
void nablas(float *trainingSet, int sample, int numberOfLayers, int totalNeurons, int concurrentSamples,
            float *nablaTheta, float *deltas, float *a)
{
  int neuron = blockIdx.x * blockDim.x + threadIdx.x; // Which neuron am I working on?

  int sum = 0, layer = numberOfLayers - 1;
  for(int i = 0;i < numberOfLayers - 2;i++)
  {
    sum += deviceSizesOfLayers[i + 1];
    if(sum > neuron)
    {
      layer = i + 1;
      break;
    }
  }

  int theta = blockIdx.y * blockDim.y + threadIdx.y; // Which variable am I working on?

  float *input;
  int inputIndex, inputCoefficient;
  if(layer == 1)
  {
    input = trainingSet;
    inputCoefficient = deviceSizesOfLayers[0];
    inputIndex = sample * deviceSizesOfLayers[0] + theta - 1;
  }
  else
  {
    input = a;
    inputCoefficient = totalNeurons;
    inputIndex = theta - 1;
    for(int i = 0;i < layer - 2;i++)
      inputIndex += deviceSizesOfLayers[i + 1];
  }

  int nablaThetaIndex = theta;
  int previousNeurons = 0;
  for(int i = 0;i < layer - 1;i++)
    previousNeurons += deviceSizesOfLayers[i + 1];
  for(int i = 0;i < neuron - previousNeurons;i++)
    nablaThetaIndex += deviceSizesOfLayers[layer - 1] + 1;
  for(int i = 0;i < layer - 1;i++)
    nablaThetaIndex += (deviceSizesOfLayers[i] + 1) * deviceSizesOfLayers[i + 1];

  // Every thread on this block is working on the same neuron and thus using the same deltas.
  // Cooperatively load these deltas on the shared memory for faster reading.
  __shared__ float sharedDeltas[MAX_CONCURRENT_SAMPLES];
  int inputSize = deviceSizesOfLayers[layer - 1], currentLayerSize = deviceSizesOfLayers[layer];
  float threadsAvailable = (MAX_THREADS_PER_BLOCK < currentLayerSize) ? MAX_THREADS_PER_BLOCK : currentLayerSize;
  for(int i = 0;i < (int)(concurrentSamples / threadsAvailable) + 1;i++)
  {
    int tempIndex = threadIdx.y + i * threadsAvailable;
    if(tempIndex >= 0 && tempIndex < concurrentSamples)
      sharedDeltas[tempIndex] = deltas[neuron + tempIndex * totalNeurons];
  }

  // Ensuring that all delta values have been loaded and dismissing the unneeded threads.
  __syncthreads();
  if(neuron < 0 || neuron >= totalNeurons)
    return;
  if(theta < 0 || theta > inputSize)
    return;

  // Calculating the derivate of the cost function with respect to the variable this thread is working on.
  float sum2 = nablaTheta[nablaThetaIndex];
  for(int i = 0;i < concurrentSamples;i++)
  {
    if(theta != 0)
      sum2 += input[inputIndex + i * inputCoefficient] * sharedDeltas[i];
    else
      sum2 += sharedDeltas[i];
  }

  nablaTheta[nablaThetaIndex] = sum2;
}

void backPropagation(float *deviceTrainingSet, float *deviceTrainingLabels, int numberOfSamples,
                     float* deviceWeights, float *deviceBiases, int numberOfLayers, int totalNeurons,
                     float *deviceNablaTheta, float *deviceActivations, float *deviceDeltas, int *sizesOfLayers,
                     int concurrentSamples)
{
  cudaError_t err;
  for(int sample = 0;sample < numberOfSamples;sample += concurrentSamples)
  {
    // Calculating activations (feed forward).
    for(int i = 0;i < numberOfLayers - 1;i++) // For each layer...
    {
      // Each cuda thread works on a specific neuron and a specific sample.
      dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1);
      dim3 dimGrid((int)(sizesOfLayers[i + 1] / MAX_THREADS_PER_BLOCK) + 1, concurrentSamples);
      activations<<<dimGrid, dimBlock>>>(deviceTrainingSet, sample, deviceWeights, deviceBiases, numberOfLayers,
                                         totalNeurons, i, deviceActivations);

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for activations " << err << std::endl;
        exit(1);
      }
    }

    // Calculate deltas (backpropagation).
    // Each cuda thread works on a specific neuron and a specific sample.
    dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1);
    dim3 dimGrid((int)(sizesOfLayers[numberOfLayers - 1] / MAX_THREADS_PER_BLOCK) + 1, concurrentSamples);
    firstDeltas<<<dimGrid, dimBlock>>>(deviceTrainingLabels, sample, deviceActivations, deviceDeltas,
                                       numberOfLayers, totalNeurons);

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
      std::cout << "Error in synchronizing for firstDeltas " << err << std::endl;
      exit(1);
    }

    for(int i = numberOfLayers - 3;i >= 0;i--) // For each layer(beginning for the end, without the input layer).
    {
      dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1);
      dim3 dimGrid((int)(sizesOfLayers[i + 1] / MAX_THREADS_PER_BLOCK) + 1, concurrentSamples);
      deltas<<<dimGrid, dimBlock>>>(deviceDeltas, deviceWeights, deviceActivations, numberOfLayers, i, totalNeurons);

      err = cudaDeviceSynchronize();
      if(err != cudaSuccess)
      {
        std::cout << "Error in synchronizing for deltas " << err << std::endl;
        exit(1);
      }
    }

      // Calculate nablaThetas.
      dim3 dimBlock2(1, MAX_THREADS_PER_BLOCK);
      int maxSizeOfLayer = sizesOfLayers[0];
      for(int i = 1;i < numberOfLayers - 1;i++)
        if(sizesOfLayers[i] > maxSizeOfLayer)
          maxSizeOfLayer = sizesOfLayers[i];

      // Each cuda thread works on a specific variable of a specific neuron.
      dim3 dimGrid2(totalNeurons, (int)((maxSizeOfLayer + 1) / MAX_THREADS_PER_BLOCK) + 1); // +1 for the biase.
      nablas<<<dimGrid2, dimBlock2>>>(deviceTrainingSet, sample, numberOfLayers, totalNeurons, concurrentSamples,
                                      deviceNablaTheta, deviceDeltas, deviceActivations);

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
      std::cout << "Error in synchronizing for nablas " << err << std::endl;
      exit(1);
    }
  }
}

// Updates network's variables using gradient descent.
__global__
void updateThetasCU(int trainingSetSize, int numberOfLayers, int totalNeurons, float *weights, float *biases,
                    float *nablaTheta, float gamma, float lambda)
{
  int neuron = blockIdx.x * blockDim.x + threadIdx.x; // Which neuron am I working on?
  if(neuron < 0 || neuron >= totalNeurons)
    return;

  int sum = 0, layer = numberOfLayers - 1;
  for(int i = 0;i < numberOfLayers - 2;i++)
  {
    sum += deviceSizesOfLayers[i + 1];
    if(sum > neuron)
    {
      layer = i + 1;
      break;
    }
  }

  int theta = blockIdx.y * blockDim.y; // Which variable am I working on?
  if(theta < 0 || theta > deviceSizesOfLayers[layer - 1])
    return;

  int nablaThetaIndex = theta;
  int previousNeurons = 0;
  for(int i = 0;i < layer - 1;i++)
    previousNeurons += deviceSizesOfLayers[i + 1];
  for(int i = 0;i < neuron - previousNeurons;i++)
    nablaThetaIndex += deviceSizesOfLayers[layer - 1] + 1;
  for(int i = 0;i < layer - 1;i++)
    nablaThetaIndex += (deviceSizesOfLayers[i] + 1) * deviceSizesOfLayers[i + 1];

  // Updating the variable.
  if(theta == 0) // I am a bias.
    biases[neuron] -= gamma * nablaTheta[nablaThetaIndex] / trainingSetSize;
  else
  {
    int weightsIndex = theta - 1;
    for(int i = 0;i < neuron - previousNeurons;i++)
      weightsIndex += deviceSizesOfLayers[layer - 1];
    for(int i = 0;i < layer - 1;i++)
      weightsIndex += deviceSizesOfLayers[i] * deviceSizesOfLayers[i + 1];
    weights[weightsIndex] -= gamma * ((nablaTheta[nablaThetaIndex] +
                                       lambda * weights[weightsIndex]) / trainingSetSize);
  }
}

// updateThetasCU wrapper.
void updateThetas(int trainingSetSize, int *sizesOfLayers, int numberOfLayers, int totalNeurons,
                  float *deviceWeights, float *deviceBiases, float *deviceNablaTheta, float gamma, float lambda)
{
  cudaError_t err;
  dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1);
  int maxSizeOfLayer = sizesOfLayers[0];
  for(int i = 1;i < numberOfLayers - 1;i++)
    if(sizesOfLayers[i] > maxSizeOfLayer)
      maxSizeOfLayer = sizesOfLayers[i];

  // Each cuda thread works on a specific variable of a specific neuron.
  dim3 dimGrid((int)(totalNeurons / MAX_THREADS_PER_BLOCK) + 1, maxSizeOfLayer + 1);
  updateThetasCU<<<dimGrid, dimBlock>>>(trainingSetSize, numberOfLayers, totalNeurons, deviceWeights,
                                        deviceBiases, deviceNablaTheta, gamma, lambda);

  err = cudaDeviceSynchronize();
  if(err != cudaSuccess)
  {
    std::cout << "Error in synchronizing updateThetasCU " << err << std::endl;
    exit(1);
  }
}

// Sets up cuda's random-number-generator.
__global__
void setupKernelCU(curandState *states, int processLoad)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < 0 || id >= processLoad)
    return;

  curand_init(id, id, 0, &states[id]);
}

// Distorts training set to virtually increase the size of it.
__global__
void distortCU(curandState *states, float *trainingSet, int processLoad)
{
  int image = blockIdx.x * blockDim.x + threadIdx.x; // Which image am I working on?
  if(image < 0 || image >= processLoad)
    return;

  float destortionType, parameter;
  destortionType = (curand_normal(&states[image]) + 1) / 2; // In what way should I destort the image?

  float trfMtx[2][2], pi = acosf(-1);
  if(destortionType < 0.4) // Rotating.
  {
    parameter = (curand_normal(&states[image]) / 12) * pi; // Angle.
    trfMtx[0][0] = cosf(parameter);
    trfMtx[0][1] = sinf(parameter);
    trfMtx[1][0] = -sinf(parameter);
    trfMtx[1][1] = cosf(parameter);
  }
  else if(destortionType < 0.7)  // Scaling.
  {
    parameter = (curand_normal(&states[image]) * 15 / 100) + 1; // Volume for horizontal axis.
    trfMtx[0][0] = 1 / parameter;
    trfMtx[0][1] = 0;
    trfMtx[1][0] = 0;
    parameter = (curand_normal(&states[image]) * 15 / 100) + 1; // Volume for vertical axis.
    trfMtx[1][1] = 1 / parameter;
  }
  else  // Shearing.
  {
    parameter = (curand_normal(&states[image]) * 15 / 100);
    trfMtx[0][0] = 1;
    trfMtx[0][1] = parameter;
    trfMtx[1][0] = 0;
    trfMtx[1][1] = 1;
  }

  image *= deviceSizesOfLayers[0];

  int size = deviceSizesOfLayers[0];
  float buffer[784];
  for(int i = 0;i < size;i++)
    buffer[i] = trainingSet[image + i];

  int index, l = sqrtf(size), xa, xb, yd, yb;
  float threshold = 1.0e-6, l1, l2, x, y, ca, cb, cc, cd;
  // Apply the randomly choosen transformation matrix to the image.
  // For each pixel on the transformed image, calculate the corresponding point on the initial image.
  // The pixel takes the color of the corresponding point which is calculated using linear interpolation.
  for(int xPrime = -l / 2;xPrime < l / 2;xPrime++)
  {
    for(int yPrime = -l / 2;yPrime < l / 2;yPrime++)
    {
      x = trfMtx[0][0] * xPrime + trfMtx[0][1] * yPrime;
      y = trfMtx[1][0] * xPrime + trfMtx[1][1] * yPrime;

      index = image + (l / 2 - 1 - yPrime) * l + (xPrime + l / 2);

      if(x < -l / 2 || x > l / 2 - 1 || y < -l / 2 || y > l / 2 - 1) // Point out of image bounds.
        trainingSet[index] = 0;
      else if((fabsf(x - l / 2 + 1) < threshold && fabsf(y - l / 2 + 1) < threshold))
        trainingSet[index] = buffer[l - 1];
      else if(fabsf(x - l / 2 + 1) < threshold)
      {
        x = l / 2 - 1;
        yd = floorf(y), yb = yd + 1;
        cb = buffer[(l / 2 - 1 - yb) * l + (int)(x + l / 2)];
        cd = buffer[(l / 2 - 1 - yd) * l + (int)(x + l / 2)];

        l1 = yb - y;
        trainingSet[index] = l1 * cd + (1 - l1) * cb;
      }
      else if(fabsf(y - l / 2 + 1) < threshold)
      {
        y = l / 2 - 1;
        xa = floorf(x), xb = xa + 1;
        ca = buffer[xa + l / 2];
        cb = buffer[xb + l / 2];

        l2 = xb - x;
        trainingSet[index] = l2 * ca + (1 - l2) * cb;
      }
      else
      {
        xa = floorf(x), xb = xa + 1;
        yd = floorf(y), yb = yd + 1;

        ca = buffer[(l / 2 - 1 - yb) * l + (xa + l / 2)];
        cb = buffer[(l / 2 - 1 - yb) * l + (xb + l / 2)];
        cc = buffer[(l / 2 - 1 - yd) * l + (xa + l / 2)];
        cd = buffer[(l / 2 - 1 - yd) * l + (xb + l / 2)];

        l1 = yb - y;
        l2 = xb - x;
        trainingSet[index] = l2 * (l1 * cc + (1 - l1) * ca) + (1 - l2) * (l1 * cd + (1 - l1) * cb);
      }
    }
  }
}

// distortCU wrapper.
void distort(curandState *deviceStates, float *deviceTrainingSet, int processLoad, bool setupKernel)
{
  cudaError_t err;

  // Each cuda thread works on a specific image.
  dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1);
  dim3 dimGrid((int)(processLoad / MAX_THREADS_PER_BLOCK) + 1, 1);

  if(setupKernel) // Has the random number generator been initialized?
    setupKernelCU<<<dimGrid, dimBlock>>>(deviceStates, processLoad);
  distortCU<<<dimGrid, dimBlock>>>(deviceStates, deviceTrainingSet, processLoad);

  err = cudaDeviceSynchronize();
  if(err != cudaSuccess)
  {
    std::cout << "Error in synchronizing distortCU " << err << std::endl;
    exit(1);
  }
}
