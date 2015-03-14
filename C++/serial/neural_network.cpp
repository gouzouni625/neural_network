#include "neural_network.h"

namespace neural_network{

NeuralNetwork::NeuralNetwork(const std::vector<int>& sizesOfLayers): sizesOfLayers_(sizesOfLayers), numberOfLayers_(sizesOfLayers.size()){
  srand(time(NULL));

  // Randomly initializing network's parameters.
  weights_ = new double**[numberOfLayers_ - 1];
  biases_ = new double*[numberOfLayers_ - 1];
  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    weights_[i] = new double*[sizesOfLayers_[i + 1]];
    biases_[i] = new double[sizesOfLayers_[i + 1]];

    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
    {
      weights_[i][j] = new double[sizesOfLayers_[i]];

      for(int k = 0;k < sizesOfLayers_[i];k++)
        weights_[i][j][k] = ((double) rand() / (RAND_MAX)) * 0.5 - 0.25;
      biases_[i][j] = ((double) rand() / (RAND_MAX)) * 0.5 - 0.25;
    }
  }
}

void NeuralNetwork::feedForward(double* input, double* &output) const{
  double* buffer = input;
  output = new double[sizesOfLayers_[1]];
  for(int i = 0;i < numberOfLayers_ - 1;i++) /* for each layer. */
  {
    for(int j = 0;j < sizesOfLayers_[i + 1];j++) /* for each neuron on current layer. */
    {
      double sum = 0;
      for(int k = 0;k < sizesOfLayers_[i];k++)
        sum += weights_[i][j][k] * buffer[k];

      output[j] = this->sigmoid(sum + biases_[i][j]);
    }

    if(i + 2 < numberOfLayers_)
    {
      buffer = output;
      output = new double[sizesOfLayers_[i + 2]];
    }
  }
}

void  NeuralNetwork::train(double** trainingSet, double** labels, int trainingSetSize, int numberOfIterations, double gamma){
  int sampleSize = sizesOfLayers_[0];
  int labelSize = sizesOfLayers_[numberOfLayers_ - 1];

  // Allocating memory for the derivative of the cost function with respect to every parameter.
  double*** nablaTheta = new double**[numberOfLayers_ - 1];
  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    nablaTheta[i] = new double*[sizesOfLayers_[i + 1]];
    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
    {
      nablaTheta[i][j] = new double[sizesOfLayers_[i] + 1];

      for(int k = 0;k < sizesOfLayers_[i] + 1;k++)
        nablaTheta[i][j][k] = 0;
    }
  }

  // Training the network.
  for(int iteration = 0;iteration < numberOfIterations;iteration++)
  {
    // Calculating the derivate of the cost function for each sample.
    for(int sample = 0;sample < trainingSetSize;sample++) 
      this->backPropagation(trainingSet[sample], labels[sample], nablaTheta);

    // Updating network's parameters.
    for(int i = 0;i < numberOfLayers_ - 1;i++)
    {
      for(int j = 0;j < sizesOfLayers_[i + 1];j++)
      {
        biases_[i][j] -= gamma * nablaTheta[i][j][0] / trainingSetSize;
        for(int k = 0;k < sizesOfLayers_[i];k++)
          weights_[i][j][k] -= gamma * nablaTheta[i][j][k + 1] / trainingSetSize;
      }
    }

    for(int i = 0;i < numberOfLayers_ - 1;i++)
      for(int j = 0;j < sizesOfLayers_[i + 1];j++)
        for(int k = 0;k < sizesOfLayers_[i] + 1;k++)
          nablaTheta[i][j][k] = 0;
  }

  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
      delete[] nablaTheta[i][j];
    delete[] nablaTheta[i];
  }
  delete[] nablaTheta;
}

int NeuralNetwork::getNumberOfLayers() const{
  return numberOfLayers_;
}

void NeuralNetwork::set(double *weights, double *biases)
{
  int bIndex = 0, wIndex = 0;
  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
    {
      biases_[i][j] = biases[bIndex];
      bIndex++;
      for(int k = 0;k < sizesOfLayers_[i];k++)
      {
        weights_[i][j][k] = weights[wIndex];
        wIndex++;
      }
    }
  }
}

// Saves network's variables.
void NeuralNetwork::saveNetwork() const{
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
  
  for(int i = 0;i < numberOfLayers_ - 1;i++){
    for(int j = 0;j < sizesOfLayers_[i + 1];j++){
      for(int k = 0;k < sizesOfLayers_[i];k++){
        weightsFile << weights_[i][j][k];
      }
      biasesFile << biases_[i][j];
    }
  }

  weightsFile.close();
  biasesFile.close();
}

// Loads network's variables.
void NeuralNetwork::loadNetwork(){
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

  for(int i = 0;i < numberOfLayers_ - 1;i++){
    for(int j = 0;j < sizesOfLayers_[i + 1];j++){
      for(int k = 0;k < sizesOfLayers_[i];k++){
        weightsFile >> weights_[i][j][k];
      }
      biasesFile >> biases_[i][j];
    }
  }

  weightsFile.close();
  biasesFile.close();
}

NeuralNetwork::~NeuralNetwork(){
  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
      delete[] weights_[i][j];

    delete[] weights_[i];
    delete[] biases_[i];
  }
  delete[] weights_;
  delete[] biases_;
}

void NeuralNetwork::backPropagation(double *sample, double* label, double*** &nablaTheta) const{ // nablaTheta is already
                                                                                          // initialized.
  // Calculating the output of each neuron.
  double* buffer = sample;
  double** activation = new double*[numberOfLayers_ - 1];
  activation[0] = new double[sizesOfLayers_[1]];
  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
    {
      double sum = 0;
      for(int k = 0;k < sizesOfLayers_[i];k++)
        sum += weights_[i][j][k] * buffer[k];

      activation[i][j] = this->sigmoid(sum + biases_[i][j]);
    }

    if(i + 2 < numberOfLayers_)
    {
      buffer = activation[i];
      activation[i + 1] = new double[sizesOfLayers_[i + 2]];
    }
  }

  // Calculating the error of each neuron.
  double** delta = new double*[numberOfLayers_ - 1];
  delta[numberOfLayers_ - 2] = new double[sizesOfLayers_[numberOfLayers_ - 1]];
  for(int j = 0;j < sizesOfLayers_[numberOfLayers_ - 1];j++)
    delta[numberOfLayers_ - 2][j] = activation[numberOfLayers_ - 2][j] - label[j];
  for(int i = numberOfLayers_ - 3;i >= 0;i--) // for each layer.
  {
    delta[i] = new double[sizesOfLayers_[i + 1]];

    for(int j = 0;j < sizesOfLayers_[i + 1];j++) // for each node of the current layer.
    {
      delta[i][j] = 0;
      for(int k = 0;k < sizesOfLayers_[i + 2];k++)
        delta[i][j] += weights_[i + 1][k][j] * delta[i + 1][k];

      delta[i][j] *= activation[i][j] * (1 - activation[i][j]);
    }
  }

  // Calculating the derivative of the cost function with respect to every parameter.
  for(int j = 0;j < sizesOfLayers_[1];j++) // activation[][] doesn't include the sample.
  {
    nablaTheta[0][j][0] += delta[0][j]; // biases.
    for(int k = 1;k < sizesOfLayers_[0] + 1;k++)
      nablaTheta[0][j][k] += sample[k - 1] * delta[0][j];
  }
  for(int i = 1;i < numberOfLayers_ - 1;i++)
  {
    for(int j = 0;j < sizesOfLayers_[i + 1];j++)
    {
      nablaTheta[i][j][0] += delta[i][j]; // biases.
      for(int k = 1;k < sizesOfLayers_[i] + 1;k++)
        nablaTheta[i][j][k] += activation[i - 1][k - 1] * delta[i][j];
    }
  }

  for(int i = 0;i < numberOfLayers_ - 1;i++)
  {
    delete[] activation[i];
    delete[] delta[i];
  }
  delete[] activation;
  delete[] delta;
}

double NeuralNetwork::sigmoid(double z) const{
  return (1 / (1 + exp(-z)));
}

}
