#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <fstream>
#include <vector>

#include <cstdlib>
#include <ctime>
#include <cmath>

namespace neural_network{

class NeuralNetwork{
public:
  // Constructor. Allocates memory for network's parameters and randomly
  // initializes them.
  NeuralNetwork(const std::vector<int>& sizesOfLayers);

  // Given an input, calculates the output of the neural network.
  void feedForward(double* input, double* &output) const; /* input must be the size of the input layer.
                                                       The returned vector has size equal to output
                                                       layer. */

  // Given a trainingSet and training labels, trains the neural network.
  void train(double** trainingSet, double** labels, int trainingSetSize, int numberOfIterations, double gamma);

  int getNumberOfLayers() const;

  // Sets the parameters of the network according to input.
  void set(double *weights, double *biases);

  // Saves the parameters of the networks to a given file.
  void saveNetwork() const;

  // Loads the parameters of the network from a given file.
  void loadNetwork();

  // Destructor. Cleans up memory.
  ~NeuralNetwork();

private:
  int numberOfLayers_; /* including input and output layer. */
  std::vector<int> sizesOfLayers_; /* including input and output layer. */
  double*** weights_; /* weights[i][j][k] means k-th weight of the j-th neuron on the i-th layer. */
  double** biases_; /* biases[i][j] means the bias of the j-th neuron on the i-th layer. */

  // Backpropagation algorithm is used to train the neural network.
  void backPropagation(double *sample, double* label, double*** &nablaTheta) const;

  // Activation function of each neuron in the neural network.
  double sigmoid(double z) const;
};

}

#endif // NEURAL_NETWORK_H
