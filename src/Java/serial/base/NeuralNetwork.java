package base;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/** \class NeuralNetwork.
 *  \brief A class implementing a neural network. Includes the network's
 *         parameters, training and evaluating methods.
 */
public class NeuralNetwork{

  /** \brief The constructor of the class. Randomly initializes the
   *         network's parameters with values in [-0.25, 0.25).
   * 
   *  \param sizesOfLayers The number of neurons in each layer including
   *         the input and output layers. The number of layers is equal
   *         to sizesOfLayers.length.
   */
  public NeuralNetwork(int[] sizesOfLayers){
    sizesOfLayers_ = sizesOfLayers;
    numberOfLayers_ = sizesOfLayers_.length;

    weights_ = new double[numberOfLayers_ - 1][][];
    biases_ = new double[numberOfLayers_ - 1][];

    for(int i = 0;i < numberOfLayers_ - 1;i++){
      weights_[i] = new double[sizesOfLayers_[i + 1]][];
      biases_[i] = new double[sizesOfLayers_[i + 1]];

      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        weights_[i][j] = new double[sizesOfLayers_[i]];

        for(int k = 0;k < sizesOfLayers_[i];k++){
          weights_[i][j][k] = ((double)Math.random()) * 0.5 - 0.25;
        }
        biases_[i][j] = ((double)Math.random()) * 0.5 - 0.25;
      }
    }
  }

  /** \brief Applies an input to the neural network and returns its output.
   * 
   *  \param input The input to the neural network. The length of the input
   *         must be equal to sizesOfLayers_[0].
   *  \return The output of the layer for the given input. The length of the
   *          output is equal to sizesOfLayers_[sizesOfLayers_.length - 1].
   */
  public double[] feedForward(double[] input){
    double[] inputBuffer = input.clone();

    double[] output = new double[sizesOfLayers_[1]];
    for(int i = 0;i < numberOfLayers_ - 1;i++){
      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        double sum = 0;
        for(int k = 0;k < sizesOfLayers_[i];k++){
          sum += weights_[i][j][k] * inputBuffer[k];
        }

        output[j] = this.sigmoid(sum + biases_[i][j]);
      }

      if(i + 2 < numberOfLayers_){
        inputBuffer = output.clone();
        output = new double[sizesOfLayers_[i + 2]];
      }
    }
    return output;
  }

  /** \brief Trains the neural network on a given set of data.
   * 
   *  \param trainingSet The set of data on which the neural network will
   *         be trained. The length of each sample of the data set must be
   *         equal to sizesOfLayers_[0].
   *  \param labels The labels for the training set. The length of each
   *         sample of the labels set must be equal to
   *         sizesOfLayers_[sizesOfLayers_.length - 1].
   *  \param trainingSetSize The number of training samples in the training set.
   *  \param numberOfIterations The number of iterations of the training method. 
   *  \param gamma Parameter gamma of the training method(gradient descent).
   */
  public void train(double[][] trainingSet, double[][] labels,
                    int trainingSetSize, int numberOfIterations,
                    double gamma){
    
    double[][][] nablaTheta = new double[numberOfLayers_ - 1][][];
    for(int i = 0;i < numberOfLayers_ - 1;i++){
      
      nablaTheta[i] = new double[sizesOfLayers_[i + 1]][];
      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        nablaTheta[i][j] = new double[sizesOfLayers_[i] + 1];

        for(int k = 0;k < sizesOfLayers_[i] + 1;k++){
          nablaTheta[i][j][k] = 0;
        }
      }
    }

    // Training loop.
    for(int iteration = 0;iteration < numberOfIterations;iteration++){

      // Calculating the derivative of the cost function for each sample
      // using the back propagation algorithm.
      for(int sample = 0;sample < trainingSetSize;sample++){
        this.backPropagation(trainingSet[sample], labels[sample], nablaTheta);
      }

      // Updating network's parameters using the gradient descent algorithm.
      for(int i = 0;i < numberOfLayers_ - 1;i++){
        for(int j = 0;j < sizesOfLayers_[i + 1];j++){
          biases_[i][j] -= gamma * nablaTheta[i][j][0] / trainingSetSize;
          for(int k = 0;k < sizesOfLayers_[i];k++){
            weights_[i][j][k] -= gamma * nablaTheta[i][j][k + 1] /
                                 trainingSetSize;
          }
        }
      }

      // Set nablaTheta to zero.
      for(int i = 0;i < numberOfLayers_ - 1;i++){
        for(int j = 0;j < sizesOfLayers_[i + 1];j++){
          for(int k = 0;k < sizesOfLayers_[i] + 1;k++){
            nablaTheta[i][j][k] = 0;
          }
        }
      }

    }
  }

  /** \brief getter function for the number of this neural network's layers.
   * 
   *  \return The number of this neural network's layers.
   */
  public int getNumberOfLayers(){
    return numberOfLayers_;
  }

  /** \brief Saves the parameters of the neural network to a binary file.
   * 
   *  \param path The absolute, or relative path of the file where the
   *         parameters will be saved.
   *         
   *  \throws FileNotFoundException
   *  \throws IOException
   */
  public void saveNetwork(String path) throws FileNotFoundException,
                                              IOException{
    
    FileOutputStream fileOutputStream = new FileOutputStream(path);
    DataOutputStream dataOutputStream = new DataOutputStream(fileOutputStream);
  
    for(int i = 0;i < numberOfLayers_ - 1;i++){
      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        dataOutputStream.writeDouble(biases_[i][j]);
          
        for(int k = 0;k < sizesOfLayers_[i];k++){
          dataOutputStream.writeDouble(weights_[i][j][k]);
        }
      }
    }
    dataOutputStream.close();        
  }

  /** \brief Load the parameters of a neural network to this neural network.
   * 
   *  \param path The path of the file where the parameters are saved. 
   *  
   *  \throws FileNotFoundException
   *  \throws IOException
   */
  public void loadNetwork(String path) throws FileNotFoundException,
                                              IOException{
    
    FileInputStream fileInputStream = new FileInputStream(path);
    DataInputStream dataInputStream = new DataInputStream(fileInputStream);
      
    for(int i = 0;i < numberOfLayers_ - 1;i++){
      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        biases_[i][j] = dataInputStream.readDouble();
          
        for(int k = 0;k < sizesOfLayers_[i];k++){
          weights_[i][j][k] = dataInputStream.readDouble();
        }
      }
    }
    dataInputStream.close();
  }

  /** \brief Implement the back propagation algorithm to calculate the
   *         derivative of the cost function with respect to each parameter.
   * 
   *  \param sample The input to the neural network.
   *  \param label The label of the input.
   *  \param nablaTheta The array in which the derivative of the cost function
   *         will be saved.
   */
  private void backPropagation(double[] sample, double[] label,
                               double[][][] nablaTheta){
    
    // Calculating the output of each neuron.
    double[] sampleBuffer = sample.clone();
    double[][] activations = new double[numberOfLayers_ - 1][];
    activations[0] = new double[sizesOfLayers_[1]];
    for(int i = 0;i < numberOfLayers_ - 1;i++){

      for(int j = 0;j < sizesOfLayers_[i + 1];j++){

        double sum = 0;
        for(int k = 0;k < sizesOfLayers_[i];k++){
          sum += weights_[i][j][k] * sampleBuffer[k];
        }

        activations[i][j] = this.sigmoid(sum + biases_[i][j]);
      }

      if(i + 2 < numberOfLayers_){
        sampleBuffer = activations[i];
        activations[i + 1] = new double[sizesOfLayers_[i + 2]];
      }
    }

    // Calculating the error of each neuron.
    double[][] delta = new double[numberOfLayers_ - 1][];
    delta[numberOfLayers_ - 2] = new double[sizesOfLayers_[numberOfLayers_ - 1]];
    for(int j = 0;j < sizesOfLayers_[numberOfLayers_ - 1];j++){
      delta[numberOfLayers_ - 2][j] = activations[numberOfLayers_ - 2][j] - label[j];
    }
    for(int i = numberOfLayers_ - 3;i >= 0;i--){

      delta[i] = new double[sizesOfLayers_[i + 1]];

      for(int j = 0;j < sizesOfLayers_[i + 1];j++){

        delta[i][j] = 0;
        for(int k = 0;k < sizesOfLayers_[i + 2];k++){
          delta[i][j] += weights_[i + 1][k][j] * delta[i + 1][k];
        }

        delta[i][j] *= activations[i][j] * (1 - activations[i][j]);
      }
    }

    // Calculating the derivative of the cost function with respect to every
    // parameter.
    for(int j = 0;j < sizesOfLayers_[1];j++){
      nablaTheta[0][j][0] += delta[0][j]; // biases.
      for(int k = 1;k < sizesOfLayers_[0] + 1;k++){
        nablaTheta[0][j][k] += sample[k - 1] * delta[0][j];
      }
    }
    for(int i = 1;i < numberOfLayers_ - 1;i++){
      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        nablaTheta[i][j][0] += delta[i][j]; // biases.
        for(int k = 1;k < sizesOfLayers_[i] + 1;k++){
          nablaTheta[i][j][k] += activations[i - 1][k - 1] * delta[i][j];
        }
      }
    }

  }

  /** \brief Activation function for the neurons.
   * 
   *  \param z The independent variable.
   *  \return The sigmoid of z.
   */
  private double sigmoid(double z){
    return (1 / (1 + Math.exp(-z)));
  }

  private int numberOfLayers_; //!< The number of layers of the neural network.
  private int[] sizesOfLayers_; //!< The number of neurons in each layer.

  private double[][][] weights_; //!< dimensions are: layer, neuron, weight.
  private double[][] biases_; //!< dimensions are: layer, neuron.
};
