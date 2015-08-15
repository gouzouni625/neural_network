package main.base;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/** @class NeuralNetwork
 *
 *  @brief Implements a feed forward NeuralNetwork.
 *
 *  The implementation includes the NeuralNetwork parameters, training and evaluating methods.
 */
public class NeuralNetwork{

  public NeuralNetwork(){
    momentumCoefficient_ = 0;
  }

  /**
   *  @brief Constructor
   *
   *  Randomly initializes the NeuralNetwork parameters with values in [-0.25, 0.25).
   *
   *  @param sizesOfLayers The number of neurons in each layer including the input and output layers. The number of layers
   *         is equal to sizesOfLayers.length.
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

    momentumCoefficient_ = 0;
  }

  /**
   *  @brief Applies an input to this NeuralNetwork and returns its output.
   *
   *  @param input The input to this NeuralNetwork. The length of the input must be equal to sizesOfLayers_[0].
   *
   *  @return Returns the output of this NeuralNetwork for the given input. The length of the output is equal to
   *          sizesOfLayers_[sizesOfLayers_.length - 1].
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

        output[j] = this.activationFunction(sum + biases_[i][j]);
      }

      if(i + 2 < numberOfLayers_){
        inputBuffer = output.clone();
        output = new double[sizesOfLayers_[i + 2]];
      }
    }

    return output;
  }

  /**
   *  @brief Trains this NeuralNetwork on a given set of data.
   *
   *  @param trainingSet The set of data on which this NeuralNetwork will be trained. The length of each sample
   *         of the data set must be equal to sizesOfLayers_[0].
   *  @param labels The labels for the training set. The length of each sample of the labels set must be equal to
   *         sizesOfLayers_[sizesOfLayers_.length - 1].
   *  @param trainingSetSize The number of training samples in the training set.
   *  @param numberOfIterations The number of iterations of the training method.
   *  @param gamma Parameter gamma of the training method(gradient descent).
   */
  public void train(double[][] trainingSet, double[][] labels, int trainingSetSize, int numberOfIterations, double gamma){
    double[][][] nablaTheta = new double[numberOfLayers_ - 1][][];
    double[][][] momentum = new double[numberOfLayers_ - 1][][];
    for(int i = 0;i < numberOfLayers_ - 1;i++){

      nablaTheta[i] = new double[sizesOfLayers_[i + 1]][];
      momentum[i] = new double[sizesOfLayers_[i + 1]][];;
      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        nablaTheta[i][j] = new double[sizesOfLayers_[i] + 1];
        momentum[i][j] = new double[sizesOfLayers_[i] + 1];

        for(int k = 0;k < sizesOfLayers_[i] + 1;k++){
          nablaTheta[i][j][k] = 0;
          momentum[i][j][k] = 0;
        }
      }
    }

    // Training loop.
    for(int iteration = 0;iteration < numberOfIterations;iteration++){

      // Calculating the derivative of the cost function for each sample using the back propagation algorithm.
      for(int sample = 0;sample < trainingSetSize;sample++){
        this.backPropagation(trainingSet[sample], labels[sample], nablaTheta);
      }

      // Updating network's parameters using the gradient descent algorithm.
      for(int i = 0;i < numberOfLayers_ - 1;i++){
        for(int j = 0;j < sizesOfLayers_[i + 1];j++){
          momentum[i][j][0] = (nablaTheta[i][j][0] * (1 - momentumCoefficient_) +
                                momentum[i][j][0] * momentumCoefficient_)  / trainingSetSize;
          biases_[i][j] -= gamma * momentum[i][j][0];
          for(int k = 0;k < sizesOfLayers_[i];k++){
            momentum[i][j][k + 1] = (nablaTheta[i][j][k + 1] * (1 - momentumCoefficient_) +
                                      momentum[i][j][k + 1] * momentumCoefficient_)  / trainingSetSize;
            weights_[i][j][k] -= gamma * momentum[i][j][k + 1];
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

  /**
   *  @brief Getter function for the number of this NeuralNetwork layers.
   *
   *  \return The number of this NeuralNetwork layers.
   */
  public int getNumberOfLayers(){
    return numberOfLayers_;
  }

  /**
   *  @brief Saves the parameters of this NeuralNetwork to a binary file.
   *
   *  @param path The absolute, or relative path of the file where the parameters will be saved.
   *
   *  @throws IOException When an exception occurs while writing on the file.
   */
  public void saveToBinary(String path) throws IOException{
    FileOutputStream fileOutputStream = new FileOutputStream(path);
    DataOutputStream dataOutputStream = new DataOutputStream(fileOutputStream);

    dataOutputStream.writeInt(numberOfLayers_);
    for(int i = 0;i < numberOfLayers_;i++){
      dataOutputStream.writeInt(sizesOfLayers_[i]);
    }

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

  /**
   *  @brief Saves the parameters of this NeuralNetwork to an xml file.
   *
   *  @param path The absolute or relative path of the file where the parameters will be saved.
   *
   *  @throws IOException When an exception occurs while writing on the file.
   */
  public void saveToXML(String path) throws IOException{
    PrintWriter printWriter = new PrintWriter(path);

    printWriter.print("<sizes_of_layers>");
    for(int i = 0;i < numberOfLayers_ - 1;i++){
      printWriter.print(sizesOfLayers_[i] + " ");
    }
    printWriter.println(sizesOfLayers_[numberOfLayers_ - 1] + "</sizes_of_layers>");

    for(int i = 0;i < numberOfLayers_ - 1;i++){
      printWriter.println("<layer>");

      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        printWriter.print("  <neuron>\n    <bias>" + biases_[i][j] + "</bias>\n    <weights>");

        for(int k = 0;k < sizesOfLayers_[i] - 1;k++){
          printWriter.print(weights_[i][j][k] + " ");
        }
        printWriter.println(weights_[i][j][sizesOfLayers_[i] - 1] + "</weights>\n  </neuron>");
      }

      printWriter.println("</layer>");
    }

    printWriter.close();
  }

  /**
   *  @brief Loads the parameters for this NeuralNetwork from a binary file.
   *
   *  @param path The path of the file where the parameters are saved.
   *
   *  @throws IOException When an exception occurs while reading from the file.
   */
  public void loadFromBinary(String path) throws IOException{
    FileInputStream fileInputStream = new FileInputStream(path);
    DataInputStream dataInputStream = new DataInputStream(fileInputStream);

    numberOfLayers_ = dataInputStream.readInt();
    sizesOfLayers_ = new int[numberOfLayers_];
    for(int i = 0;i < numberOfLayers_;i++){
      sizesOfLayers_[i] = dataInputStream.readInt();
    }

    // Allocate the needed memory for the weights and the biases.
    weights_ = new double[numberOfLayers_ - 1][][];
    biases_ = new double[numberOfLayers_ - 1][];

    for(int i = 0;i < numberOfLayers_ - 1;i++){
      weights_[i] = new double[sizesOfLayers_[i + 1]][];
      biases_[i] = new double[sizesOfLayers_[i + 1]];

      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        weights_[i][j] = new double[sizesOfLayers_[i]];
      }
    }

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

  /**
   *  @brief Loads the parameters for this NeuralNetwork from an xml file.
   *
   *  @param path The path of the file where the parameters are saved.
   *
   *  @throws IOException When an exception occurs while reading from the file.
   */
  public void loadFromXML(String path) throws IOException{
    // Load all the data from the given file.
    String xmlData = new String(Files.readAllBytes(Paths.get(path)));

    // Parse the sizes of the layers.
    int startOfSizesOfLayers = xmlData.indexOf("<sizes_of_layers>");
    int endOfSizesOfLayers = xmlData.indexOf("</sizes_of_layers>");

    String[] sizesOfLayers = xmlData.substring(startOfSizesOfLayers + ("<sizes_of_layers>").length(), endOfSizesOfLayers)
                                    .split(" ");

    numberOfLayers_ = sizesOfLayers.length;
    sizesOfLayers_ = new int[numberOfLayers_];
    for(int i = 0;i < numberOfLayers_;i++){
      sizesOfLayers_[i] = Integer.parseInt(sizesOfLayers[i]);
    }

    // Allocate the needed memory for the weights and the biases.
    weights_ = new double[numberOfLayers_ - 1][][];
    biases_ = new double[numberOfLayers_ - 1][];

    for(int i = 0;i < numberOfLayers_ - 1;i++){
      weights_[i] = new double[sizesOfLayers_[i + 1]][];
      biases_[i] = new double[sizesOfLayers_[i + 1]];

      for(int j = 0;j < sizesOfLayers_[i + 1];j++){
        weights_[i][j] = new double[sizesOfLayers_[i]];
      }
    }

    xmlData = xmlData.substring(endOfSizesOfLayers + ("</sizes_of_layers>").length());

    // Parse each layer.
    int startOfLayer = xmlData.indexOf("<layer>");
    int currentLayer = 0;
    while(startOfLayer != -1){

      // Parse each neuron in the current layer.
      int currentNeuron = 0;
      while(currentNeuron < sizesOfLayers_[currentLayer + 1]){

        int startOfBias = xmlData.indexOf("<bias>");
        int endOfBias = xmlData.indexOf("</bias>");

        biases_[currentLayer][currentNeuron] = Double.parseDouble(xmlData.substring(startOfBias + ("<bias>").length(),
                                                                                    endOfBias));

        int startOfWeights = xmlData.indexOf("<weights>");
        int endOfWeights = xmlData.indexOf("</weights>");

        String[] weights = xmlData.substring(startOfWeights + ("<weights>").length(), endOfWeights).split(" ");
        for(int i = 0;i < weights.length;i++){
          weights_[currentLayer][currentNeuron][i] = Double.parseDouble(weights[i]);
        }

        xmlData = xmlData.substring(xmlData.indexOf("</neuron>") + ("</neuron>").length());
        currentNeuron++;
      }

      xmlData = xmlData.substring(xmlData.indexOf("</layer>") + ("</layer>").length());
      startOfLayer = xmlData.indexOf("<layer>");
      currentLayer++;
    }
  }

  /**
   *  @brief Creates a NeuralNetwork and loads the parameters from the given binary file.
   *
   *  @param path The path of the binary file that holds the parameters of the NeuralNetwork to be created.
   *
   *  @return Returns the created NeuralNetwork.
   *
   *  @throws IOException When an exception occurs while reading from the file.
   */
  public static NeuralNetwork createFromBinary(String path) throws IOException{
    NeuralNetwork neuralNetwork = new NeuralNetwork();

    neuralNetwork.loadFromBinary(path);

    return neuralNetwork;
  }

  /**
   *  @brief Creates a NeuralNetwork and loads the parameters from the given XML file.
   *
   *  @param path The path of the XML file that holds the parameters of the NeuralNetwork to be created.
   *
   *  @return Returns the created NeuralNetwork.
   *
   *  @throws IOException When an exception occurs while reading from the file.
   */
  public static NeuralNetwork createFromXML(String path) throws IOException{
    NeuralNetwork neuralNetwork = new NeuralNetwork();

    neuralNetwork.loadFromXML(path);

    return neuralNetwork;
  }

  /**
   *  @brief Implements the back propagation algorithm to calculate the derivative of the cost function with respect
   *         to each parameter.
   *
   *  @param sample The input to this NeuralNetwork.
   *  @param label The label of the input.
   *  @param nablaTheta The array in which the derivative of the cost function will be saved.
   */
  private void backPropagation(double[] sample, double[] label, double[][][] nablaTheta){
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

        activations[i][j] = this.activationFunction(sum + biases_[i][j]);
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
      delta[numberOfLayers_ - 2][j] = costFunction(activations[numberOfLayers_ - 2][j], label[j]);
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

    // Calculating the derivative of the cost function with respect to every parameter.
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

  /**
   *  @brief Implements the cost function of this NeuralNetwork.
   *
   *  @param activation The output of last layer of this NeuralNetwork.
   *  @param label The expected output.
   *
   *  @return Returns the value of the cost function for the given activation and label.
   */
  private double costFunction(double activation, double label){
    return (activation - label);
  }

  /**
   *  @brief The activation function for each neuron.
   *
   *  @param z The independent variable.
   *
   *  @return The value of the activation function at z.
   */
  private double activationFunction(double z){
    return (sigmoid(z));
  }

  /**
   *  @brief Implements the sigmoid function.
   *
   *  This function is used as an activation function for the neurons.
   *
   *  @param z The independent variab.e
   *
   *  @return Returns the value of the sigmoid function on z.
   */
  private double sigmoid(double z){
    return (1 / (1 + Math.exp(-z)));
  }

  /**
   *  @brief Getter method for the sizes of layers of this independent.
   *
   *  @return Returns the sizes of the layers of this independent.
   */
  public int[] getSizesOfLayers(){
    return sizesOfLayers_;
  }

  /**
   *  @brief Setter method for the momentum coefficient.
   *
   *  @param momentumCoefficient The new value for the momentum coefficient.
   */
  public void setMomentumCoefficient(double momentumCoefficient){
    momentumCoefficient_ = momentumCoefficient;
  }

  /**
   *  @brief Getter method for the momentum coefficient.
   *
   *  @return Returns the current value of the momentum coefficient.
   */
  public double getMomentumCoefficient(){
    return momentumCoefficient_;
  }

  private int numberOfLayers_; //!< The number of layers of this NeuralNetwork.
  private int[] sizesOfLayers_; //!< The number of neurons in each layer.

  private double[][][] weights_; //!< The weight parameters of this NeuralNetwork. dimensions are: layer, neuron, weight.
  private double[][] biases_; //!< The bias parameters of this NeuralNetwork. dimensions are: layer, neuron.

  private double momentumCoefficient_; //!< The momentum coefficient of this NeuralNetwork.

}
