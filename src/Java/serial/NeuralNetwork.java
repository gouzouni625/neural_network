import java.lang.Math;

public class NeuralNetwork{

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

  public void train(double[][] trainingSet, double[][] labels,
                    int trainingSetSize, int numberOfIterations,
                    double gamma){
    int sampleSize = sizesOfLayers_[0];
    int labelSize = sizesOfLayers_[numberOfLayers_ - 1];

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

    // Training the network.
    for(int iteration = 0;iteration < numberOfIterations;iteration++){

      // Calculating the derivative of the const function for each sample.
      for(int sample = 0;sample < trainingSetSize;sample++){
        this.backPropagation(trainingSet[sample], labels[sample], nablaTheta);
      }

      // Updating network's parameters.
      for(int i = 0;i < numberOfLayers_ - 1;i++){

        for(int j = 0;j < sizesOfLayers_[i + 1];j++){

          biases_[i][j] -= gamma * nablaTheta[i][j][0] / trainingSetSize;
          for(int k = 0;k < sizesOfLayers_[i];k++){
            weights_[i][j][k] -= gamma * nablaTheta[i][j][k + 1] / trainingSetSize;
          }
        }
      }

      for(int i = 0;i < numberOfLayers_ - 1;i++){
        for(int j = 0;j < sizesOfLayers_[i + 1];j++){
          for(int k = 0;k < sizesOfLayers_[i] + 1;k++){
            nablaTheta[i][j][k] = 0;
          }
        }
      }

    }
  }

  public int getNumberOfLayers(){
    return numberOfLayers_;
  }

  public void saveNetwork(){

  }

  public void loadNetwork(){

  }

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

    // Calculating the derivative of the const function with respect to every parameter.
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

  private double sigmoid(double z){
    return (1 / (1 + Math.exp(-z)));
  }

  private int numberOfLayers_;
  private int[] sizesOfLayers_;

  private double[][][] weights_;
  private double[][] biases_;

};
