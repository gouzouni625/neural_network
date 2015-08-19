package main.java.trainers;

import java.io.IOException;

import main.java.base.NeuralNetwork;
import main.java.distorters.Distorter;
import main.java.utilities.data.DataSet;

/** @class Trainer
 *
 *  @brief Implements a Trainer for main.base.NeuralNetwork.
 */
public abstract class Trainer{
  /**
   *  @brief Constructor.
   *
   *  @param sizesOfLayers The size of the layers of the main.java.base.NeuralNetwork.
   */
  public Trainer(int[] sizesOfLayers){
    sizesOfLayers_ = sizesOfLayers;
    neuralNetwork_ = new NeuralNetwork(sizesOfLayers);

    trainingSetPath_ = "";
    trainingLabelsPath_ = "";
    testingSetPath_ = "";
    testingLabelsPath_ = "";

    distorter_ = null;
  }

  /**
   *  @brief Constructor
   *
   *  @param sizesOfLayers The size of the layers of the main.java.base.NeuralNetwork.
   *  @param trainingSetPath The full path of the training set data.
   *  @param trainingLabelsPath The full path of the training labels.
   *  @param testingSetPath The full path of the testing set data.
   *  @param testingLabelsPath The full path of the testing labels.
   */
  public Trainer(int[] sizesOfLayers, String trainingSetPath, String trainingLabelsPath, String testingSetPath,
                                      String testingLabelsPath){
    sizesOfLayers_ = sizesOfLayers;
    neuralNetwork_ = new NeuralNetwork(sizesOfLayers);

    trainingSetPath_ = trainingSetPath;
    trainingLabelsPath_ = trainingLabelsPath;
    testingSetPath_ = testingSetPath;
    testingLabelsPath_ = testingLabelsPath;

    distorter_ = null;
  }

  /**
   *  @brief Constructor.
   *
   *  @param sizesOfLayers The size of the layers of the main.base.NeuralNetwork.
   *  @param trainingSetPath The full path of the training set data.
   *  @param trainingLabelsPath The full path of the training labels.
   *  @param testingSetPath The full path of the testing set data.
   *  @param testingLabelsPath The full path of the testing labels.
   *  @param distorter The main.java.distorters.Distorter to be used to distort the training set data, while training.
   */
  public Trainer(int[] sizesOfLayers, String trainingSetPath, String trainingLabelsPath, String testingSetPath,
                                      String testingLabelsPath, Distorter distorter){
    sizesOfLayers_ = sizesOfLayers;
    neuralNetwork_ = new NeuralNetwork(sizesOfLayers);

    trainingSetPath_ = trainingSetPath;
    trainingLabelsPath_ = trainingLabelsPath;
    testingSetPath_ = testingSetPath;
    testingLabelsPath_ = testingLabelsPath;

    distorter_ = distorter;
  }

  /**
   *  @brief Loads a training and a testing set.
   *
   *  @throws Exception The exception is thrown to allow classes extending this class, to load data from files using
   *                    file input streams.
   */
  abstract public void load(DataSet trainingSet, DataSet testingSet) throws Exception;

  /**
   *  @brief Trains a main.java.base.NeuralNetwork on the given data.
   *
   *  @throws Exception The exception is thrown to allow classes extending this class, to save the
   *          main.java.base.NeuralNetwork at the end of training using file output streams.
   */
  abstract public void train() throws Exception;

  /**
   *  @brief Saves the trained main.java.base.NeuralNetwork.
   *
   *  @param fullPath The full path of the file to save the main.java.base.NeuralNetwork.
   *
   *  @throws IOException In case main.java.base.NeuralNetwork.saveNetwork throws an exception.
   */
  public void save(String fullPath) throws IOException{
    neuralNetwork_.saveToBinary(fullPath);
  }

  /**
   *  @brief Setter method for the main.java.base.NeuralNetwork.
   *
   *  @param neuralNetwork The main.java.base.NeuralNetwork to be used.
   */
  public void setNeuralNetwork(NeuralNetwork neuralNetwork){
    neuralNetwork_ = neuralNetwork;
    sizesOfLayers_ = neuralNetwork.getSizesOfLayers();
  }

  /**
   *  @brief Getter method for the main.java.base.NeuralNetwork.
   *
   *  @return Returns the current main.java.base.NeuralNetwork.
   */
  public NeuralNetwork getNeuralNetwork(){
    return neuralNetwork_;
  }

  /**
   *  @brief Setter method for the training set path.
   *
   *  @param trainingSetPath The value for the training set path.
   */
  public void setTrainingSetPath(String trainingSetPath){
    trainingSetPath_ = trainingSetPath;
  }

  /**
   *  @brief Getter method for the training set path.
   *
   *  @return Returns the current training set path.
   */
  public String getTrainingSetPath(){
    return trainingSetPath_;
  }

  /**
   *  @brief Setter method for the training labels path.
   *
   *  @param trainingLabelsPath The value for the training labels path.
   */
  public void setTrainingLabelsPath(String trainingLabelsPath){
    trainingLabelsPath_ = trainingLabelsPath;
  }

  /**
   *  @brief Getter method for the training labels path.
   *
   *  @return Returns the current training labels path.
   */
  public String getTrainingLabelsPath(){
    return trainingLabelsPath_;
  }

  /**
   *  @brief Setter method for the testing set path.
   *
   *  @param testingSetPath The value for the testing set path.
   */
  public void setTestingSetPath(String testingSetPath){
    testingSetPath_ = testingSetPath;
  }

  /**
   *  @brief Getter method for the testing set path.
   *
   *  @return Returns the testing set path.
   */
  public String getTestingSetPath(){
    return testingSetPath_;
  }

  /**
   *  @brief Setter method for the testing labels path.
   *
   *  @param testingLabelsPath the value for the testing labels path.
   */
  public void setTestingLabelsPath(String testingLabelsPath){
    testingLabelsPath_ = testingLabelsPath;
  }

  /**
   *  @brief Getter method for the testing labels path.
   *
   *  @return Returns the testing labels path.
   */
  public String getTestingLabelsPath(){
    return testingLabelsPath_;
  }

  /**
   *  @brief Setter method for the main.java.distorters.Distorter.
   *
   *  @param distorter The main.java.distorters.Distorter to be used.
   */
  public void setDistorter(Distorter distorter){
    distorter_ = distorter;
  }

  /**
   *  @brief Getter method for the main.java.distorters.Distorter.
   *
   *  @return Returns the current main.java.distorters.Distorter.
   */
  public Distorter getDistorter(){
    return distorter_;
  }

  /**
   *  @brief Setter method for the number of training samples.
   *
   *  @param numberOfTrainingSamples The number of training samples.
   */
  public void setNumberOfTrainingSamples(int numberOfTrainingSamples){
    numberOfTrainingSamples_ = numberOfTrainingSamples;
  }

  /**
   *  @brief Getter method for the number of training samples.
   *
   *  @return Getter method for the number of training samples.
   */
  public int getNumberOfTrainingSamples(){
    return numberOfTrainingSamples_;
  }

  /**
   *  @brief Setter method for the number of testing samples.
   *
   *  @param numberOfTestingSamples The number of testing samples.
   */
  public void setNumberOfTestingSamples(int numberOfTestingSamples){
    numberOfTestingSamples_ = numberOfTestingSamples;
  }

  /**
   *  @brief Getter method for the number of testing samples.
   *
   *  @return Returns the number of testing samples.
   */
  public int getNumberOfTestingSamples(){
    return numberOfTestingSamples_;
  }

  /**
   *  @brief Setter method for the number of labels.
   *
   *  This is the size of the output layer of the main.java.base.NeuralNetwork.
   *
   *  @param numberOfLabels The number of labels.
   */
  public void setNumberOfLabels(int numberOfLabels){
    numberOfLabels_ = numberOfLabels;
  }

  /**
   *  @brief Getter method for the number of labels.
   *
   *  This is the size of the output layer of the main.java.base.NeuralNetwork.
   *
   *  @return Returns the number of labels.
   */
  public int getNumberOfLabels(){
    return numberOfLabels_;
  }

  /**
   *  @brief Setter method for the length of each sample.
   *
   *  This is the size of the input layer of the main.java.base.NeuralNetwork.
   *
   *  @param sampleLength The sample length.
   */
  public void setSampleLength(int sampleLength){
    sampleLength_ = sampleLength;
  }

  /**
   *  @brief Getter method for the length of each sample.
   *
   *  This is the size of the input layer of the main.java.base.NeuralNetwork.
   *
   *  @return Returns the sample length.
   */
  public int getSampleLength(){
    return sampleLength_;
  }

  /**
   *  @brief Setter method for the quiet mode of this Trainer.
   *
   *  @param quiet The quiet mode of this Trainer.
   */
  public void setQuiet(boolean quiet){
    quiet_ = quiet;
  }

  /**
   *  @brief Getter method for the quiet mode of this Trainer.
   *
   *  @return Returns the quiet mode of this Trainer.
   */
  public boolean isQuiet(){
    return quiet_;
  }

  /**
   *  @brief Setter method for the number of epochs.
   *
   *  An epoch is done when the parameters of the main.java.base.NeuralNetwork have been updated using the whole set of data.
   *
   *  @param numberOfEpochs The number of epochs.
   */
  public void setNumberOfEpochs(int numberOfEpochs){
    numberOfEpochs_ = numberOfEpochs;
  }

  /**
   *  @brief Getter method for the number of epochs.
   *
   *  An epoch is done when the parameters of the main.java.base.NeuralNetwork have been updated using the whole set of data.

   *  @return Returns the number of epochs.
   */
  public int getNumberOfEpochs(){
    return numberOfEpochs_;
  }

  /**
   *  @brief Setter method for the batch size.
   *
   *  A batch is the smallest amount of data that are used to update the parameters of the main.java.base.NeuralNetwork.
   *  In every epoch, the training set is split into batches and the parameters of the main.java.base.NeuralNetwork get
   *  updated as many times as the batches are.
   *
   *  @param batchSize The size of the batch.
   */
  public void setBatchSize(int batchSize){
    batchSize_ = batchSize;
  }

  /**
   *  @brief Getter method for the batch size.
   *
   *  A batch is the smallest amount of data that are used to update the parameters of the main.java.base.NeuralNetwork.
   *  In every epoch, the training set is split into batches and the parameters of the main.java.base.NeuralNetwork get
   *  updated as many times as the batches are.
   *
   *  @return Returns the size of the batch.
   */
  public int getBatchSize(){
    return batchSize_;
  }

  /**
   *  @brief Setter method for gamma.
   *
   *  Gamma is the learning rate of the training algorithm.

   *  @param gamma The value of gamma.
   */
  public void setGamma(double gamma){
    gamma_ = gamma;
  }

  /**
   *  @brief Getter method for gamma.
   *
   *  Gamma is the learning rate of the training algorithm.
   *
   *  @return Returns the value of gamma.
   */
  public double getGamma(){
    return gamma_;
  }

  /**
   *  @brief Setter method for the main.java.base.NeuralNetwork save path.
   *
   *  @param neuralNetworkSavePath The full path of the file to save the main.java.base.NeuralNetwork.
   */
  public void setNeuralNetworkSavePath(String neuralNetworkSavePath){
    neuralNetworkSavePath_ = neuralNetworkSavePath;
  }

  /**
   *  @brief Getter method for the main.java.base.NeuralNetwork save path.
   *
   *  @return Returns the main.java.base.NeuralNetwork save path.
   */
  public String getNeuralNetworkSavePath(){
    return neuralNetworkSavePath_;
  }

  protected int[] sizesOfLayers_; //!< The sizes of the layers of the main.java.base.NeuralNetwork.
  protected NeuralNetwork neuralNetwork_; //!< The main.java.base.NeuralNetwork of this Trainer.

  protected double[][] trainingSet_; //!< The training set of data of this Trainer.
  protected double[][] trainingLabels_; //!< The labels of the training set of this Trainer.
  protected double[][] testingSet_; //!< The testing set of this Trainer.
  protected int[] testingLabels_; //!< The labels of the testing set of this Trainer.

  protected int numberOfTrainingSamples_; //!< The number of training samples of this Trainer.
  protected int numberOfTestingSamples_; //!< The number of testing samples of this Trainer.
  protected int numberOfLabels_; //!< The number of labels of the main.java.base.NeuralNetwork of this Trainer.
  protected int sampleLength_; //!< The length of each sample of this Trainer.

  protected String trainingSetPath_; //!< The training set path of this Trainer.
  protected String trainingLabelsPath_; //!< The training labels path of this Trainer.
  protected String testingSetPath_; //!< The testing set path of this Trainer.
  protected String testingLabelsPath_; //!< The testing labels path of this Trainer.
  protected String neuralNetworkSavePath_; //!< The save path of the main.java.base.NeuralNetwork of this Trainer.

  protected Distorter distorter_; //!< The main.java.distorters.Distorter of this Trainer.

  protected int numberOfEpochs_; //!< The number of epochs of this Trainer.
  protected int batchSize_; //!< The size of the batch of this Trainer.

  protected double gamma_; //!< The gamma parameter of this Trainer.

  protected boolean quiet_ = true; //!< The quiet mode parameter of this Trainer.

}
