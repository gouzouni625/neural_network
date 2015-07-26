package main.trainers;

import java.io.IOException;

import main.base.NeuralNetwork;
import main.distorters.Distorter;
import main.utilities.data.DataSet;

/** \class Abstract class Trainer.
 *  \brief Implements a neural network trainer.
 */
public abstract class Trainer{

  /** \brief Constructor of the class. Initializes the neural network.
   *  \param sizesOfLayers The number of neurons in each layer including
   *         the input and output layers. The number of layers is equal
   *         to sizesOfLayers.length.
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

  public Trainer(int[] sizesOfLayers, String trainingSetPath,
                                      String trainingLabelsPath,
                                      String testingSetPath,
                                      String testingLabelsPath){
    sizesOfLayers_ = sizesOfLayers;
    neuralNetwork_ = new NeuralNetwork(sizesOfLayers);

    trainingSetPath_ = trainingSetPath;
    trainingLabelsPath_ = trainingLabelsPath;
    testingSetPath_ = testingSetPath;
    testingLabelsPath_ = testingLabelsPath;

    distorter_ = null;
  }

  public Trainer(int[] sizesOfLayers, String trainingSetPath,
                                      String trainingLabelsPath,
                                      String testingSetPath,
                                      String testingLabelsPath,
                                      Distorter distorter){
    sizesOfLayers_ = sizesOfLayers;
    neuralNetwork_ = new NeuralNetwork(sizesOfLayers);

    trainingSetPath_ = trainingSetPath;
    trainingLabelsPath_ = trainingLabelsPath;
    testingSetPath_ = testingSetPath;
    testingLabelsPath_ = testingLabelsPath;

    distorter_ = distorter;
  }

  /** \brief Abstract method for loading the training and testing data.
   *
   *  \throws Exception The exeptions are thrown to allow classes extending
   *                    this class, to load data from files using
   *                    file input streams.
   */
  abstract public void load(DataSet trainingSet, DataSet testingSet) throws Exception;

  /** \brief Abstract method for training the neural network on the data.
   *
   *  \throws Exception The exeptions are thrown to allow classes extending
   *                    this class, to save the network at the end of training
   *                    using file output streams.
   */
  abstract public void train() throws Exception;

  public void save(String fullPath) throws IOException{
    neuralNetwork_.saveNetwork(fullPath);
  }

  public void save() throws IOException{
         save("/home/george/Desktop/workspace/resources/data/github/" +
              "repositories/neural_network/data/trained_networks/" +
              "network" +
              "_tr" + trainingSet_.length + "_e" + numberOfEpochs_ +
              "_b" + batchSize_ + "_g" + gamma_);
  }

  /** \brief Training and testing data values are mapped in [-1, 1].
   */
  protected void prepareData(){
    for(int i = 0;i < numberOfTrainingSamples_;i++){
      for(int j = 0;j < sampleLength_;j++){
        trainingSet_[i][j] = trainingSet_[i][j] / 127.5 - 1;
      }
    }

    for(int i = 0;i < numberOfTestingSamples_;i++){
      for(int j = 0;j < sampleLength_;j++){
        testingSet_[i][j] = testingSet_[i][j] / 127.5 - 1;
      }
    }
  }

  public void setNeuralNetwork(NeuralNetwork neuralNetwork){
    neuralNetwork_ = neuralNetwork;
    sizesOfLayers_ = neuralNetwork.getSizesOfLayers();
  }

  public NeuralNetwork getNeuralNetwork(){
    return neuralNetwork_;
  }

  public void setTrainingSetPath(String trainingSetPath){
    trainingSetPath_ = trainingSetPath;
  }

  public String getTrainingSetPath(){
    return trainingSetPath_;
  }

  public void setTrainingLabelsPath(String trainingLabelsPath){
    trainingLabelsPath_ = trainingLabelsPath;
  }

  public String getTrainingLabelsPath(){
    return trainingLabelsPath_;
  }

  public void setTestingSetPath(String testingSetPath){
    testingSetPath_ = testingSetPath;
  }

  public String getTestingSetPath(){
    return testingSetPath_;
  }

  public void setTestingLabelsPath(String testingLabelsPath){
    testingLabelsPath_ = testingLabelsPath;
  }

  public String getTestingLabelsPath(){
    return testingLabelsPath_;
  }

  public void setDistorter(Distorter distorter){
    distorter_ = distorter;
  }

  public Distorter getDistorter(){
    return distorter_;
  }

  public void setNumberOfTrainingSamples(int numberOfTrainingSamples){
    numberOfTrainingSamples_ = numberOfTrainingSamples;
  }

  public int getNumberOfTrainingSamples(){
    return numberOfTrainingSamples_;
  }

  public void setNumberOfTestingSamples(int numberOfTestingSamples){
    numberOfTestingSamples_ = numberOfTestingSamples;
  }

  public int getNumberOfTestingSamples(){
    return numberOfTestingSamples_;
  }

  public void setNumberOfLabels(int numberOfLabels){
    numberOfLabels_ = numberOfLabels;
  }

  public int getNumberOfLabels(){
    return numberOfLabels_;
  }

  public void setSampleLength(int sampleLength){
    sampleLength_ = sampleLength;
  }

  public int getSampleLength(){
    return sampleLength_;
  }

  public void setQuiet(boolean quiet){
    quiet_ = quiet;
  }

  public boolean getQuiet(){
    return quiet_;
  }

  public void setNumberOfEpochs(int numberOfEpochs){
    numberOfEpochs_ = numberOfEpochs;
  }

  public int getNumberOfEpochs(){
    return numberOfEpochs_;
  }

  public void setBatchSize(int batchSize){
    batchSize_ = batchSize;
  }

  public int getBatchSize(){
    return batchSize_;
  }

  public void setGamma(double gamma){
    gamma_ = gamma;
  }

  public double getGamma(){
    return gamma_;
  }

  public void setNeuralNetworkSavePath(String neuralNetworkSavePath){
    neuralNetworkSavePath_ = neuralNetworkSavePath;
  }

  public String getNeuralNetworkSavePath(){
    return neuralNetworkSavePath_;
  }

  protected int[] sizesOfLayers_; //!< The number of neurons in each layer.
  protected NeuralNetwork neuralNetwork_; //!< The trainer's neural network.

  protected double[][] trainingSet_; //!< The training set's data.
  protected double[][] trainingLabels_; //!< The training set's labels.
  protected double[][] testingSet_; //!< The testing set's data.
  protected int[] testingLabels_; //!< The testing set's labels.

  protected int numberOfTrainingSamples_;
  protected int numberOfTestingSamples_;
  protected int numberOfLabels_;
  protected int sampleLength_;

  protected String trainingSetPath_;
  protected String trainingLabelsPath_;
  protected String testingSetPath_;
  protected String testingLabelsPath_;
  protected String neuralNetworkSavePath_;

  protected Distorter distorter_;

  protected int numberOfEpochs_;
  protected int batchSize_;

  protected double gamma_;

  protected boolean quiet_ = false;

}
