package trainers;

import base.NeuralNetwork;

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
  }
  
  /** \brief Abstract method for loading the training and testing data.
   * 
   *  \throws Exception The exeptions are thrown to allow classes extending
   *                    this class, to load data from files using
   *                    file input streams.
   */
  abstract public void load() throws Exception;
  
  /** \brief Abstract method for training the neural network on the data.
   * 
   *  \throws Exception The exeptions are thrown to allow classes extending
   *                    this class, to save the network at the end of training
   *                    using file output streams.
   */
  abstract public void train() throws Exception;
  
  protected int[] sizesOfLayers_; //!< The number of neurons in each layer.
  protected NeuralNetwork neuralNetwork_; //!< The trainer's neural network.
  
  protected double[][] trainingSet_; //!< The training set's data.
  protected double[][] trainingLabels_; //!< The training set's labels.
  protected double[][] testingSet_; //!< The testing set's data.
  protected int[] testingLabels_; //!< The testing set's labels.
  
  protected int numberOfEpochs_;
  protected int batchSize_;

  protected double gamma_;
}
