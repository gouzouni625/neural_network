package org.nn.distorters;

/** @class Distorter
 *
 *  @brief Implements and abstract Distorter.
 *
 *  A Distorter is used to apply distortions on data. This method is used to virtually increase the size of a set of
 *  data when using it to train a machine learning algorithm.
 */
public abstract class Distorter{
  /**
   *  @brief Default Constructor.
   */
  public Distorter(){
    distortFrequency_ = -1;
  }

  /**
   *  @brief Constructor.
   *
   *  @param distortFrequency The value for the distort frequency. This is a number denoting the frequency that the
   *         distortions should be applied.
   */
  public Distorter(int distortFrequency){
    distortFrequency_ = distortFrequency;
  }

  /**
   *  @brief The method that will be called to apply a distortion on a set of data.
   *
   *  @param data The set of data to apply the distortion on.
   *
   *  @return Returns the distorted data.
   */
  abstract public double[][] distort(double[][] data);

  /**
   *  @brief Setter method for the distortion frequency.
   *
   *  @param distortFrequency The value for the distortion frequency.
   */
  public void setDistortFrequency(int distortFrequency){
    distortFrequency_ = distortFrequency;
  }

  /**
   *  @brief Getter method for the distortion frequency.
   *
   *  @return Returns the current value of the distortion frequency.
   */
  public int getDistortFrequency(){
    return distortFrequency_;
  }

  private int distortFrequency_; //!< The distortion frequency of this Distorter.

}
