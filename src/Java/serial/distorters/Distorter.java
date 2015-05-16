package distorters;

public abstract class Distorter{
  public Distorter(int distortFrequency){
    distortFrequency_ = distortFrequency;
  }
  
  abstract public double[][] distort(double[][] data);
  
  public void setDistortFrequency(int distortFrequency){
    distortFrequency_ = distortFrequency;
  }
  
  public int getDistortFrequency(){
    return distortFrequency_;
  }
  
  private int distortFrequency_;
}
