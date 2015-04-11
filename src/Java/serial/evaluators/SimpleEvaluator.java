package evaluators;

import java.io.IOException;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat; 
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import base.NeuralNetwork;

public class SimpleEvaluator{
  
  public SimpleEvaluator(String currentDirectory, int[] sizesOfLayers,
                         int numberOfRows, int numberOfColumns){
    
    System.load(currentDirectory + "/lib/lib" + Core.NATIVE_LIBRARY_NAME +
                                                                        ".so");
    sizesOfLayers_ = sizesOfLayers;
    
    numberOfRows_ = numberOfRows;
    numberOfColumns_ = numberOfColumns;
    
    sampleLength_ = sizesOfLayers[0];
    numberOfLabels_ = sizesOfLayers[sizesOfLayers.length - 1];
    
    neuralNetwork_ = new NeuralNetwork(sizesOfLayers);
  }
  
  public void loadNetwork(String path) throws IOException{
    neuralNetwork_.loadNetwork(path);
  }
  
  public String getTraceGroups(){
    return traceGroups_;
  }
  
  public void setTraceGroups(String traceGroups){
    traceGroups_ = traceGroups;
  }
  
  public int evaluateNetwork(){
    // Make sure the string stays unchanged.
    String traceGroups = traceGroups_;
    
    ArrayList<ArrayList<Point> > traces = new ArrayList<ArrayList<Point> >();

    int startOfTrace = traceGroups.indexOf("<trace>");
    int endOfTrace = traceGroups.indexOf("</trace>");
    while(startOfTrace != -1){
      String[] traceData = traceGroups.substring(startOfTrace + 7,
                          endOfTrace).split(", "); // ("<trace>").length = 7.

      ArrayList<Point> trace = new ArrayList<Point>();
      for(int i = 0;i < traceData.length;i++){
        double x = Double.parseDouble(traceData[i].split(" ")[0]);
        double y = Double.parseDouble(traceData[i].split(" ")[1]);

        trace.add(new Point(x, y));
      }
      traces.add(trace);
      
      // ("</trace>").length = 8.
      traceGroups = traceGroups.substring(endOfTrace + 8);
      startOfTrace = traceGroups.indexOf("<trace>");
      endOfTrace = traceGroups.indexOf("</trace>");
    }
    
    for(int i = 0;i < traces.size();i++){
      for(int j = 0;j < traces.get(i).size();j++){
        traces.get(i).get(j).x = (int)(traces.get(i).get(j).x * 100);
        traces.get(i).get(j).y = (int)(traces.get(i).get(j).y * 100);
      }
    }
    
    int minX = (int)(traces.get(0).get(0).x);
    int maxX = (int)(traces.get(0).get(0).x);
    int minY = (int)(traces.get(0).get(0).y);
    int maxY = (int)(traces.get(0).get(0).y);
    
    for(int i = 0;i < traces.size();i++){
      for(int j = 0;j < traces.get(i).size();j++){
        if(traces.get(i).get(j).x < minX){
          minX = (int)(traces.get(i).get(j).x);
        }
        if(traces.get(i).get(j).x > maxX){
          maxX = (int)(traces.get(i).get(j).x);
        }
        if(traces.get(i).get(j).y < minY){
          minY = (int)(traces.get(i).get(j).y);
        }
        if(traces.get(i).get(j).y > maxY){
          maxY = (int)(traces.get(i).get(j).y);
        }
      }
    }
    
    int width = maxX - minX;
    if(width == 0){
      width = 100;
    }
    int height = maxY - minY;
    if(height == 0){
      height = 100;
    }
    
    for(int i = 0;i < traces.size();i++){
      for(int j = 0;j < traces.get(i).size();j++){
        traces.get(i).get(j).x -= minX;
        traces.get(i).get(j).y -= minY;
      }
    }
    
    Mat image = Mat.zeros(height, width, CvType.CV_64F);
    
    int thickness = (int)((width + height) / 2 * 30 / 1000);
    
    for(int i = 0;i < traces.size();i++){
      for(int j = 0;j < traces.get(i).size() - 1;j++){
        Core.line(image, new Point(traces.get(i).get(j).x,
                                              height - traces.get(i).get(j).y),
                         new Point(traces.get(i).get(j + 1).x,
                                          height - traces.get(i).get(j + 1).y),
                         new Scalar(255), thickness);
      }
    }

    Imgproc.resize(image, image, new Size(1000, 1000));
    width = 1000;
    height = 1000;
    
    Imgproc.blur(image, image, new Size(height / 5, width / 5));
    
    Imgproc.resize(image, image, new Size(numberOfRows_, numberOfColumns_));
   
    int meanValue = 0;
    for(int i = 0;i < numberOfRows_;i++){
      for(int j = 0;j < numberOfColumns_;j++){
        meanValue += image.get(i, j)[0];
      }
    }
    meanValue /= numberOfRows_ * numberOfColumns_;

    int value;
    for(int i = 0;i < numberOfRows_;i++){
      for(int j = 0;j < numberOfColumns_;j++){
        value = (int)(image.get(i, j)[0]);
        if(value > meanValue){
          image.put(i, j, 255);
        }
        else{
          image.put(i, j, 0);
        }
      }
    }
    
    double[] sample = new double[sampleLength_];
    
    for(int i = 0;i < numberOfRows_;i++){
      for(int j = 0;j < numberOfColumns_;j++){
        sample[i * numberOfColumns_ + j] = image.get(i, j)[0] / 127.5 - 1;
      }
    }
    
    double[] output = neuralNetwork_.feedForward(sample);
    
    double max = output[0];
    int index = 0;
    for(int i = 0;i < output.length;i++){
      if(output[i] > max){
        max = output[i];
        index = i;
      }
    }
    
    return index;
  }

  private int[] sizesOfLayers_;
  private NeuralNetwork neuralNetwork_;
  
  private int numberOfRows_;
  private int numberOfColumns_;
  private int sampleLength_;
  private int numberOfLabels_;
  
  private static String traceGroups_;
}
