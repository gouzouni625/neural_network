package main.data_managers;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import com.mysql.jdbc.Connection;
import com.mysql.jdbc.Statement;

/** \class MySQLDataManager class. Connects to a MySQL database, retrieves
 *         data saved in inkML format, creates 28x28 pixels images and saves
 *         the data to binary files.    
 */
public class MySQLDataManager{
  
  public MySQLDataManager(){
  }
  
  public MySQLDataManager(String database, String databaseUsername,
                          String databasePassword, String databaseTable,
                          String databaseTableColumn){
    database_ = database;
    databaseUsername_ = databaseUsername;
    databasePassword_ = databasePassword;
    databaseTable_ = databaseTable;
    databaseTableColumn_ = databaseTableColumn;
  }

  public void loadData(String database, String databaseUsername,
                       String databasePassword, String databaseTable,
                       String databaseTableColumn)
                       throws SQLException, UnsupportedEncodingException{
    database_ = database;
    databaseUsername_ = databaseUsername;
    databasePassword_ = databasePassword;
    databaseTable_ = databaseTable;
    databaseTableColumn_ = databaseTableColumn;

    loadData();
  }

  public void loadData() throws SQLException, UnsupportedEncodingException{
    databaseData_ = new ArrayList<ArrayList<ArrayList<Point> > >();

    // Connect to the database and get the data. ==============================
    Connection connection = (Connection) DriverManager.getConnection(
              database_, databaseUsername_, databasePassword_);
    
    // Process the data. ======================================================
    Statement statement = (Statement) connection.createStatement();
    ResultSet resultSet = statement.executeQuery(
                                       "SELECT " + databaseTableColumn_ +
                                       " FROM " + databaseTable_);
      
    while(resultSet.next()){
      ArrayList<ArrayList<Point> > traces = new ArrayList<ArrayList<Point> >();
      
      // getString counts columns starting with 1.
      String traceGroups = resultSet.getString(1);
      traceGroups = URLDecoder.decode(traceGroups, "UTF-8");

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
      databaseData_.add(traces);
    }
      
    connection.close();
  }
  
  public void processData(){
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    
    // Multiply data by 100. ==================================================
    for(int i = 0;i < databaseData_.size();i++){
      for(int j = 0;j < databaseData_.get(i).size();j++){
        for(int k = 0;k < databaseData_.get(i).get(j).size();k++){
          databaseData_.get(i).get(j).get(k).x =
                     (int)(databaseData_.get(i).get(j).get(k).x * 100);
          databaseData_.get(i).get(j).get(k).y =
                     (int)(databaseData_.get(i).get(j).get(k).y * 100);
        }
      }
    }

    // Find a bounding box around the traces. =================================
    int[] minXs = new int[databaseData_.size()];
    int[] maxXs = new int[databaseData_.size()];
    int[] minYs = new int[databaseData_.size()];
    int[] maxYs = new int[databaseData_.size()];
    for(int i = 0;i < databaseData_.size();i++){
      minXs[i] = (int)databaseData_.get(i).get(0).get(0).x;
      maxXs[i] = (int)databaseData_.get(i).get(0).get(0).x;
      minYs[i] = (int)databaseData_.get(i).get(0).get(0).y;
      maxYs[i] = (int)databaseData_.get(i).get(0).get(0).y;
    }

    for(int i = 0;i < databaseData_.size();i++){
      
      for(int j = 0;j < databaseData_.get(i).size();j++){
        for(int k = 0;k < databaseData_.get(i).get(j).size();k++){
          if(databaseData_.get(i).get(j).get(k).x > maxXs[i]){
            maxXs[i] = (int)databaseData_.get(i).get(j).get(k).x;
          }
          if(databaseData_.get(i).get(j).get(k).x < minXs[i]){
            minXs[i] = (int)databaseData_.get(i).get(j).get(k).x;
          }
          if(databaseData_.get(i).get(j).get(k).y > maxYs[i]){
            maxYs[i] = (int)databaseData_.get(i).get(j).get(k).y;
          }
          if(databaseData_.get(i).get(j).get(k).y < minYs[i]){
            minYs[i] = (int)databaseData_.get(i).get(j).get(k).y;
          }
        }
      }
      
    }
    
    width_ = new int[databaseData_.size()];
    height_ = new int[databaseData_.size()];
    for(int i = 0;i < databaseData_.size();i++){
      width_[i] = maxXs[i] - minXs[i];
      if(width_[i] == 0){
        width_[i] = 10;
      }
      height_[i] = maxYs[i] - minYs[i];
      if(height_[i] == 0){
        height_[i] = 10;
      }
    }

    // Translate the traces around the beginning of the axes. =================
    for(int i = 0;i < databaseData_.size();i++){
      for(int j = 0;j < databaseData_.get(i).size();j++){
        for(int k = 0;k < databaseData_.get(i).get(j).size();k++){
          databaseData_.get(i).get(j).get(k).x -= minXs[i];
          databaseData_.get(i).get(j).get(k).y -= minYs[i];
        }
      }
    }
    
  }
  
  public void saveData(int[] dataLabels, String dataPath) throws IOException{
    saveData(dataLabels, dataPath, false, "");
  }
  
  public void saveData(int[] dataLabels, String dataPath, boolean saveImages,
                                         String imagesPath) throws IOException{
    
    // Convert data to images. ================================================
    Mat[] images = new Mat[databaseData_.size()];
    for(int i = 0;i < images.length;i++){
      images[i] = Mat.zeros(height_[i], width_[i], CvType.CV_64F);
      
      // TODO
      // Maybe randomize the thickness to get bigger set of data.
      // e.g. Print the same equation with different thicknesses
      // in order to create more than one sample per ink file.
      
      // Wanted thickness at 1000 x 1000 pixels = 30.
      int thickness = (int)((width_[i] + height_[i]) / 2 * 30 / 1000);

      for(int j = 0;j < databaseData_.get(i).size();j++){
        for(int k = 0;k < databaseData_.get(i).get(j).size() - 1;k++){
          Core.line(images[i],
              new Point(databaseData_.get(i).get(j).get(k).x,
                height_[i] - databaseData_.get(i).get(j).get(k).y),
              new Point(databaseData_.get(i).get(j).get(k + 1).x,
                height_[i] - databaseData_.get(i).get(j).get(k + 1).y),
              new Scalar(255, 255, 255), thickness);
        }
      }
    }
    
    // Process images. ========================================================
    for(int i = 0;i < images.length;i++){
      Imgproc.resize(images[i], images[i], new Size(1000, 1000));
      
      Imgproc.copyMakeBorder(images[i], images[i],
                                 1000 / 2, 1000 / 2,
                                 1000 / 2, 1000 / 2,
                                 Imgproc.BORDER_CONSTANT, new Scalar(0, 0, 0));

      Imgproc.blur(images[i], images[i], new Size(1000 / 5, 1000 / 5));
      
      Imgproc.resize(images[i], images[i], new Size(28, 28));
      
      int meanValue = 0;
      for(int j = 0;j < 28;j++){
        for(int k = 0;k < 28;k++){
          meanValue += images[i].get(j, k)[0];
        }
      }
      meanValue /= (28 * 28);
      
      int value;
      for(int j = 0;j < 28;j++){
        for(int k = 0;k < 28;k++){
          value = (int)images[i].get(j, k)[0];
          if(value > meanValue){
            images[i].put(j, k, 255);
          }
          else{
            images[i].put(j, k, 0);
          }
        }
      }

      if(saveImages){
        Highgui.imwrite(imagesPath + "/image" + i + ".tiff", images[i]);
      }
    }

    // Save image in binary format. ===========================================
    FileOutputStream fileOutputStream = null;
    DataOutputStream dataOutputStream = null;

    fileOutputStream = new FileOutputStream(dataPath + "/data");
    dataOutputStream = new DataOutputStream(fileOutputStream);
      
    dataOutputStream.writeInt(images.length);
    for(int i = 0;i < images.length;i++){
        
      for(int j = 0;j < 28;j++){
        for(int k = 0;k < 28;k++){
          dataOutputStream.writeInt((int)(images[i].get(j, k)[0]));
        }
      }
    } 
    dataOutputStream.close();
   
    // Save labels in binary format. ==========================================
    fileOutputStream = new FileOutputStream(dataPath + "/labels");
    dataOutputStream = new DataOutputStream(fileOutputStream);
      
    dataOutputStream.writeInt(dataLabels.length);
    for(int i = 0;i < dataLabels.length;i++){
      dataOutputStream.writeInt(dataLabels[i]);
    }
    dataOutputStream.close();
  }
  
  private String database_;
  private String databaseUsername_;
  private String databasePassword_;
  private String databaseTable_;
  private String databaseTableColumn_;
  
  private ArrayList<ArrayList<ArrayList<Point> > > databaseData_;
  int[] width_;
  int[] height_;
};