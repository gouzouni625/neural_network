package main.data_managers;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;

import com.mysql.jdbc.Connection;
import com.mysql.jdbc.Statement;

import main.utilities.Point;
import main.utilities.TraceGroup;
import main.utilities.Trace;
import main.utilities.DataSet;
import main.utilities.DataSample;
import main.utilities.Utilities;

/** \class MySQLDataManager class. Connects to a MySQL database, retrieves
 *         data saved in inkML format, creates 28x28 pixels images and saves
 *         the data to binary files.
 */
public class MySQLDataManager{

  public MySQLDataManager(){
  }

  public MySQLDataManager(String database, String databaseUsername, String databasePassword, String databaseTable, String databaseTableColumn){
    database_ = database;
    databaseUsername_ = databaseUsername;
    databasePassword_ = databasePassword;
    databaseTable_ = databaseTable;
    databaseTableColumn_ = databaseTableColumn;
  }

  public void loadData(String database, String databaseUsername, String databasePassword, String databaseTable, String databaseTableColumn) throws SQLException, UnsupportedEncodingException{
    database_ = database;
    databaseUsername_ = databaseUsername;
    databasePassword_ = databasePassword;
    databaseTable_ = databaseTable;
    databaseTableColumn_ = databaseTableColumn;

    loadData();
  }

  public void loadData() throws SQLException, UnsupportedEncodingException{
    databaseData_ = new ArrayList<TraceGroup>();

    // Connect to the database and get the data. ==============================
    Connection connection = (Connection) DriverManager.getConnection(database_, databaseUsername_, databasePassword_);
    Statement statement = (Statement) connection.createStatement();
    ResultSet resultSet = statement.executeQuery("SELECT " + databaseTableColumn_ + " FROM " + databaseTable_);

    // Process the data. ======================================================
    while(resultSet.next()){
      TraceGroup traceGroup = new TraceGroup();
      String traceGroups = resultSet.getString(1);

      int startOfTrace = traceGroups.indexOf("<trace>");
      int endOfTrace = traceGroups.indexOf("</trace>");
      while(startOfTrace != -1){
        String[] traceData = traceGroups.substring(startOfTrace + 7, endOfTrace).split(", "); // ("<trace>").length = 7.

        Trace trace = new Trace();
        for(int i = 0;i < traceData.length;i++){
          double x = Double.parseDouble(traceData[i].split(" ")[0]);
          double y = Double.parseDouble(traceData[i].split(" ")[1]);
          trace.add(new Point(x, y));
        }
        traceGroup.add(trace);

        // ("</trace>").length = 8.
        traceGroups = traceGroups.substring(endOfTrace + 8);
        startOfTrace = traceGroups.indexOf("<trace>");
        endOfTrace = traceGroups.indexOf("</trace>");
      }
      databaseData_.add(traceGroup);
    }

    connection.close();
  }

  public void saveData(Size imageSize, String dataFile, int[] labels, String labelsFile, boolean saveImages, String imagesPath) throws IOException{
    Mat[] images = new Mat[databaseData_.size()];
    DataSet dataSet = new DataSet();

    for(int i = 0;i < databaseData_.size();i++){
      images[i] = databaseData_.get(i).print(imageSize);

      dataSet.add(new DataSample(Utilities.imageToByteArray(images[i]), labels[i]));

      if(saveImages){
        Highgui.imwrite(imagesPath + "/image" + i + ".tiff", images[i]);
      }
    }

    dataSet.saveMNISTLike(dataFile, labelsFile);
  }

  private String database_;
  private String databaseUsername_;
  private String databasePassword_;
  private String databaseTable_;
  private String databaseTableColumn_;

  private ArrayList<TraceGroup> databaseData_;

}
