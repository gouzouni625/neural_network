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

import main.utilities.Utilities;
import main.utilities.data.DataSample;
import main.utilities.data.DataSet;
import main.utilities.traces.Point;
import main.utilities.traces.Trace;
import main.utilities.traces.TraceGroup;

/** \class MySQLDataManager class. Connects to a MySQL database, retrieves
 *         data saved in inkML format, creates images of any size and saves
 *         the data to IDX formatted files.
 */
public class MySQLDataManager{

  public MySQLDataManager(){
  }

  public MySQLDataManager(String database, String databaseUsername, String databasePassword, String databaseTable, String databaseTableDataColumn, String databaseTableLabelsColumn){
    database_ = database;
    databaseUsername_ = databaseUsername;
    databasePassword_ = databasePassword;
    databaseTable_ = databaseTable;
    databaseTableDataColumn_ = databaseTableDataColumn;
    databaseTableLabelsColumn_ = databaseTableLabelsColumn;
  }

  public void loadFromDatabase(String database, String databaseUsername, String databasePassword, String databaseTable, String databaseTableDataColumn, String databaseTableLabelsColumn) throws SQLException, UnsupportedEncodingException{
    database_ = database;
    databaseUsername_ = databaseUsername;
    databasePassword_ = databasePassword;
    databaseTable_ = databaseTable;
    databaseTableDataColumn_ = databaseTableDataColumn;
    databaseTableLabelsColumn_ = databaseTableLabelsColumn;

    loadFromDatabase();
  }

  public void loadFromDatabase() throws SQLException, UnsupportedEncodingException{
    databaseData_ = new ArrayList<TraceGroup>();

    // Connect to the database and get the data. ==============================
    Connection connection = (Connection) DriverManager.getConnection(database_, databaseUsername_, databasePassword_);
    Statement statement = (Statement) connection.createStatement();
    ResultSet resultSet = statement.executeQuery("SELECT " + databaseTableDataColumn_ + " FROM " + databaseTable_);

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

    // Retrieve labels.
    if(databaseTableLabelsColumn_ != null){
      labels_ = new ArrayList<Byte>();

      resultSet = statement.executeQuery("SELECT " + databaseTableLabelsColumn_ + " FROM " + databaseTable_);

      while(resultSet.next()){
        labels_.add(Byte.parseByte(resultSet.getString(1)));
      }
    }

    connection.close();
  }

  public void saveToDatabase(String[] data) throws SQLException{
    Connection connection = (Connection) DriverManager.getConnection(database_, databaseUsername_, databasePassword_);
    Statement statement = (Statement) connection.createStatement();

    String values = new String("");
    for(String value : data){
      values += value + ", ";
    }
    // Remove last comma from values.
    values = values.substring(0, values.length() - 2);

    String query = "INSERT INTO " + databaseTable_ + " VALUES (" + values + ")";
    System.out.println(query);
    statement.executeUpdate(query);

    connection.close();
  }

  public void saveToIDX(Size imageSize, String dataFile, String labelsFile, boolean saveImages, String imagesPath) throws IOException{
    byte[] labels = new byte[labels_.size()];
    for(int i = 0;i < labels_.size();i++){
      labels[i] = labels_.get(i);
    }

    this.saveToIDX(imageSize, dataFile, labels, labelsFile, saveImages, imagesPath);
  }

  public void saveToIDX(Size imageSize, String dataFile, byte[] labels, String labelsFile, boolean saveImages, String imagesPath) throws IOException{
    Mat[] images = new Mat[databaseData_.size()];
    DataSet dataSet = new DataSet();

    for(int i = 0;i < databaseData_.size();i++){
      images[i] = databaseData_.get(i).print(imageSize);

      dataSet.add(new DataSample(Utilities.imageToByteArray(images[i]), labels[i]));

      if(saveImages){
        Highgui.imwrite(imagesPath + "/image" + i + ".tiff", images[i]);
      }
    }

    dataSet.saveIDXFormat(dataFile, labelsFile);
  }

  public ArrayList<TraceGroup> getDatabaseData(){
    return databaseData_;
  }

  public ArrayList<Byte> getLabels(){
    return labels_;
  }

  private String database_;
  private String databaseUsername_;
  private String databasePassword_;
  private String databaseTable_;
  private String databaseTableDataColumn_;
  private String databaseTableLabelsColumn_;

  private ArrayList<TraceGroup> databaseData_;
  private ArrayList<Byte> labels_;

}
