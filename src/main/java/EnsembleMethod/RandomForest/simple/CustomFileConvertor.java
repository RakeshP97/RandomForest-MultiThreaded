package EnsembleMethod.RandomForest.simple;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CustomFileConvertor {

	 public static void convertCsvToArrf(String inputFile, String outputFile) throws Exception {

		    // load input CSV file
		    CSVLoader fileLoader = new CSVLoader();
		    fileLoader.setSource(new File(inputFile));
		    Instances data = fileLoader.getDataSet();

		    // save to output ARFF file
		    File f = new File(outputFile);
		    ArffSaver arffSaver = new ArffSaver();
		    arffSaver.setInstances(data);
		    arffSaver.setFile(f);
		    arffSaver.writeBatch();
		  }
}
