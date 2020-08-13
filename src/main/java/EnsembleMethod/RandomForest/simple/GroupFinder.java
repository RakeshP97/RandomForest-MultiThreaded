package EnsembleMethod.RandomForest.simple;

import weka.core.Instances;

public interface GroupFinder {
	

	  /**
	   * Helps to set the group size of the given Model
	   *
	   * @param size
	   */
	  void setGroupSize(String size);

	  /**
	   * Help to find the group size of the model
	   *
	   * @return string
	   */
	  String getGroupSize();

	  /**
	   * Frequency finding for the given instance
	   * 
	   * @param data
	   * @return frequency of the given instance
	   * @throws Exception
	   */
	  double[][] frequencyOfTheInstances(Instances data) throws Exception;

	  /**
	   * It return boolean value if the model able predict efficient
	   *
	   * @return boolean
	   */
	  boolean implementsMoreEfficientGroupFinding();

}
