package EnsembleMethod.RandomForest.simple;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;


public interface CustomModel {
	

	/**
	 * This method used to build a classifier
	 * 
	 * @param data
	 * @throws Exception
	 */

  public abstract void buildModels(Instances data) throws Exception;

  /**
   * Used to classify the instance based on input 
   *
   * @param testData the instance to be classified
   * @return it will return mostly likely predicted class
   *
   * @exception throws exception when their is no prediction
   */
  public double IndentifyInstance(Instance testData) throws Exception;

  /**
   * This helps to find out the distribution of the given instance in a model 
   *
   * @param testData try to find the distribution of the instance
   * @return for given data instance distribution in a given instance
   * @exception Exception 
   */
  public double[] frequencyOfTheInstance(Instance testData) throws Exception;

  /**
   * It returns the capabilities of the given instance
   *
   * @return            capabilities of the given resource
   */
  public Capabilities getresources();
}

