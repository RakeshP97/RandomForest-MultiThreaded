package EnsembleMethod.RandomForest.simple;

import java.util.Enumeration;

import weka.core.Option;

public interface ChoiceManager {


	  /**
	   * This method returns the list of choice of the model
	   *
	   * @return enumertaion 
	   */
	  Enumeration<Option> listOfChoices();

	  /**
	   * set the array choice as given choice
	   * @param args
	   * @exception Exception 
	   */
	  void setChoices(String[] args) throws Exception;

	  /**
	   *gets the avaliable choices for the given model
	   *
	   * @return array of choice
	   */
	 String[] getChoice();
}
