package EnsembleMethod.RandomForest.simple;

import java.io.Serializable;
import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.CommandlineRunnable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SerializedObject;
import weka.core.Utils;

public abstract class AbstractModel
		implements Cloneable, Serializable, CustomModel, FacilitiesHandler, ChoiceManager, GroupFinder {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4365642674041880270L;

	public static String DEFAULT_GROUP_SIZE = "100";
	protected String m_GroupSize = DEFAULT_GROUP_SIZE;

	public static CustomModel forName(String modelName, String[] options) throws Exception {

		return ((AbstractModel) Utils.forName(CustomModel.class, modelName, options));
	}

	  public static CustomModel getCopy(CustomModel model) throws Exception {

		    return (CustomModel) new SerializedObject(model).getObject();
		  }
	  
	  
	  public static CustomModel[] getCopies(CustomModel model, int num)
			    throws Exception {

			    if (model == null) {
			      throw new Exception("As of no model created");
			    }
			    CustomModel[] models = new CustomModel[num];
			    SerializedObject so = new SerializedObject(model);
			    for (int i = 0; i < models.length; i++) {
			    	models[i] = (CustomModel) so.getObject();
			    }
			    return models;
			  }
	  
	  public static void runModel(CustomModel model, String[] options) {
		  String classifier1 = null;
			try {
		      if (model instanceof CommandlineRunnable) {
		        ((CommandlineRunnable)model).preExecution();
		      }
		      System.out.println(Evaluation.evaluateModel(classifier1, options));
		    } catch (Exception e) {
		      if (((e.getMessage() != null)
		        && (e.getMessage().indexOf("General options") == -1))
		        || (e.getMessage() == null)) {
		        e.printStackTrace();
		      } else {
		        System.err.println(e.getMessage());
		      }
		    }
			CommandlineRunnable classifier = null;
		    if (classifier instanceof CommandlineRunnable) {
		      try {
		        ((CommandlineRunnable) classifier).postExecution();
		      } catch (Exception ex) {
		        ex.printStackTrace();
		      }
		    }
		  }
	  
	@Override
	public Capabilities getFacilities() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void buildModels(Instances data) throws Exception {
		// TODO Auto-generated method stub

	}

	@Override
	public double IndentifyInstance(Instance testData) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] frequencyOfTheInstance(Instance testData) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getresources() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setGroupSize(String size) {
		// TODO Auto-generated method stub

	}

	@Override
	public String getGroupSize() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[][] frequencyOfTheInstances(Instances data) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean implementsMoreEfficientGroupFinding() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Enumeration<Option> listOfChoices() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setChoices(String[] args) throws Exception {
		// TODO Auto-generated method stub

	}

	@Override
	public String[] getChoice() {
		// TODO Auto-generated method stub
		return null;
	}

}
