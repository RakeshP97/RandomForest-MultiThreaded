package EnsembleMethod.RandomForest.simple;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.gui.ProgrammaticProperty;

public class CustomRandomForestImpl extends CustomBaggingImpl {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5059943149587756450L;

	protected int default_num_iterations = 1000;

	protected int defaultNumrOfIterations() {
		return default_num_iterations;
	}

	/**
	 * Construct the base model for the Random Forest and initialise few required
	 * attributes
	 */

	public CustomRandomForestImpl() {
		CustomRandomTree tree = new CustomRandomTree();
		tree.setDoNotCheckCapabilities(true);
		// Initialise the base classifier as tree
		super.setClassifier(tree);
		// Setting copies using the weights
		super.setR_ReprUsingWeights(true);
		// set the default iterations value
		setNumIterations(defaultNumrOfIterations());

	}

	/**
	 * This method will default capabilities of the given base classifier
	 * 
	 * @return Capabilities of the base classifier
	 */
	public Capabilities getCapabilities() {

		return (new CustomRandomTree()).getCapabilities();
	}

	/**
	 * This method returns the default string value of the classifier class
	 * 
	 * @return base default classifier class package
	 */
	protected String defaultClassifierString() {

		return "weka.classifiers.trees.CustomRandomTree";
	}

	/**
	 * This method returns the default classifier options
	 * 
	 * @return default options array
	 */
	protected String[] defaultClassifierOptions() {

		String[] options = { "-do-not-check-capabilities" };
		return options;
	}

	/**
	 * This method validate the given classifier instance of CustomRandomTree
	 * 
	 * @param classifier passing a new classifier
	 * @exception when classifier is not type of Random Tree
	 */
	@ProgrammaticProperty
	public void setClassifier(Classifier classifier) {

		if (classifier instanceof CustomRandomTree)
			super.setClassifier(classifier);
		else
			throw new IllegalArgumentException("Invalid classifier instance");
	}

	/**
	 * This method used to set the represent weight explicitly
	 *
	 * @param representUsingWeights
	 */

	@ProgrammaticProperty
	public void setRCopiesUsingWeights(boolean representUsingWeights) {
		if (representUsingWeights) {
			super.setR_ReprUsingWeights(representUsingWeights);
		} else {
			throw new IllegalArgumentException("This should be always ture in the case of Random Forest");
		}
	}

	/**
	 * This method used to get the number of tip texts
	 *
	 * @return tip text value
	 */
	public String numFeaturesTipText() {
		return null;
	}

	/**
	 * This method get the number of features selects randomly
	 *
	 * @return The value of the number of features.
	 */
	public int getNumFeatures() {

		return ((CustomRandomTree) getClassifier()).getkValue();
	}

	/**
	 * This allow to set the number features used to in random sub sampling
	 *
	 * @param numberFeatures set to K value .
	 */
	public void setNumFeatures(int numberFeatures) {

		((CustomRandomTree) getClassifier()).setkValue(numberFeatures);
	}

	/**
	 * 
	 * This method returns maximum depth the tip text
	 *
	 * @return tip text mainly for the GUI applications
	 */
	public String maxDepthTipText() {
		return null;
	}

	/**
	 * This method will returns the maximum depth of the tree
	 *
	 * @return value of depth of tree
	 */
	public int getMaxDepth() {
		return ((CustomRandomTree) getClassifier()).getMaxDepth();
	}

	/**
	 * This method allowed to set the depth of the tree
	 *
	 * @param depth
	 */
	public void setMaxDepth(int value) {
		((CustomRandomTree) getClassifier()).setMaxDepth(value);
	}

	/**
	 * This method returns tip text of the ties which is break by randomly
	 *
	 * @return String value tip text
	 */
	public String breakTiesRandomlyTipText() {
		return null;
	}

	/**
	 * This method allow to get the break ties random or not
	 *
	 * @return boolean
	 */
	public boolean getBreakTiesRandomly() {

		return ((CustomRandomTree) getClassifier()).isBreakTiesRandomly();
	}

	/**
	 * This allow set the Ties to break randomly or not
	 *
	 * @param boolean
	 */
	public void setBreakTiesRandomly(boolean breakTiesRandomly) {

		((CustomRandomTree) getClassifier()).setBreakTiesRandomly(breakTiesRandomly);
	}

	/**
	 * return String representation of the classifier built
	 * 
	 * @return String
	 */
	@Override
	public String toString() {

		if (m_Classifiers == null)
			return "Their is no Classifier built for now";
		else {
			StringBuilder builder = new StringBuilder();
			builder.append(super.toString());
			return builder.toString();
		}
	}

	/**
	 * This method returns all the available options including base classifier
	 * option as well
	 * 
	 * @return Enumeration of option
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> optionsVector = new Vector<Option>();

		optionsVector.addElement(new Option("Size of the each bag", "P", 1, "-P"));

		optionsVector.addElement(new Option("meassure the out of bag error", "O", 0, "-O"));

		optionsVector.addElement(new Option("It will help to decided to store the OOB errors to local evelution",
				"store-out-of-bag-predictions", 0, "-store-out-of-bag-predictions"));

		optionsVector.addElement(new Option("Allow to messaure bag complexity and statistics",
				"output-out-of-bag-complexity-statistics", 0, "-output-out-of-bag-complexity-statistics"));

		optionsVector.addElement(new Option("Allow to print the each classifier as string", "print", 0, "-print"));

		optionsVector.addElement(new Option("Allow to set the Number if Iterations for the classifier", "I", 1, "-I"));

		optionsVector.addElement(new Option("Total number of exceution slots", "num-slots", 1, "-num-slots"));

		// getting and adding base classifier options
		Enumeration<Option> baseOption = ((OptionHandler) getClassifier()).listOptions();
		optionsVector.addAll(Collections.list(baseOption));

		return optionsVector.elements();
	}

	@Override
	public String[] getOptions() {

		Vector<String> optionsVector = new Vector<String>();

		optionsVector.add("-P");
		optionsVector.add("" + getR_BagSizePer());

		if (isR_OutOfBag()) {
			optionsVector.add("-O");
		}

		if (isR_StoreOutOfBagPred()) {
			optionsVector.add("-store-out-of-bag-predictions");
		}

		if (isR_OptOutOfBagComplStat()) {
			optionsVector.add("-output-out-of-bag-complexity-statistics");
		}

		optionsVector.add("-I");
		optionsVector.add("" + getNumIterations());

		optionsVector.add("-num-slots");
		optionsVector.add("" + getNumExecutionSlots());

		if (getDoNotCheckCapabilities()) {
			optionsVector.add("-do-not-check-capabilities");
		}

		// Add base classifier options
		String[] baseOption = ((OptionHandler) getClassifier()).getOptions();

		Vector<String> baseOptions = new Vector<String>();
		Collections.addAll(baseOptions, baseOption);

		Option.deleteFlagString(baseOptions, "-do-not-check-capabilities");
		optionsVector.addAll(baseOptions);

		int s = optionsVector.size();

		return optionsVector.toArray(new String[s]);

	}

	/***
	 * Options which is used for customBagging classifier
	 * 
	 * P: Total bag size, which is the percentage of the total training set size
	 * 
	 * O: Help to calculate the number of out of bag errors
	 * 
	 * store-out-of-bag-predictions: Help to identify we have to save the out of bag
	 * failures for local analysis
	 * 
	 * output-out-of-bag-complexity-statistics: It decides we have to return
	 * complexity based calculation when the OOB is identified
	 * 
	 * represent-copies-using-weights: it represents the copies which using weight
	 * other than explicit
	 * 
	 * print: it allow to print the individual classifier
	 * 
	 * I: This will help to identify the number if iterations
	 * 
	 * do-not-check-capabilities: This allow to skip the capabilities before built a classifier
	 */

	public void setOptions(String[] arrayOptions) throws Exception {

		String bagSize = Utils.getOption('P', arrayOptions);
		if (bagSize.length() != 0) {
			setR_BagSizePer(Integer.parseInt(bagSize));
		} else {
			setR_BagSizePer(100);
		}

		setR_OutOfBag(Utils.getFlag('O', arrayOptions));

		setR_StoreOutOfBagPred(Utils.getFlag("store-out-of-bag-predictions", arrayOptions));

		setR_OptOutOfBagComplStat(Utils.getFlag("output-out-of-bag-complexity-statistics", arrayOptions));

		setR_printClassifiers(Utils.getFlag("print", arrayOptions));

		String iter = Utils.getOption('I', arrayOptions);
		if (iter.length() == 0) {
			setNumIterations(defaultNumberOfIterations());
		} else
			setNumIterations(Integer.parseInt(iter));

		String numberSlots = Utils.getOption("num-slots", arrayOptions);
		if (numberSlots.length() == 0)
			setNumExecutionSlots(1);
		else
			setNumExecutionSlots(Integer.parseInt(numberSlots));

		CustomRandomTree classifier = ((CustomRandomTree) AbstractClassifier.forName(defaultClassifierString(), arrayOptions));
		classifier.setDoNotCheckCapabilities(true);
		setDoNotCheckCapabilities(classifier.getDoNotCheckCapabilities());

		// get values from base classifier and set to local variables
		//setSeed(classifier.getSeed());
		setDebug(classifier.getDebug());
		setNumFeatures(classifier.getNumDecimalPlaces());

		setClassifier(classifier);

		Utils.checkForRemainingOptions(arrayOptions);

	}

}
