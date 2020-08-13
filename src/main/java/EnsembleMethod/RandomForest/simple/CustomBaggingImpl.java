package EnsembleMethod.RandomForest.simple;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.core.AdditionalMeasureProducer;
import weka.core.Aggregateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.PartitionGenerator;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class CustomBaggingImpl extends RandomizableParallelIteratedSingleClassifierEnhancer
		implements WeightedInstancesHandler, AdditionalMeasureProducer, TechnicalInformationHandler, PartitionGenerator,
		Aggregateable<CustomBaggingImpl> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2592565544455795490L;

	protected List<Classifier> r_classifierList;

	// This evalution object holding the out of bag error value
	protected Evaluation r_OutOfBagEvalObj = null;

	protected boolean r_OutOfBag = false;

	protected Random random;

	protected boolean[][] bag;

	protected Instances r_data;

	protected int r_BagSizePer = 100;

	protected boolean r_ReprUsingWeights = false;

	private boolean r_StoreOutOfBagPred = false;

	private boolean r_printClassifiers;

	private boolean r_OptOutOfBagComplStat;

	// validate the given class is numeric or not
	private boolean r_Numeric = false;

	/**
	 * @return the r_BagSizePer
	 */
	public int getR_BagSizePer() {
		return r_BagSizePer;
	}

	/**
	 * @param r_BagSizePer the r_BagSizePer to set
	 */
	public void setR_BagSizePer(int r_BagSizePer) {
		this.r_BagSizePer = r_BagSizePer;
	}

	/**
	 * @return 
	 * 
	 */
	public  CustomBaggingImpl() {
		m_Classifier = new CustomRandomTree();
	}

	public boolean isR_ReprUsingWeights() {
		return r_ReprUsingWeights;
	}

	public void setR_ReprUsingWeights(boolean r_ReprUsingWeights) {
		this.r_ReprUsingWeights = r_ReprUsingWeights;
	}

	/**
	 * @return the r_OutOfBag
	 */
	public boolean isR_OutOfBag() {
		return r_OutOfBag;
	}

	/**
	 * @param r_OutOfBag the r_OutOfBag to set
	 */
	public void setR_OutOfBag(boolean r_OutOfBag) {
		this.r_OutOfBag = r_OutOfBag;
	}

	/**
	 * @return the r_StoreOutOfBagPred
	 */
	public boolean isR_StoreOutOfBagPred() {
		return r_StoreOutOfBagPred;
	}

	/**
	 * @param r_StoreOutOfBagPred the r_StoreOutOfBagPred to set
	 */
	public void setR_StoreOutOfBagPred(boolean r_StoreOutOfBagPred) {
		this.r_StoreOutOfBagPred = r_StoreOutOfBagPred;
	}

	/**
	 * @return the r_printClassifiers
	 */
	public boolean isR_printClassifiers() {
		return r_printClassifiers;
	}

	/**
	 * @param r_printClassifiers the r_printClassifiers to set
	 */
	public void setR_printClassifiers(boolean r_printClassifiers) {
		this.r_printClassifiers = r_printClassifiers;
	}

	/**
	 * @return the r_OptOutOfBagComplStat
	 */
	public boolean isR_OptOutOfBagComplStat() {
		return r_OptOutOfBagComplStat;
	}

	/**
	 * @param r_OptOutOfBagComplStat the r_OptOutOfBagComplStat to set
	 */
	public void setR_OptOutOfBagComplStat(boolean r_OptOutOfBagComplStat) {
		this.r_OptOutOfBagComplStat = r_OptOutOfBagComplStat;
	}

	/**
	 * This method gather all classifier objects
	 * 
	 * @param CustomBaggingImpl object to gather
	 * @return current object reference
	 * @throws Ecpetion Object will throws
	 */

	@Override
	public CustomBaggingImpl aggregate(CustomBaggingImpl toAggregate) throws Exception {

		// check for the if object is aggregateable or not
		if (!m_Classifier.getClass().isAssignableFrom(toAggregate.m_Classifier.getClass())) {
			throw new Exception("The Object is not aggregatable");
		}

		if (r_classifierList == null) {
			r_classifierList = new ArrayList<Classifier>();
			r_classifierList.add(toAggregate);

		}
		r_classifierList.add(toAggregate);

		return this;
	}

	/**
	 * This method to helps to finish the aggregate process and clear the cache
	 */
	@Override
	public void finalizeAggregation() throws Exception {
		m_Classifiers = r_classifierList.toArray(new Classifier[1]);
		m_NumIterations = m_Classifiers.length;

		// Clearing the cache list values
		r_classifierList = null;

	}

	/**
	 * This method used to generate the partitions of the data
	 */
	@Override
	public void generatePartition(Instances data) throws Exception {

		if (m_Classifier instanceof PartitionGenerator) {
			// then build classifier with the instance
			buildClassifier(data);
		} else {
			throw new Exception("Classifier cannot be generated for the given instance");
		}
	}

	/**
	 * This method calculate the array of the values which indicates the leaf node
	 */

	@Override
	public double[] getMembershipValues(Instance inst) throws Exception {

		double[] v = new double[20000];

		if (m_Classifier instanceof PartitionGenerator) {
			List<double[]> list = new ArrayList<double[]>();
			int lenght = 0;
			for (int i = 0; i < m_Classifiers.length; i++) {
				double[] values = ((PartitionGenerator) m_Classifiers[i]).getMembershipValues(inst);
				list.add(values);

				System.arraycopy(values, 0, v, lenght, values.length);
				lenght += values.length;

			}
		} else {
			throw new Exception("Classifier not allow generate Memebership values");
		}
		return v;
	}

	/**
	 * 
	 * This method returns the total number of elements
	 */
	@Override
	public int numElements() throws Exception {

		int size = 0;
		if (m_Classifier instanceof PartitionGenerator) {
			for (int i = 0; i < m_Classifiers.length; i++) {
				size += ((PartitionGenerator) m_Classifiers[i]).numElements();

			}
		} else {
			throw new Exception("Classifier cannot be messaured");
		}
		return size;
	}

	/**
	 * This method will generate the technical information of the developer
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation technical = new TechnicalInformation(Type.UNPUBLISHED);

		technical.setValue(Field.AUTHOR, "RP");

		return technical;
	}

	/**
	 * This method will returns the additional measures of Bagging
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {

		Vector<String> v = new Vector<String>(1);
		v.add("messaureOutofBag");
		return v.elements();
	}

	/**
	 * 
	 */
	public double measureOBagError() {
		if (r_OutOfBagEvalObj == null)
			return -1;
		if (r_Numeric)
			return r_OutOfBagEvalObj.meanAbsoluteError();
		else
			return r_OutOfBagEvalObj.errorRate();
	}

	/**
	 * This method returns the value if the measure name
	 */

	@Override
	public double getMeasure(String measureName) {
		if (measureName.equalsIgnoreCase("messaureOutofBag")) {
			return measureOBagError();
		} else
			throw new IllegalArgumentException(measureName + " is not supported");

	}

	/**
	 * 
	 * generate a training set for the given value
	 * 
	 * @param iteration
	 * @return
	 * @throws Exception
	 */
	@Override
	protected synchronized Instances getTrainingSet(int iteration) throws Exception {
		int bagSize = (int) (r_data.numInstances() * (r_BagSizePer / 100.0));
		Instances data = null;
		Random random = new Random(m_Seed + iteration);

		if (r_OutOfBag) {
			bag[iteration] = new boolean[r_data.numInstances()];
			data = r_data.resampleWithWeights(random, bag[iteration], isR_ReprUsingWeights());

		} else {
			if (bagSize < r_data.numInstances()) {
				data = r_data.resampleWithWeights(random, bag[iteration], isR_ReprUsingWeights());
				data.randomize(random);
				Instances newBag = new Instances(data, 0, bagSize);
				data = newBag;
			} else {
				data = r_data.resampleWithWeights(random, isR_ReprUsingWeights());
			}
		}

		return data;
	}

	/**
	 * 
	 * This is used to building classifier
	 * 
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		// check for the classifier to handle
		getCapabilities().testWithFail(data);

		if (isR_ReprUsingWeights() && !(m_Classifier instanceof WeightedInstancesHandler)) {
			throw new IllegalArgumentException("Exception WeightedInstancesHandler ");
		}

		// get new instance object
		r_data = new Instances(data);
		super.buildClassifier(r_data);
		if (r_OutOfBag && (r_BagSizePer != 100)) {
			throw new IllegalArgumentException("Bag size need to match 100 percentage");
		}

		random = new Random(m_Seed);
		bag = null;
		if (r_OutOfBag) {
			bag = new boolean[m_Classifiers.length][];
		}

		for (int j = 0; j < m_Classifiers.length; j++) {
			if (m_Classifier instanceof Randomizable) {
				((Randomizable) m_Classifiers[j]).setSeed(random.nextInt());
			}
		}

		r_Numeric = r_data.classAttribute().isNumeric();

		buildParllelClassifiers();

		// calculate the One Out of Bag error?
		if (isR_OutOfBag()) {
			r_OutOfBagEvalObj = new Evaluation(r_data);

			for (int i = 0; i < r_data.numInstances(); i++) {
				double[] votes;
				if (r_Numeric)
					votes = new double[1];
				else
					votes = new double[r_data.numClasses()];

				// determine predictions for instance
				int numVotes = 0;
				for (int j = 0; j < m_Classifiers.length; j++) {
					if (bag[j][i])
						continue;

					if (r_Numeric) {
						double pred = m_Classifiers[j].classifyInstance(r_data.instance(i));
						if (!Utils.isMissingValue(pred)) {
							votes[0] += pred;
							numVotes++;
						}
					} else {
						numVotes++;
						double[] newProbs = m_Classifiers[j].distributionForInstance(r_data.instance(i));
						// sum the probability estimates
						for (int k = 0; k < newProbs.length; k++) {
							votes[k] += newProbs[k];
						}
					}
				}

				if (r_Numeric) {
					if (numVotes > 0) {
						votes[0] /= numVotes;
						r_OutOfBagEvalObj.evaluationForSingleInstance(votes, r_data.instance(i),
								isR_StoreOutOfBagPred());
					}
				} else {
					double sum = Utils.sum(votes);
					if (sum > 0) {
						Utils.normalize(votes, sum);
						r_OutOfBagEvalObj.evaluationForSingleInstance(votes, r_data.instance(i),
								isR_StoreOutOfBagPred());
					}
				}
			}
		} else {
			r_OutOfBagEvalObj = null;
		}

		r_data = null;
	}

	protected void buildParllelClassifiers() throws Exception {

		if (m_numExecutionSlots != 0) {

			// Defaulted number of threads in eclipse
			int numumberOfThreads = 10;
			// trying to get the number of processor available
			int numberOfCores = (m_numExecutionSlots == 1) ? Runtime.getRuntime().availableProcessors()
					: m_numExecutionSlots;
			ExecutorService executorPool = Executors.newFixedThreadPool(numberOfCores * numumberOfThreads);

			final CountDownLatch latch = new CountDownLatch(m_Classifiers.length);
			final AtomicInteger atomic = new AtomicInteger();
			long start = System.currentTimeMillis();
			for (int i = 0; i < m_Classifiers.length; i++) {

				final Classifier currentClassifier = m_Classifiers[i];
				// Check for the classifier is NULL or not
				if (currentClassifier == null)
					continue;
				final int iteration = i;
				// Making Classifier creation is parallel
				Runnable newTask = new Runnable() {
					@Override
					public void run() {
						try {
							currentClassifier.buildClassifier(getTrainingSet(iteration));
						} catch (Throwable ex) {
							ex.printStackTrace();
							atomic.incrementAndGet();

						} finally {
							latch.countDown();
						}
					}
				};
				// launch this task
				executorPool.submit(newTask);
			}
			long finish = System.currentTimeMillis();
			System.out.println("Total Time take to submit the threads in mill Seconds:" + (finish - start));
			System.out.println("Total number of threads is active:" + Thread.activeCount());
			// make sure all the threads got completed
			latch.await();
			executorPool.shutdownNow();
			// Check for the error based on failures noted
			if (m_Debug && atomic.intValue() > 0) {
				System.err.println(
						"There is error occurs whicle building classifiers - because of that some iterations failed.");
			}
			long finish1 = System.currentTimeMillis();
			System.out.println("Total Time took to complete the Threads in mill Seconds:" + (finish1 - start));

		} else {
			long start = System.currentTimeMillis();
			// simple single-threaded execution
			for (int i = 0; i < m_Classifiers.length; i++) {
				m_Classifiers[i].buildClassifier(getTrainingSet(i));
			}
			long finish = System.currentTimeMillis();
			System.out.println("Total number of threads is active:" + Thread.activeCount());
			System.out.println(
					"Total Time took to complete running as single thread in mill seconds:" + (finish - start));
		}
	}

	/**
	 * 
	 * Identify the distribution of the given instance in the classifiers
	 * 
	 * @param Instance
	 * @return double[]
	 * @throws Exception
	 * 
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		double[] s = new double[instance.numClasses()], newProbability;

		double numPredictions = 0;
		for (int i = 0; i < m_NumIterations; i++) {
			if (r_Numeric) {
				double pred = m_Classifiers[i].classifyInstance(instance);
				if (!Utils.isMissingValue(pred)) {
					s[0] += pred;
					numPredictions++;
				}
			} else {
				newProbability = m_Classifiers[i].distributionForInstance(instance);
				for (int j = 0; j < newProbability.length; j++)
					s[j] += newProbability[j];
			}
		}
		if (r_Numeric) {
			if (numPredictions == 0) {
				s[0] = Utils.missingValue();
			} else {
				s[0] /= numPredictions;
			}
			return s;
		} else if (Utils.eq(Utils.sum(s), 0)) {
			return s;
		} else {
			Utils.normalize(s);
			return s;
		}
	}

	@Override
	public String toString() {

		if (m_Classifiers == null)
			return "No Classifier are built for now";
		StringBuilder builder = new StringBuilder();
		builder.append("Total number of baggs is:" + getNumIterations() + " and the base learnser config is:"
				+ getClassifierSpec());
		if (isR_printClassifiers()) {
			for (Classifier c : m_Classifiers)
				builder.append(c.toString());
		}

		return builder.toString();
	}

	/**
	 * 
	 * This method allow to getting the list options which set to create the
	 * classifier
	 * 
	 * @return array of options
	 */
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

		if (isR_ReprUsingWeights()) {
			optionsVector.add("-represent-copies-using-weights");
		}

		if (isR_printClassifiers()) {
			optionsVector.add("-print");
		}

		Collections.addAll(optionsVector, super.getOptions());

		return optionsVector.toArray(new String[0]);
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
	 */

	@Override
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

		setR_ReprUsingWeights(Utils.getFlag("represent-copies-using-weights", arrayOptions));

		setR_printClassifiers(Utils.getFlag("print", arrayOptions));

		super.setOptions(arrayOptions);

		Utils.checkForRemainingOptions(arrayOptions);
	}
}
