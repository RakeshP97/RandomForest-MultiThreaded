package EnsembleMethod.RandomForest.simple;

import java.io.Serializable;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class CustomRandomTree2 extends AbstractClassifier implements WeightedInstancesHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1971232304431176379L;

	protected CustomTree CustTree = null;

	protected Instances treeInfo = null;

	// Number nodes at each leaf
	protected double minNumber = 1.0;

	protected int numFolds = 0;

	protected Classifier zeroRandom;

	protected double minVarProp = 1e-3;

	protected boolean allowUnclassifiedInstances = false;

	protected int maxDepth = 0;

	protected boolean breakTiesRandomly = false;

	// Number features used to split
	protected int kValue = 0;

	protected int randomSeed = 1;

	/**
	 * @return the kValue
	 */
	public int getkValue() {
		return kValue;
	}

	/**
	 * @param kValue the kValue to set
	 */
	public void setkValue(int kValue) {
		this.kValue = kValue;
	}

	/**
	 * @return the randomSeed
	 */
	public int getRandomSeed() {
		return randomSeed;
	}

	/**
	 * @param randomSeed the randomSeed to set
	 */
	public void setRandomSeed(int randomSeed) {
		this.randomSeed = randomSeed;
	}

	/**
	 * @return the breakTiesRandomly
	 */
	public boolean isBreakTiesRandomly() {
		return breakTiesRandomly;
	}

	/**
	 * @param breakTiesRandomly the breakTiesRandomly to set
	 */
	public void setBreakTiesRandomly(boolean breakTiesRandomly) {
		this.breakTiesRandomly = breakTiesRandomly;
	}

	/**
	 * @param m_AllowUnclassifiedInstances the m_AllowUnclassifiedInstances to set
	 */
	public void setAllowUnclassifiedInstances(boolean allowUnclassifiedInstances) {
		this.allowUnclassifiedInstances = allowUnclassifiedInstances;
	}

	/**
	 * @return the maxDepth
	 */
	public int getMaxDepth() {
		return maxDepth;
	}

	/**
	 * @param maxDepth the maxDepth to set
	 */
	public void setMaxDepth(int maxDepth) {
		this.maxDepth = maxDepth;
	}

	/**
	 * @return the m_AllowUnclassifiedInstances
	 */
	public boolean isAllowUnclassifiedInstances() {
		return allowUnclassifiedInstances;
	}

	@Override
	public void buildClassifier(Instances inst) throws Exception {

		// Make sure K value is in range
		if (kValue > inst.numAttributes() - 1) {
			kValue = inst.numAttributes() - 1;
		}
		if (kValue < 1) {
			kValue = (int) Utils.log2(inst.numAttributes() - 1) + 1;
		}

		// can classifier handle the data?
		getCapabilities().testWithFail(inst);

		// remove instances with missing class
		inst = new Instances(inst);
		inst.deleteWithMissingClass();

		// only class? -> build ZeroR model
		if (inst.numAttributes() == 1) {
			System.err.println(
					"Cannot build model (only class attribute present in data!), " + "using ZeroR model instead!");
			zeroRandom = new weka.classifiers.rules.ZeroR();
			zeroRandom.buildClassifier(inst);
			return;
		} else {
			zeroRandom = null;
		}

		// Figure out appropriate datasets
		Instances train = null;
		Instances backfit = null;
		Random rand = inst.getRandomNumberGenerator(randomSeed);
		if (numFolds <= 0) {
			train = inst;
		} else {
			inst.randomize(rand);
			inst.stratify(numFolds);
			train = inst.trainCV(numFolds, 1, rand);
			backfit = inst.testCV(numFolds, 1);
		}

		// Create the attribute indices window
		int[] attributeIndsWindow = new int[inst.numAttributes() - 1];
		int j = 0;
		for (int i = 0; i < attributeIndsWindow.length; i++) {
			if (j == inst.classIndex()) {
				j++; 
			}
			attributeIndsWindow[i] = j++;
		}

		double sumOfWeight = 0;
		double sumSquaredTotal = 0;

		// Compute initial class counts
		double[] classProbs = new double[train.numClasses()];
		for (int i = 0; i < train.numInstances(); i++) {
			Instance data = train.instance(i);
			if (data.classAttribute().isNominal()) {
				classProbs[(int) data.classValue()] += data.weight();
				sumOfWeight += data.weight();
			} else {
				classProbs[0] += data.classValue() * data.weight();
				sumSquaredTotal += data.classValue() * data.classValue() * data.weight();
				sumOfWeight += data.weight();
			}
		}

		double trainVariance = 0;
		if (inst.classAttribute().isNumeric()) {
			trainVariance = CustomRandomTree2.oneVariance(classProbs[0], sumSquaredTotal, sumOfWeight) / sumOfWeight;
			classProbs[0] /= sumOfWeight;
		}

		// Used to build custom tree
		CustTree = new CustomTree();
		treeInfo = new Instances(inst, 0);
		CustTree.buildCustomTree(train, classProbs, attributeIndsWindow, sumOfWeight, rand, 0, minVarProp * trainVariance);

		/*
		 * // if needs back fitting data to tree if (backfit != null) {
		 * CustTree.backfitData(backfit); }
		 */

	}

	/**
	 * Calculate the single point variance
	 * 
	 * @param d1
	 * @param d2
	 * @param w 
	 * @return the variance
	 */
	protected static double oneVariance(double d1, double d2, double w) {

		return d2 - ((d1 * d1) / w);
	}

	/**
	 * Calculate the variance of the given set of data
	 * 
	 * @param d1
	 * @param d2
	 * @param totalWeights
	 * @return the variance
	 */
	protected static double variance(double[] d1, double[] d2, double[] totalWeights) {

		double variance = 0;

		for (int i = 0; i < d1.length; i++) {
			if (totalWeights[i] > 0) {
				variance += oneVariance(d1[i], d2[i], totalWeights[i]);
			}
		}

		return variance;
	}

	class CustomTree implements Serializable {


		/**
		 * 
		 */
		private static final long serialVersionUID = -1361419761678733784L;

		protected CustomTree[] next;

		protected int features = -1;

		protected double splitPoint = Double.NaN;

		protected double[] proportion = null;

		
		protected double[] modelProb = null;
		
		protected double[] probability = null;

		
		public void backfitData(Instances data) throws Exception {

			double totalWeight = 0;
			double totalSumSquared = 0;

			// Compute initial class counts
			double[] classProbs = new double[data.numClasses()];
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				if (data.classAttribute().isNominal()) {
					classProbs[(int) inst.classValue()] += inst.weight();
					totalWeight += inst.weight();
				} else {
					classProbs[0] += inst.classValue() * inst.weight();
					totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
					totalWeight += inst.weight();
				}
			}

			double trainVariance = 0;
			if (data.classAttribute().isNumeric()) {
				trainVariance = CustomRandomTree2.oneVariance(classProbs[0], totalSumSquared, totalWeight)
						/ totalWeight;
				classProbs[0] /= totalWeight;
			}

			// Fit data into CustomTree
			backfitData(data, classProbs, totalWeight);
		}

		/**
		 * Calculate the distribution of the custom tree
		 * 
		 * @param data 
		 * @return model dist
		 * @throws Exception 
		 */
		public double[] distributionForInstance(Instance data) throws Exception {

			double[] dist = null;

			if (features > -1) {

				if (treeInfo.attribute(features).isNominal()) {
					dist = next[(int) data.value(features)].distributionForInstance(data);

				} else if (data.isMissing(features)) {

					dist = new double[treeInfo.numClasses()];

					// Split instance up
					for (int i = 0; i < next.length; i++) {
						double[] help = next[i].distributionForInstance(data);
						if (help != null) {
							for (int j = 0; j < help.length; j++) {
								dist[j] += proportion[i] * help[j];
							}
						}
					}
				} else {
					if (data.value(features) >= splitPoint) {
						dist = next[1].distributionForInstance(data);
					} else {
						dist = next[0].distributionForInstance(data);
					}
				}
			}

			if ((features == -1) || (dist == null)) {

				// Is node empty?
				if (modelProb == null) {
					if (isAllowUnclassifiedInstances()) {
						double[] result = new double[treeInfo.numClasses()];
						if (treeInfo.classAttribute().isNumeric()) {
							result[0] = Utils.missingValue();
						}
						return result;
					} else {
						return null;
					}
				}
				double[] normalizedDistribution = modelProb.clone();
				if (treeInfo.classAttribute().isNominal()) {
					Utils.normalize(normalizedDistribution);
				}
				return normalizedDistribution;
			} else {
				return dist;
			}
		}

		/**
		 * Outputs one node for graph.
		 * 
		 * @param text the buffer to append the output to
		 * @param num  unique node id
		 * @return the next node id
		 * @throws Exception if generation fails
		 */
		public int toGraph(StringBuffer text, int num) throws Exception {

			int maxIndex = Utils.maxIndex(modelProb);
			String classValue = treeInfo.classAttribute().isNominal() ? treeInfo.classAttribute().value(maxIndex)
					: Utils.doubleToString(modelProb[0], 2);

			num++;
			if (features == -1) {
				text.append("N" + Integer.toHexString(hashCode()) + " [label=\"" + num + ": " + classValue + "\""
						+ "shape=box]\n");
			} else {
				text.append("N" + Integer.toHexString(hashCode()) + " [label=\"" + num + ": " + classValue + "\"]\n");
				for (int i = 0; i < next.length; i++) {
					text.append("N" + Integer.toHexString(hashCode()) + "->" + "N"
							+ Integer.toHexString(next[i].hashCode()) + " [label=\""
							+ treeInfo.attribute(features).name());
					if (treeInfo.attribute(features).isNumeric()) {
						if (i == 0) {
							text.append(" < " + Utils.doubleToString(splitPoint, 2));
						} else {
							text.append(" >= " + Utils.doubleToString(splitPoint, 2));
						}
					} else {
						text.append(" = " + treeInfo.attribute(features).value(i));
					}
					text.append("\"]\n");
					num = next[i].toGraph(text, num);
				}
			}

			return num;
		}

		/**
		 * Outputs a leaf.
		 * 
		 * @return the leaf as string
		 * @throws Exception if generation fails
		 */
		protected String leafString() throws Exception {

			double sum = 0, maxCount = 0;
			int maxIndex = 0;
			double classMean = 0;
			double avgError = 0;
			if (modelProb != null) {
				if (treeInfo.classAttribute().isNominal()) {
					sum = Utils.sum(modelProb);
					maxIndex = Utils.maxIndex(modelProb);
					maxCount = modelProb[maxIndex];
				} else {
					classMean = modelProb[0];
					if (probability[1] > 0) {
						avgError = probability[0] / probability[1];
					}
				}
			}

			if (treeInfo.classAttribute().isNumeric()) {
				return " : " + Utils.doubleToString(classMean, 2) + " (" + Utils.doubleToString(probability[1], 2)
						+ "/" + Utils.doubleToString(avgError, 2) + ")";
			}

			return " : " + treeInfo.classAttribute().value(maxIndex) + " (" + Utils.doubleToString(sum, 2) + "/"
					+ Utils.doubleToString(sum - maxCount, 2) + ")";
		}

		/**
		 * Recursively outputs the CustomTree.
		 * 
		 * @param level the current level of the CustomTree
		 * @return the generated subCustomTree
		 */
		protected String toString(int level) {

			try {
				StringBuffer text = new StringBuffer();

				if (features == -1) {

					// Output leaf info
					return leafString();
				} else if (treeInfo.attribute(features).isNominal()) {

					// For nominal attributes
					for (int i = 0; i < next.length; i++) {
						text.append("\n");
						for (int j = 0; j < level; j++) {
							text.append("|   ");
						}
						text.append(treeInfo.attribute(features).name() + " = "
								+ treeInfo.attribute(features).value(i));
						text.append(next[i].toString(level + 1));
					}
				} else {

					// For numeric attributes
					text.append("\n");
					for (int j = 0; j < level; j++) {
						text.append("|   ");
					}
					text.append(treeInfo.attribute(features).name() + " < " + Utils.doubleToString(splitPoint, 2));
					text.append(next[0].toString(level + 1));
					text.append("\n");
					for (int j = 0; j < level; j++) {
						text.append("|   ");
					}
					text.append(
							treeInfo.attribute(features).name() + " >= " + Utils.doubleToString(splitPoint, 2));
					text.append(next[1].toString(level + 1));
				}

				return text.toString();
			} catch (Exception e) {
				e.printStackTrace();
				return "CustomRandomTree: CustomTree can't be printed";
			}
		}

		/**
		 * Recursively backfits data into the CustomTree.
		 * 
		 * @param data       the data to work with
		 * @param classProbs the class distribution
		 * @throws Exception if generation fails
		 */
		protected void backfitData(Instances data, double[] classProbs, double totalWeight) throws Exception {

			// Make leaf if there are no training instances
			if (data.numInstances() == 0) {
				features = -1;
				modelProb = null;
				if (data.classAttribute().isNumeric()) {
					probability = new double[2];
				}
				proportion = null;
				return;
			}

			double priorVar = 0;
			if (data.classAttribute().isNumeric()) {

				// Compute prior variance
				double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					totalSum += inst.classValue() * inst.weight();
					totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
					totalSumOfWeights += inst.weight();
				}
				priorVar = CustomRandomTree2.oneVariance(totalSum, totalSumSquared, totalSumOfWeights);
			}

			// Check if node doesn't contain enough instances or is pure
			// or maximum depth reached
			modelProb = classProbs.clone();

			/*
			 * if (Utils.sum(m_ClassDistribution) < 2 * m_MinNum ||
			 * Utils.eq(m_ClassDistribution[Utils.maxIndex(m_ClassDistribution)], Utils
			 * .sum(m_ClassDistribution))) {
			 * 
			 * // Make leaf m_Attribute = -1; m_Prop = null; return; }
			 */

			// Are we at an inner node
			if (features > -1) {

				// Compute new weights for subsets based on backfit data
				proportion = new double[next.length];
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (!inst.isMissing(features)) {
						if (data.attribute(features).isNominal()) {
							proportion[(int) inst.value(features)] += inst.weight();
						} else {
							proportion[(inst.value(features) < splitPoint) ? 0 : 1] += inst.weight();
						}
					}
				}

				// If we only have missing values we can make this node into a leaf
				if (Utils.sum(proportion) <= 0) {
					features = -1;
					proportion = null;

					if (data.classAttribute().isNumeric()) {
						probability = new double[2];
						probability[0] = priorVar;
						probability[1] = totalWeight;
					}

					return;
				}

				// Otherwise normalize the proportions
				Utils.normalize(proportion);

				// Split data
				Instances[] subsets = splitData(data);

				// Go through subsets
				for (int i = 0; i < subsets.length; i++) {

					// Compute distribution for current subset
					double[] dist = new double[data.numClasses()];
					double sumOfWeights = 0;
					for (int j = 0; j < subsets[i].numInstances(); j++) {
						if (data.classAttribute().isNominal()) {
							dist[(int) subsets[i].instance(j).classValue()] += subsets[i].instance(j).weight();
						} else {
							dist[0] += subsets[i].instance(j).classValue() * subsets[i].instance(j).weight();
							sumOfWeights += subsets[i].instance(j).weight();
						}
					}

					if (sumOfWeights > 0) {
						dist[0] /= sumOfWeights;
					}

					// Backfit subset
					next[i].backfitData(subsets[i], dist, totalWeight);
				}

				// If unclassified instances are allowed, we don't need to store the
				// class distribution
				if (isAllowUnclassifiedInstances()) {
					modelProb = null;
					return;
				}

				for (int i = 0; i < subsets.length; i++) {
					if (next[i].modelProb == null) {
						return;
					}
				}
				modelProb = null;

				// If we have a least two non-empty successors, we should keep this CustomTree
				/*
				 * int nonEmptySuccessors = 0; for (int i = 0; i < subsets.length; i++) { if
				 * (m_Successors[i].m_ClassDistribution != null) { nonEmptySuccessors++; if
				 * (nonEmptySuccessors > 1) { return; } } }
				 * 
				 * // Otherwise, this node is a leaf or should become a leaf m_Successors =
				 * null; m_Attribute = -1; m_Prop = null; return;
				 */
			}
		}

		/**
		 * Recursively generates a CustomTree.
		 * 
		 * @param data             the data to work with
		 * @param classProbs       the class distribution
		 * @param attIndicesWindow the attribute window to choose attributes from
		 * @param random           random number generator for choosing random
		 *                         attributes
		 * @param depth            the current depth
		 * @throws Exception if generation fails
		 */
		protected void buildCustomTree(Instances data, double[] classProbs, int[] attIndicesWindow, double totalWeight,
				Random random, int depth, double minVariance) throws Exception {

			// Make leaf if there are no training instances
			if (data.numInstances() == 0) {
				features = -1;
				modelProb = null;
				proportion = null;

				if (data.classAttribute().isNumeric()) {
					probability = new double[2];
				}
				return;
			}

			double priorVar = 0;
			if (data.classAttribute().isNumeric()) {

				// Compute prior variance
				double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					totalSum += inst.classValue() * inst.weight();
					totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
					totalSumOfWeights += inst.weight();
				}
				priorVar = CustomRandomTree2.oneVariance(totalSum, totalSumSquared, totalSumOfWeights);
			}

			// Check if node doesn't contain enough instances or is pure
			// or maximum depth reached
			if (data.classAttribute().isNominal()) {
				totalWeight = Utils.sum(classProbs);
			}
			// System.err.println("Total weight " + totalWeight);
			// double sum = Utils.sum(classProbs);
			if (totalWeight < 2 * minNumber ||

			// Nominal case
					(data.classAttribute().isNominal()
							&& Utils.eq(classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs)))

					||

					// Numeric case
					(data.classAttribute().isNumeric() && priorVar / totalWeight < minVariance)

					||

					// check CustomTree depth
					((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {

				// Make leaf
				features = -1;
				modelProb = classProbs.clone();
				if (data.classAttribute().isNumeric()) {
					probability = new double[2];
					probability[0] = priorVar;
					probability[1] = totalWeight;
				}

				proportion = null;
				return;
			}

			// Compute class distributions and value of splitting
			// criterion for each attribute
			double val = -Double.MAX_VALUE;
			double split = -Double.MAX_VALUE;
			double[][] bestDists = null;
			double[] bestProps = null;
			int bestIndex = 0;

			// Handles to get arrays out of distribution method
			double[][] props = new double[1][0];
			double[][][] dists = new double[1][0][0];
			double[][] totalSubsetWeights = new double[data.numAttributes()][0];

			// Investigate K random attributes
			int attIndex = 0;
			int windowSize = attIndicesWindow.length;
			int k = kValue;
			boolean gainFound = false;
			double[] tempNumericVals = new double[data.numAttributes()];
			while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

				int chosenIndex = random.nextInt(windowSize);
				attIndex = attIndicesWindow[chosenIndex];

				// shift chosen attIndex out of window
				attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
				attIndicesWindow[windowSize - 1] = attIndex;
				windowSize--;

				double currSplit = data.classAttribute().isNominal() ? distribution(props, dists, attIndex, data)
						: numericDistribution(props, dists, attIndex, totalSubsetWeights, data, tempNumericVals);

				double currVal = data.classAttribute().isNominal() ? gain(dists[0], priorVal(dists[0]))
						: tempNumericVals[attIndex];

				if (Utils.gr(currVal, 0)) {
					gainFound = true;
				}

				if ((currVal > val) || ((!isBreakTiesRandomly()) && (currVal == val) && (attIndex < bestIndex))) {
					val = currVal;
					bestIndex = attIndex;
					split = currSplit;
					bestProps = props[0];
					bestDists = dists[0];
				}
			}

			// Find best attribute
			features = bestIndex;

			// Any useful split found?
			if (Utils.gr(val, 0)) {

				// Build subCustomTrees
				splitPoint = split;
				proportion = bestProps;
				Instances[] subsets = splitData(data);
				next = new CustomTree[bestDists.length];
				double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];

				for (int i = 0; i < bestDists.length; i++) {
					next[i] = new CustomTree();
					next[i].buildCustomTree(subsets[i], bestDists[i], attIndicesWindow,
							data.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i], random, depth + 1,
							minVariance);
				}

				// If all successors are non-empty, we don't need to store the class
				// distribution
				boolean emptySuccessor = false;
				for (int i = 0; i < subsets.length; i++) {
					if (next[i].modelProb == null) {
						emptySuccessor = true;
						break;
					}
				}
				if (emptySuccessor) {
					modelProb = classProbs.clone();
				}
			} else {

				// Make leaf
				features = -1;
				modelProb = classProbs.clone();
				if (data.classAttribute().isNumeric()) {
					probability = new double[2];
					probability[0] = priorVar;
					probability[1] = totalWeight;
				}
			}
		}

		/**
		 * Computes size of the CustomTree.
		 * 
		 * @return the number of nodes
		 */
		public int numNodes() {

			if (features == -1) {
				return 1;
			} else {
				int size = 1;
				for (CustomTree m_Successor : next) {
					size += m_Successor.numNodes();
				}
				return size;
			}
		}

		/**
		 * Splits instances into subsets based on the given split.
		 * 
		 * @param data the data to work with
		 * @return the subsets of instances
		 * @throws Exception if something goes wrong
		 */
		protected Instances[] splitData(Instances data) throws Exception {

			// Allocate array of Instances objects
			Instances[] subsets = new Instances[proportion.length];
			for (int i = 0; i < proportion.length; i++) {
				subsets[i] = new Instances(data, data.numInstances());
			}

			// Go through the data
			for (int i = 0; i < data.numInstances(); i++) {

				// Get instance
				Instance inst = data.instance(i);

				// Does the instance have a missing value?
				if (inst.isMissing(features)) {

					// Split instance up
					for (int k = 0; k < proportion.length; k++) {
						if (proportion[k] > 0) {
							Instance copy = (Instance) inst.copy();
							copy.setWeight(proportion[k] * inst.weight());
							subsets[k].add(copy);
						}
					}

					// Proceed to next instance
					continue;
				}

				// Do we have a nominal attribute?
				if (data.attribute(features).isNominal()) {
					subsets[(int) inst.value(features)].add(inst);

					// Proceed to next instance
					continue;
				}

				// Do we have a numeric attribute?
				if (data.attribute(features).isNumeric()) {
					subsets[(inst.value(features) < splitPoint) ? 0 : 1].add(inst);

					// Proceed to next instance
					continue;
				}

				// Else throw an exception
				throw new IllegalArgumentException("Unknown attribute type");
			}

			// Save memory
			for (int i = 0; i < proportion.length; i++) {
				subsets[i].compactify();
			}

			// Return the subsets
			return subsets;
		}

		/**
		 * Computes numeric class distribution for an attribute
		 * 
		 * @param props
		 * @param dists
		 * @param att
		 * @param subsetWeights
		 * @param data
		 * @param vals
		 * @return
		 * @throws Exception if a problem occurs
		 */
		protected double numericDistribution(double[][] props, double[][][] dists, int att, double[][] subsetWeights,
				Instances data, double[] vals) throws Exception {

			double splitPoint = Double.NaN;
			Attribute attribute = data.attribute(att);
			double[][] dist = null;
			double[] sums = null;
			double[] sumSquared = null;
			double[] sumOfWeights = null;
			double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
			int indexOfFirstMissingValue = data.numInstances();

			if (attribute.isNominal()) {
				sums = new double[attribute.numValues()];
				sumSquared = new double[attribute.numValues()];
				sumOfWeights = new double[attribute.numValues()];
				int attVal;

				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(att)) {

						// Skip missing values at this stage
						if (indexOfFirstMissingValue == data.numInstances()) {
							indexOfFirstMissingValue = i;
						}
						continue;
					}

					attVal = (int) inst.value(att);
					sums[attVal] += inst.classValue() * inst.weight();
					sumSquared[attVal] += inst.classValue() * inst.classValue() * inst.weight();
					sumOfWeights[attVal] += inst.weight();
				}

				totalSum = Utils.sum(sums);
				totalSumSquared = Utils.sum(sumSquared);
				totalSumOfWeights = Utils.sum(sumOfWeights);
			} else {
				// For numeric attributes
				sums = new double[2];
				sumSquared = new double[2];
				sumOfWeights = new double[2];
				double[] currSums = new double[2];
				double[] currSumSquared = new double[2];
				double[] currSumOfWeights = new double[2];

				// Sort data
				data.sort(att);

				// Move all instances into second subset
				for (int j = 0; j < data.numInstances(); j++) {
					Instance inst = data.instance(j);
					if (inst.isMissing(att)) {

						// Can stop as soon as we hit a missing value
						indexOfFirstMissingValue = j;
						break;
					}

					currSums[1] += inst.classValue() * inst.weight();
					currSumSquared[1] += inst.classValue() * inst.classValue() * inst.weight();
					currSumOfWeights[1] += inst.weight();
				}

				totalSum = currSums[1];
				totalSumSquared = currSumSquared[1];
				totalSumOfWeights = currSumOfWeights[1];

				sums[1] = currSums[1];
				sumSquared[1] = currSumSquared[1];
				sumOfWeights[1] = currSumOfWeights[1];

				// Try all possible split points
				double currSplit = data.instance(0).value(att);
				double currVal, bestVal = Double.MAX_VALUE;

				for (int i = 0; i < indexOfFirstMissingValue; i++) {
					Instance inst = data.instance(i);

					if (inst.value(att) > currSplit) {
						currVal = CustomRandomTree2.variance(currSums, currSumSquared, currSumOfWeights);
						if (currVal < bestVal) {
							bestVal = currVal;
							splitPoint = (inst.value(att) + currSplit) / 2.0;

							// Check for numeric precision problems
							if (splitPoint <= currSplit) {
								splitPoint = inst.value(att);
							}

							for (int j = 0; j < 2; j++) {
								sums[j] = currSums[j];
								sumSquared[j] = currSumSquared[j];
								sumOfWeights[j] = currSumOfWeights[j];
							}
						}
					}

					currSplit = inst.value(att);

					double classVal = inst.classValue() * inst.weight();
					double classValSquared = inst.classValue() * classVal;

					currSums[0] += classVal;
					currSumSquared[0] += classValSquared;
					currSumOfWeights[0] += inst.weight();

					currSums[1] -= classVal;
					currSumSquared[1] -= classValSquared;
					currSumOfWeights[1] -= inst.weight();
				}
			}

			// Compute weights
			props[0] = new double[sums.length];
			for (int k = 0; k < props[0].length; k++) {
				props[0][k] = sumOfWeights[k];
			}
			if (!(Utils.sum(props[0]) > 0)) {
				for (int k = 0; k < props[0].length; k++) {
					props[0][k] = 1.0 / props[0].length;
				}
			} else {
				Utils.normalize(props[0]);
			}

			// Distribute weights for instances with missing values
			for (int i = indexOfFirstMissingValue; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);

				for (int j = 0; j < sums.length; j++) {
					sums[j] += props[0][j] * inst.classValue() * inst.weight();
					sumSquared[j] += props[0][j] * inst.classValue() * inst.classValue() * inst.weight();
					sumOfWeights[j] += props[0][j] * inst.weight();
				}
				totalSum += inst.classValue() * inst.weight();
				totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
				totalSumOfWeights += inst.weight();
			}

			// Compute final distribution
			dist = new double[sums.length][data.numClasses()];
			for (int j = 0; j < sums.length; j++) {
				if (sumOfWeights[j] > 0) {
					dist[j][0] = sums[j] / sumOfWeights[j];
				} else {
					dist[j][0] = totalSum / totalSumOfWeights;
				}
			}

			// Compute variance gain
			double priorVar = oneVariance(totalSum, totalSumSquared, totalSumOfWeights);
			double var = variance(sums, sumSquared, sumOfWeights);
			double gain = priorVar - var;

			// Return distribution and split point
			subsetWeights[att] = sumOfWeights;
			dists[0] = dist;
			vals[att] = gain;

			return splitPoint;
		}

		/**
		 * Computes class distribution for an attribute.
		 * 
		 * @param props
		 * @param dists
		 * @param att   the attribute index
		 * @param data  the data to work with
		 * @throws Exception if something goes wrong
		 */
		protected double distribution(double[][] props, double[][][] dists, int att, Instances data) throws Exception {

			double splitPoint = Double.NaN;
			Attribute attribute = data.attribute(att);
			double[][] dist = null;
			int indexOfFirstMissingValue = data.numInstances();

			if (attribute.isNominal()) {

				// For nominal attributes
				dist = new double[attribute.numValues()][data.numClasses()];
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(att)) {

						// Skip missing values at this stage
						if (indexOfFirstMissingValue == data.numInstances()) {
							indexOfFirstMissingValue = i;
						}
						continue;
					}
					dist[(int) inst.value(att)][(int) inst.classValue()] += inst.weight();
				}
			} else {

				// For numeric attributes
				double[][] currDist = new double[2][data.numClasses()];
				dist = new double[2][data.numClasses()];

				// Sort data
				data.sort(att);

				// Move all instances into second subset
				for (int j = 0; j < data.numInstances(); j++) {
					Instance inst = data.instance(j);
					if (inst.isMissing(att)) {

						// Can stop as soon as we hit a missing value
						indexOfFirstMissingValue = j;
						break;
					}
					currDist[1][(int) inst.classValue()] += inst.weight();
				}

				// Value before splitting
				double priorVal = priorVal(currDist);

				// Save initial distribution
				for (int j = 0; j < currDist.length; j++) {
					System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].length);
				}

				// Try all possible split points
				double currSplit = data.instance(0).value(att);
				double currVal, bestVal = -Double.MAX_VALUE;
				for (int i = 0; i < indexOfFirstMissingValue; i++) {
					Instance inst = data.instance(i);
					double attVal = inst.value(att);

					// Can we place a sensible split point here?
					if (attVal > currSplit) {

						// Compute gain for split point
						currVal = gain(currDist, priorVal);

						// Is the current split point the best point so far?
						if (currVal > bestVal) {

							// Store value of current point
							bestVal = currVal;

							// Save split point
							splitPoint = (attVal + currSplit) / 2.0;

							// Check for numeric precision problems
							if (splitPoint <= currSplit) {
								splitPoint = attVal;
							}

							// Save distribution
							for (int j = 0; j < currDist.length; j++) {
								System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].length);
							}
						}

						// Update value
						currSplit = attVal;
					}

					// Shift over the weight
					int classVal = (int) inst.classValue();
					currDist[0][classVal] += inst.weight();
					currDist[1][classVal] -= inst.weight();
				}
			}

			// Compute weights for subsets
			props[0] = new double[dist.length];
			for (int k = 0; k < props[0].length; k++) {
				props[0][k] = Utils.sum(dist[k]);
			}
			if (Utils.eq(Utils.sum(props[0]), 0)) {
				for (int k = 0; k < props[0].length; k++) {
					props[0][k] = 1.0 / props[0].length;
				}
			} else {
				Utils.normalize(props[0]);
			}

			// Distribute weights for instances with missing values
			for (int i = indexOfFirstMissingValue; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				if (attribute.isNominal()) {

					// Need to check if attribute value is missing
					if (inst.isMissing(att)) {
						for (int j = 0; j < dist.length; j++) {
							dist[j][(int) inst.classValue()] += props[0][j] * inst.weight();
						}
					}
				} else {

					// Can be sure that value is missing, so no test required
					for (int j = 0; j < dist.length; j++) {
						dist[j][(int) inst.classValue()] += props[0][j] * inst.weight();
					}
				}
			}

			// Return distribution and split point
			dists[0] = dist;
			return splitPoint;
		}

		/**
		 * Computes value of splitting criterion before split.
		 * 
		 * @param dist the distributions
		 * @return the splitting criterion
		 */
		protected double priorVal(double[][] dist) {

			return ContingencyTables.entropyOverColumns(dist);
		}

		/**
		 * Computes value of splitting criterion after split.
		 * 
		 * @param dist     the distributions
		 * @param priorVal the splitting criterion
		 * @return the gain after the split
		 */
		protected double gain(double[][] dist, double priorVal) {

			return priorVal - ContingencyTables.entropyConditionedOnRows(dist);
		}

		/**
		 * Returns the revision string.
		 * 
		 * @return the revision
		 */
		public String getRevision() {
			return RevisionUtils.extract("$Revision: 12505 $");
		}

		/**
		 * Outputs one node for graph.
		 * 
		 * @param text   the buffer to append the output to
		 * @param num    the current node id
		 * @param parent the parent of the nodes
		 * @return the next node id
		 * @throws Exception if something goes wrong
		 */
		protected int toGraph(StringBuffer text, int num, CustomTree parent) throws Exception {

			num++;
			if (features == -1) {
				text.append("N" + Integer.toHexString(CustomTree.this.hashCode()) + " [label=\"" + num
						+ Utils.backQuoteChars(leafString()) + "\"" + " shape=box]\n");

			} else {
				text.append("N" + Integer.toHexString(CustomTree.this.hashCode()) + " [label=\"" + num + ": "
						+ Utils.backQuoteChars(treeInfo.attribute(features).name()) + "\"]\n");
				for (int i = 0; i < next.length; i++) {
					text.append("N" + Integer.toHexString(CustomTree.this.hashCode()) + "->" + "N"
							+ Integer.toHexString(next[i].hashCode()) + " [label=\"");
					if (treeInfo.attribute(features).isNumeric()) {
						if (i == 0) {
							text.append(" < " + Utils.doubleToString(splitPoint, 2));
						} else {
							text.append(" >= " + Utils.doubleToString(splitPoint, 2));
						}
					} else {
						text.append(" = " + Utils.backQuoteChars(treeInfo.attribute(features).value(i)));
					}
					text.append("\"]\n");
					num = next[i].toGraph(text, num, this);
				}
			}

			return num;
		}
	}

}
