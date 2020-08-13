package EnsembleMethod.RandomForest.simple;

import java.io.Serializable;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class CustomRandomTree extends AbstractClassifier implements WeightedInstancesHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1971232304431176379L;

	protected CustomTree CustTree = null;

	protected Instances treeInfo = null;

	// Number nodes at each leaf
	protected double minNumber = 1.0;

	protected int numFolds = 0;

	protected Classifier basicModel;

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
			basicModel = new weka.classifiers.rules.ZeroR();
			basicModel.buildClassifier(inst);
			return;
		} else {
			basicModel = null;
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
			trainVariance = CustomRandomTree.oneVariance(classProbs[0], sumSquaredTotal, sumOfWeight) / sumOfWeight;
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
		
		/**
		 * Recursively generates a CustomTree.
		 * 
		 * @param inst             the data to work with
		 * @param classProbs       the class distribution
		 * @param attIndicesWindow the attribute window to choose attributes from
		 * @param random           random number generator for choosing random
		 *                         attributes
		 * @param depth            the current depth
		 * @throws Exception if generation fails
		 */
		protected void buildCustomTree(Instances inst, double[] classProbs, int[] attIndicesWindow, double weightTotal,
				Random random, int depth, double minVariance) throws Exception {

			//Assign some default values
			if (inst.numInstances() == 0) {
				features = -1;
				modelProb = null;
				proportion = null;

				if (inst.classAttribute().isNumeric()) {
					probability = new double[2];
				}
				return;
			}

			double formerVar = 0;
			if (inst.classAttribute().isNumeric()) {
				double sumTotal = 0;
				double sumSquaredTotal = 0;
				double sumOfWeightsAll = 0;
				int totalInst = inst.numInstances(), i=0;
				while( i < totalInst) {
					Instance data = inst.instance(i);
					sumOfWeightsAll += data.weight();
					sumTotal += data.classValue() * data.weight();
					sumSquaredTotal += data.classValue() * data.classValue() * data.weight();
					i++;
				}
				formerVar = CustomRandomTree.oneVariance(sumTotal, sumSquaredTotal, sumOfWeightsAll);
			}

			if (inst.classAttribute().isNominal()) {
				weightTotal = Utils.sum(classProbs);
			}
			if ((inst.classAttribute().isNominal()
					&& Utils.eq(classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs)) || weightTotal < 2 * minNumber)
					||(inst.classAttribute().isNumeric() && formerVar / weightTotal < minVariance) || ((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {
				features = -1;
				modelProb = classProbs.clone();
				if (inst.classAttribute().isNumeric()) {
					probability = new double[2];
					probability[0] = formerVar;
					probability[1] = weightTotal;
				}

				proportion = null;
				return;
			}
			double v = -Double.MAX_VALUE;
			double s = -Double.MAX_VALUE;
			double[][] topFreq = null;
			double[] topProp = null;
			int topInd = 0;

			double[][] p = new double[1][0];
			double[][][] d = new double[1][0][0];
			double[][] subsetWeights = new double[inst.numAttributes()][0];

			int featureIndx = 0;
			int winSize = attIndicesWindow.length;
			int k = kValue;
			boolean secureIdef = false;
			double[] temporaryNumValues = new double[inst.numAttributes()];
			while ((winSize > 0) && (k-- > 0 || !secureIdef)) {

				int i = random.nextInt(winSize);
				featureIndx = attIndicesWindow[i];
				attIndicesWindow[i] = attIndicesWindow[winSize - 1];
				attIndicesWindow[winSize - 1] = featureIndx;
				winSize--;

				double currentSplit = 0;
				if(inst.classAttribute().isNominal())
					currentSplit = distribution(inst, featureIndx, p, d);
					else 
						currentSplit = distributionOfNumeric(inst, featureIndx, p, d,  subsetWeights,temporaryNumValues);

				double currentValue = 0;
						
						if(inst.classAttribute().isNominal())
							currentValue =  totalGain(d[0], priorValue(d[0]));
						else 
							currentValue = temporaryNumValues[featureIndx];

				if (Utils.gr(currentValue, 0)) {
					secureIdef = true;
				}

				if (((!isBreakTiesRandomly()) && (currentValue == v) && (featureIndx < topInd)) || (currentValue > v)) {
					v = currentValue;
					topInd = featureIndx;
					s = currentSplit;
					topProp = p[0];
					topFreq = d[0];
				}
			}

			features = topInd;

			if (Utils.gr(v, 0)) {
				splitPoint = s;
				proportion = topProp;
				Instances[] subsets = divideInstance(inst);
				next = new CustomTree[topFreq.length];
				double[] feaSubsetWeightsTotal = subsetWeights[topInd];

				for (int i = 0; i < topFreq.length; i++) {
					next[i] = new CustomTree();
					
					if(inst.classAttribute().isNominal())
					next[i].buildCustomTree(subsets[i], topFreq[i], attIndicesWindow,0,random, depth + 1, minVariance);
					else
						next[i].buildCustomTree(subsets[i], topFreq[i], attIndicesWindow, feaSubsetWeightsTotal[i], random, depth + 1,
							minVariance);
				}
				boolean emptyNext = false;
				for (int i = 0; i < subsets.length; i++) {
					if (next[i].modelProb == null) {
						emptyNext = true;
						break;
					}
				}
				if (emptyNext) {
					modelProb = classProbs.clone();
				}
			} else {
				features = -1;
				modelProb = classProbs.clone();
				if (inst.classAttribute().isNumeric()) {
					probability = new double[2];
					probability[0] = formerVar;
					probability[1] = weightTotal;
				}
			}
		}

		/**
		 * Calculate the total size of tree
		 * 
		 * @return total nodes
		 */
		public int totalChild() {

			if (features == -1) {
				return 1;
			} else {
				int s = 1;
				for (CustomTree n : next) {
					s += n.totalChild();
				}
				return s;
			}
		}

		/**
		 * This Method divide instance into subsets 
		 * 
		 * @param instance 
		 * @return child nodes
		 * @throws Exception
		 */
		protected Instances[] divideInstance(Instances inst) throws Exception {

			Instances[] childNodes = new Instances[proportion.length];
			for (int i = 0; i < proportion.length; i++) {
				childNodes[i] = new Instances(inst, inst.numInstances());
			}

			for (int i = 0; i < inst.numInstances(); i++) {
				Instance data = inst.instance(i);
				if (data.isMissing(features)) {
					for (int k = 0; k < proportion.length; k++) {
						if (proportion[k] > 0) {
							Instance duplicate = (Instance) data.copy();
							duplicate.setWeight(proportion[k] * data.weight());
							childNodes[k].add(duplicate);
						}
					}
					continue;
				}

				if (data.attribute(features).isNominal()) {
					childNodes[(int) data.value(features)].add(data);
					continue;
				}
				if (data.attribute(features).isNumeric()) {
					if(data.value(features) < splitPoint)
						childNodes[0].add(data);
					else
						childNodes[1].add(data);
					continue;
				}
				throw new IllegalArgumentException("Undefined feature type");
			}

			for (int i = 0; i < proportion.length; i++) {
				childNodes[i].compactify();
			}

			return childNodes;
		}

		/**
		 * Computes numeric class distribution for an attribute
		 * 
		 * @param p
		 * @param d
		 * @param f
		 * @param childNodeWeights
		 * @param inst
		 * @param values
		 * @return
		 * @throws Exception if a problem occurs
		 */
		protected double distributionOfNumeric(Instances inst, int f, double[][] p, double[][][] d, double[][] childNodeWeights,
				 double[] values) throws Exception {

			double sperate = Double.NaN;
			Attribute att = inst.attribute(f);
			double[][] dist = null;
			double[] totalSums = null;
			double[] sS = null;
			double[] sW = null;
			double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
			int indexOfFirstMissingValue = inst.numInstances();

			if (!att.isNominal()) {
				// Execute  numeric features
				totalSums = new double[2];
				sS = new double[2];
				sW = new double[2];
				double[] currentS = new double[2];
				double[] currentSS = new double[2];
				double[] currentSOW = new double[2];

				inst.sort(f);

				for (int j = 0; j < inst.numInstances(); j++) {
					Instance data = inst.instance(j);
					if (data.isMissing(f)) {
						indexOfFirstMissingValue = j;
						break;
					}

					currentS[1] += data.classValue() * data.weight();
					currentSS[1] += data.classValue() * data.classValue() * data.weight();
					currentSOW[1] += data.weight();
				}

				totalSum = currentS[1];
				totalSumSquared = currentSS[1];
				totalSumOfWeights = currentSOW[1];

				totalSums[1] = currentS[1];
				sS[1] = currentSS[1];
				sW[1] = currentSOW[1];

				double currSplit = inst.instance(0).value(f);
				double currVal, bestVal = Double.MAX_VALUE;

				for (int i = 0; i < indexOfFirstMissingValue; i++) {
					Instance data = inst.instance(i);

					if (data.value(f) > currSplit) {
						currVal = CustomRandomTree.variance(currentS, currentSS, currentSOW);
						if (currVal < bestVal) {
							bestVal = currVal;
							sperate = (data.value(f) + currSplit) / 2.0;

							// validate for numeric decimal problems 
							if (sperate <= currSplit) {
								sperate = data.value(f);
							}

							for (int j = 0; j < 2; j++) {
								totalSums[j] = currentS[j];
								sS[j] = currentSS[j];
								sW[j] = currentSOW[j];
							}
						}
					}

					currSplit = data.value(f);

					double classVal = data.classValue() * data.weight();
					double classValSquared = data.classValue() * classVal;

					currentS[0] += classVal;
					currentSS[0] += classValSquared;
					currentSOW[0] += data.weight();

					currentS[1] -= classVal;
					currentSS[1] -= classValSquared;
					currentSOW[1] -= data.weight();
				}
				
			} else {
				totalSums = new double[att.numValues()];
				sS = new double[att.numValues()];
				sW = new double[att.numValues()];
				int attVal;

				for (int i = 0; i < inst.numInstances(); i++) {
					Instance data = inst.instance(i);
					if (data.isMissing(f)) {
						if (indexOfFirstMissingValue == inst.numInstances()) {
							indexOfFirstMissingValue = i;
						}
						continue;
					}

					attVal = (int) data.value(f);
					totalSums[attVal] += data.classValue() * data.weight();
					sS[attVal] += data.classValue() * data.classValue() * data.weight();
					sW[attVal] += data.weight();
				}

				totalSum = Utils.sum(totalSums);
				totalSumSquared = Utils.sum(sS);
				totalSumOfWeights = Utils.sum(sW);
			}

			// Calculate the weights
			p[0] = new double[totalSums.length];
			for (int k = 0; k < p[0].length; k++) {
				p[0][k] = sW[k];
			}
			if (!(Utils.sum(p[0]) > 0)) {
				for (int k = 0; k < p[0].length; k++) {
					p[0][k] = 1.0 / p[0].length;
				}
			} else {
				Utils.normalize(p[0]);
			}

			// Assign weights for the missing instances
			for (int i = indexOfFirstMissingValue; i < inst.numInstances(); i++) {
				Instance data = inst.instance(i);

				for (int j = 0; j < totalSums.length; j++) {
					sW[j] += p[0][j] * data.weight();
					totalSums[j] += p[0][j] * data.classValue() * data.weight();
					sS[j] += p[0][j] * data.classValue() * data.classValue() * data.weight();
				}
				totalSumOfWeights += data.weight();
				totalSum += data.classValue() * data.weight();
				totalSumSquared += data.classValue() * data.classValue() * data.weight();
			}

			// Compute final distribution
			dist = new double[totalSums.length][inst.numClasses()];
			for (int j = 0; j < totalSums.length; j++) {
				if (sW[j] > 0) {
					dist[j][0] = totalSums[j] / sW[j];
				} else {
					dist[j][0] = totalSum / totalSumOfWeights;
				}
			}

			// calculate the gain of the variance 
			double priorVar = oneVariance(totalSum, totalSumSquared, totalSumOfWeights);
			double var = variance(totalSums, sS, sW);
			double gain = priorVar - var;

			
			childNodeWeights[f] = sW;
			d[0] = dist;
			values[f] = gain;

			return sperate;
		}

		/**
		 * Computes class distribution for an attribute.
		 * 
		 * @param p
		 * @param d
		 * @param f   the attribute index
		 * @param inst  the data to work with
		 * @throws Exception if something goes wrong
		 */
		public double distribution(Instances inst, int f, double[][] p, double[][][] d) throws Exception {

			double division = Double.NaN;
			Attribute feature = inst.attribute(f);
			double[][] distance = null;
			int indOfFirstMissVal = inst.numInstances();

			if (!feature.isNominal()) {
				double[][] currentDistribution = new double[2][inst.numClasses()];
				distance = new double[2][inst.numClasses()];
				inst.sort(f);
				for (int j = 0; j < inst.numInstances(); j++) {
					Instance data = inst.instance(j);
					if (data.isMissing(f)) {
						indOfFirstMissVal = j;
						break;
					}
					currentDistribution[1][(int) data.classValue()] += data.weight();
				}

				double priorVal = priorValue(currentDistribution);

				for (int j = 0; j < currentDistribution.length; j++) {
					System.arraycopy(currentDistribution[j], 0, distance[j], 0, distance[j].length);
				}

				double currentdivd = inst.instance(0).value(f);
				double currentValue, topValue = -Double.MAX_VALUE;
				for (int i = 0; i < indOfFirstMissVal; i++) {
					Instance data = inst.instance(i);
					double attVal = data.value(f);
					if (attVal > currentdivd) {

						// calculate gain at each division
						currentValue = totalGain(currentDistribution, priorVal);

						// check for the current value division is the best division until?
						if (currentValue > topValue) {

							// pressure the current value
							topValue = currentValue;

						
							division = (attVal + currentdivd) / 2.0;

							// validate for the numeric decimal problems
							if (division <= currentdivd) {
								division = attVal;
							}

							// pressure the distribution
							for (int j = 0; j < currentDistribution.length; j++) {
								System.arraycopy(currentDistribution[j], 0, distance[j], 0, distance[j].length);
							}
						}

						// change the value
						currentdivd = attVal;
					}

					int classValue = (int) data.classValue();
					currentDistribution[0][classValue] += data.weight();
					currentDistribution[1][classValue] -= data.weight();
				}
				
			} else {
				
				distance = new double[feature.numValues()][inst.numClasses()];
				for (int i = 0; i < inst.numInstances(); i++) {
					Instance data = inst.instance(i);
					if (data.isMissing(f)) {

						// Skip missing values at this stage
						if (indOfFirstMissVal == inst.numInstances()) {
							indOfFirstMissVal = i;
						}
						continue;
					}
					distance[(int) data.value(f)][(int) data.classValue()] += data.weight();
				}
			}

			// Compute weights for subsets
			p[0] = new double[distance.length];
			for (int k = 0; k < p[0].length; k++) {
				p[0][k] = Utils.sum(distance[k]);
			}
			if (Utils.eq(Utils.sum(p[0]), 0)) {
				for (int k = 0; k < p[0].length; k++) {
					p[0][k] = 1.0 / p[0].length;
				}
			} else {
				Utils.normalize(p[0]);
			}

			for (int i = indOfFirstMissVal; i < inst.numInstances(); i++) {
				Instance data = inst.instance(i);
				if (feature.isNominal()) {
					if (data.isMissing(f)) {
						for (int j = 0; j < distance.length; j++) {
							distance[j][(int) data.classValue()] += p[0][j] * data.weight();
						}
					}
				} else {
					for (int j = 0; j < distance.length; j++) {
						distance[j][(int) data.classValue()] += p[0][j] * data.weight();
					}
				}
			}

			d[0] = distance;
			return division;
		}

		/**
		 *Calculate the previous values entropy
		 * 
		 * @param distribution 
		 * @return the splitting criterion
		 */
		protected double priorValue(double[][] distribution) {

			return ContingencyTables.entropyOverColumns(distribution);
		}

		/**
		 * Calculate the total gain each split
		 * 
		 * @param distribution
		 * @param pValue 
		 * @return total gain
		 */
		protected double totalGain(double[][] distribution, double pValue) {

			return pValue - ContingencyTables.entropyConditionedOnRows(distribution);
		}
		
	}

}
