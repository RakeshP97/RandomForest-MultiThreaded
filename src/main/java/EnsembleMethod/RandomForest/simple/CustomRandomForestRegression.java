package EnsembleMethod.RandomForest.simple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import smile.base.cart.CART;
import smile.base.cart.Loss;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.data.vector.BaseVector;
import smile.feature.TreeSHAP;
import smile.math.MathEx;
import smile.regression.DataFrameRegression;
import smile.regression.Regression;
import smile.regression.RegressionTree;

/**
 * Custom Random forest for regression. 
 * 
 */
public class CustomRandomForestRegression implements Regression<Tuple>, DataFrameRegression, TreeSHAP {
    private static final long serialVersionUID = 2L;

    private Formula f;
    private RegressionTree[] customTrees;
    private double e;

    private double[] imp;

    /**
     * Intialize default values.
     * 
     * @param f
     * @param t 
     * @param e 
     * @param imp
     */
    public CustomRandomForestRegression(Formula f, RegressionTree[] t, double e, double[] imp) {
        this.f = f;
        this.customTrees = t;
        this.e = e;
        this.imp = imp;
    }

    /**
     * This method helps to fit the data
     *
     * @param f 
     * @param d 
     * @throws Exception 
     */
    public static CustomRandomForestRegression fit(Formula f, DataFrame d) throws Exception {
        return fit(f, d, new Properties());
    }

    /**
     * helps to gets the default property values  
     *
     * @param f a symbolic description of the model to be fitted.
     * @param d the data frame of the explanatory and response variables.
     * @throws Exception 
     */
    public static CustomRandomForestRegression fit(Formula f, DataFrame d, Properties p) throws Exception {
        int ntrees = Integer.valueOf(p.getProperty("smile.random.forest.trees", "500"));
        int mtry = Integer.valueOf(p.getProperty("smile.random.forest.mtry", "0"));
        int maxDepth = Integer.valueOf(p.getProperty("smile.random.forest.max.depth", "20"));
        int maxNodes = Integer.valueOf(p.getProperty("smile.random.forest.max.nodes", String.valueOf(d.size() / 5)));
        int nodeSize = Integer.valueOf(p.getProperty("smile.random.forest.node.size", "5"));
        double subsample = Double.valueOf(p.getProperty("smile.random.forest.sample.rate", "1.0"));
        return fit(f, d, ntrees, mtry, maxDepth, maxNodes, nodeSize, subsample);
    }

    /**
     * Random forest model to fit the data
     *
     * @param f
     * @param d 
     * @param numberTrees
     * @param m 
     * @param maximumDepth
     * @param maximumNodes
     * @param instnaceSize
     * @param subset
     * @throws Exception 
     */
    public static CustomRandomForestRegression fit(Formula f, DataFrame d, int numberTrees, int m, int maximumDepth, int maximumNodes, int instnaceSize, double subset) throws Exception {
        return fitTree(f, d, numberTrees, m, maximumDepth, maximumNodes, instnaceSize, subset, null);
    }

    /**
     * Learns a random forest for regression.
     *
     * @param f
     * @param d 
     * @param numberTree 
     * @param mt 
     * @param maximumDepth 
     * @param maximumSize 
     * @param instanceSize 
     * @param subset
     * @param s
     * @throws Exception 
     */
    public static CustomRandomForestRegression fitTree(Formula f, DataFrame d, int numberTree, int mt, int maximumDepth, int maximumSize, int instanceSize, double subset, LongStream s) throws Exception {
        if (numberTree < 1) {
            throw new Exception("Incorrect number of trees: " + numberTree);
        }

        if (subset <= 0 || subset > 1) {
            throw new Exception("Incorrect sub smaple rate: " + subset);
        }

        DataFrame x = f.x(d);
        BaseVector response = f.y(d);
        StructField field = response.field();
        double[] y = response.toDoubleArray();

        if (mt > x.ncols()) {
            throw new Exception("Incorrect number of arguments to divided on at a point of the tree: " + mt);
        }

        int mtryFinal = mt > 0 ? mt : Math.max(x.ncols()/3, 1);

        final int n = x.nrows();
        double[] prediction = new double[n];
        int[] oob = new int[n];
        final int[][] order = CART.order(x);

        // generate seeds with sequential stream
        long[] seedArray = (s != null ? s : LongStream.range(-numberTree, 0)).sequential().distinct().limit(numberTree).toArray();
        if (seedArray.length != numberTree) {
            throw new Exception(String.format("Stream of seed values have %d distinct values, that should be %d", seedArray.length, numberTree));
        }

        // Run train data to tree in parallel stream
        
        ArrayList<RegressionTree> list = new ArrayList<RegressionTree>();
        int numumberOfThreads = 10;
		// trying to get the number of processor available
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ExecutorService executorPool = Executors.newFixedThreadPool(numberOfCores * numumberOfThreads);
		
		final CountDownLatch latch = new CountDownLatch(seedArray.length);
		final AtomicInteger atomic = new AtomicInteger();
		long start = System.currentTimeMillis();
        for(long ls: seedArray)
        {
        	
        	Runnable task = new Runnable() {
				@Override
				public void run() {
					try {
					  if (ls > 1) 
						  MathEx.setSeed(ls);

			            final int[] sam = new int[n];
			            if (subset == 1.0) {
			                // draw and train sample instance with replacement from dataset
			                for (int i = 0; i < n; i++) {
			                    sam[MathEx.randomInt(n)]++;
			                }
			            } else {
			                // draw and train sample instance with out replacement from dataset
			                int[] per = MathEx.permutate(n);
			                int N = (int) Math.round(n * subset);
			                for (int i = 0; i < N; i++) {
			                    sam[per[i]] = 1;
			                }
			            }

			            RegressionTree tree = new RegressionTree(x, Loss.ls(y), field, maximumDepth, maximumSize, instanceSize, mtryFinal, sam, order);

			            IntStream.range(0, n).filter(i -> sam[i] == 0).forEach(i -> {
			                double pred = tree.predict(x.get(i));
			                prediction[i] += pred;
			                oob[i]++;
			            });

					list.add(tree);
					System.out.println(Thread.currentThread().getName());
					} catch (Throwable ex) {
						ex.printStackTrace();
						atomic.incrementAndGet();

					} finally {
						latch.countDown();
					}
				}
        	};
			executorPool.submit(task);
          }
        
        long finish = System.currentTimeMillis();
		System.out.println("Total Time take to submit the threads in mill Seconds:" + (finish - start));
		System.out.println("Total number of threads is active:" + Thread.activeCount());
		System.out.println("count down latch value::"+latch);
		latch.await();
		executorPool.shutdownNow();
		// Check for the error based on failures noted
		if (atomic.intValue() > 0) {
			System.err.println(
					"There is error occurs whicle building classifiers - because of that some iterations failed.");
		}
		long finish1 = System.currentTimeMillis();
		System.out.println("Total Time took to complete the Threads in mill Seconds:" + (finish1 - start));
		
		RegressionTree[] trees = (RegressionTree[]) list.toArray();
        int m = 0;
        double error = 0.0;
        for (int i = 0; i < n; i++) {
            if (oob[i] > 0) {
                m++;
                double pred = prediction[i] / oob[i];
                error += MathEx.sqr(pred - y[i]);
            }
        }

        if (m > 0) {
            error = Math.sqrt(error / m);
        }

        double[] importance = calculateImportance(trees);
        return new CustomRandomForestRegression(f, trees, error, importance);
    }

    /**
     * Used to merge from other which come as input
     * 
     * @throws Exception 
     */
    public CustomRandomForestRegression merge(CustomRandomForestRegression other) throws Exception {
        if (!f.equals(other.f)) {
            throw new Exception("Invalid Custom random forest input");
        }

        RegressionTree[] treeArray = new RegressionTree[customTrees.length + other.customTrees.length];
        System.arraycopy(customTrees, 0, treeArray, 0, customTrees.length);
        System.arraycopy(other.customTrees, 0, treeArray, customTrees.length, other.customTrees.length);

        double mError = (this.e * other.e) / 2;
        double[] meImp = imp.clone();
        for (int i = 0; i < imp.length; i++) {
            meImp[i] += other.imp[i];
        }

        return new CustomRandomForestRegression(f, treeArray, mError, meImp);
    }

    /**
     *  Calculate the importance of the all the trees
     *  
     *   */
    private static double[] calculateImportance(RegressionTree[] t) {
        double[] impt = new double[t[0].importance().length];
        for (RegressionTree tree : t) {
            double[] imp = tree.importance();
            for (int i = 0; i < imp.length; i++) {
                impt[i] += imp[i];
            }
        }
        return impt;
    }

    @Override
    public Formula formula() {
        return f;
    }

    @Override
    public StructType schema() {
        return customTrees[0].schema();
    }

    public double error() {
        return e;
    }
        
   
    public double[] importance() {
        return imp;
    }
    
    public int size() {
        return customTrees.length;
    }

    @Override
    public RegressionTree[] trees() {
        return customTrees;
    }

    /**
     * trim the size of the tree
     * 
     * @param numberTrees
     * @throws Exception 
     */
    public void trim(int numberTrees) throws Exception {
        if (numberTrees > customTrees.length) {
            throw new Exception("The new model size is larger than the current size.");
        }
        
        if (numberTrees <= 0) {
            throw new Exception("Invalid new model size: " + numberTrees);
        }
        
        RegressionTree[] model = new RegressionTree[numberTrees];
        System.arraycopy(customTrees, 0, model, 0, numberTrees);
        customTrees = model;
    }
    
    @Override
    public double predict(Tuple x) {
        Tuple xt = f.x(x);
        double y = 0;
        for (RegressionTree tree : customTrees) {
            y += tree.predict(xt);
        }
        
        return y / customTrees.length;
    }

    /**
     * validate the data set
     *
     * @param d 
     * @return predictions
     */
    public double[][] testDataSet(DataFrame d) {
        DataFrame x = f.x(d);

        int n = x.nrows();
        int numberTrees = customTrees.length;
        double[][] pred = new double[numberTrees][n];

        for (int j = 0; j < n; j++) {
            Tuple xj = x.get(j);
            double base = 0;
            for (int i = 0; i < numberTrees; i++) {
                base = base + customTrees[i].predict(xj);
                pred[i][j] = base / (i+1);
            }
        }

        return pred;
    }
}
