package EnsembleMethod.RandomForest.simple;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import com.opencsv.CSVWriter;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class CustomRandomForestImplTest {

	private static String filePath = "C:\\MyData\\Project\\Source_Code\\DataSets\\";

	Instances dataset = null;
	CustomRandomForestImpl custRandomForest;

	/**
	 * Load the data set from arff file input as dataset instance
	 * 
	 * @param filePath
	 * @throws Exception
	 */
	public void loadDataSet(String filePath) throws Exception {
		try {
			DataSource data = new DataSource(filePath);
			dataset = data.getDataSet();
			dataset.setClassIndex(dataset.numAttributes() - 1);
		} catch (Exception e) {
			System.out.println(e.getMessage());
			throw e;
		}

	}

	/**
	 * build a custom random forest model as based model is random tree
	 * 
	 * @throws Exception
	 */

	public void buildCustomRandomForestModel() throws Exception {

		try {
			long start = System.currentTimeMillis();
			custRandomForest = new CustomRandomForestImpl();

			Remove rm = new Remove();
			rm.setAttributeIndices("1");
			FilteredClassifier filteredClassifier = new FilteredClassifier();
			filteredClassifier.setFilter(rm);
			filteredClassifier.setClassifier(custRandomForest);
			filteredClassifier.buildClassifier(dataset);
			long finish = System.currentTimeMillis();
			System.out.println("Time to complete:" + (finish - start));
		} catch (Exception e) {
			System.out.println(e.getMessage());
			throw e;
		}

	}

	public void buildCustomRandomForestRegression() throws Exception {
		
		 long[] seeds = {
		            342317953, 521642753, 72070657, 577451521, 266953217, 179976193,
		            374603777, 527788033, 303395329, 185759582, 261518209, 461300737,
		            483646580, 532528741, 159827201, 284796929, 655932697, 26390017,
		            454330473, 867526205, 824623361, 719082324, 334008833, 699933293,
		            823964929, 155216641, 150210071, 249486337, 713508520, 558398977,
		            886227770, 74062428, 670528514, 701250241, 363339915, 319216345,
		            757017601, 459643789, 170213767, 434634241, 414707201, 153100613,
		            753882113, 546490145, 412517763, 888761089, 628632833, 565587585,
		            175885057, 594903553, 78450978, 212995578, 710952449, 835852289,
		            415422977, 832538705, 624345857, 839826433, 260963602, 386066438,
		            530942946, 261866663, 269735895, 798436064, 379576194, 251582977,
		            349161809, 179653121, 218870401, 415292417, 86861523, 570214657,
		            701581299, 805955890, 358025785, 231452966, 584239408, 297276298,
		            371814913, 159451160, 284126095, 896291329, 496278529, 556314113,
		            31607297, 726761729, 217004033, 390410146, 70173193, 661580775,
		            633589889, 389049037, 112099159, 54041089, 80388281, 492196097,
		            912179201, 699398161, 482080769, 363844609, 286008078, 398098433,
		            339855361, 189583553, 697670495, 709568513, 98494337, 99107427,
		            433350529, 266601473, 888120086, 243906049, 414781441, 154685953,
		            601194298, 292273153, 212413697, 568007473, 666386113, 712261633,
		            802026964, 783034790, 188095005, 742646355, 550352897, 209421313,
		            175672961, 242531185, 157584001, 201363231, 760741889, 852924929,
		            60158977, 774572033, 311159809, 407214966, 804474160, 304456514,
		            54251009, 504009638, 902115329, 870383757, 487243777, 635554282,
		            564918017, 636074753, 870308031, 817515521, 494471884, 562424321,
		            81710593, 476321537, 595107841, 418699893, 315560449, 773617153,
		            163266399, 274201241, 290857537, 879955457, 801949697, 669025793,
		            753107969, 424060977, 661877468, 433391617, 222716929, 334154852,
		            878528257, 253742849, 480885528, 99773953, 913761493, 700407809,
		            483418083, 487870398, 58433153, 608046337, 475342337, 506376199,
		            378726401, 306604033, 724646374, 895195218, 523634541, 766543466,
		            190068097, 718704641, 254519245, 393943681, 796689751, 379497473,
		            50014340, 489234689, 129556481, 178766593, 142540536, 213594113,
		            870440184, 277912577};
		
		CustomRandomForestRegression model = CustomRandomForestRegression.fitTree(CustomLongley.formula, CustomLongley.data, 100, 3, 20, 10, 3, 1.0, Arrays.stream(seeds));
		

        double[] importance = model.importance();
        System.out.println("----- importance -----");
        for (int i = 0; i < importance.length; i++) {
            System.out.format("%-15s %12.4f%n", model.schema().fieldName(i), importance[i]);
        }

	}
	public static void main(String[] args) throws Exception {

		// create file to store the output cpu performance
		//CSVWriter writer = createOutputCsv();
		//Timer timer = new Timer();
		//TimerTask task = new CpuTask(writer);
		//timer.schedule(task, 10, 1000);
		Thread.sleep(1000);
		CustomFileConvertor.convertCsvToArrf(filePath + "ElectionData.csv", filePath + "ElectionData.arff");
		CustomRandomForestImplTest model = new CustomRandomForestImplTest();
		model.loadDataSet(filePath + "ElectionData.arff");

		// building new model Random tree as a base model
		model.buildCustomRandomForestRegression();
		//Thread.sleep(2000);
		//timer.cancel();
		//writer.close();

	}

	/**
	 * This method used create output csv writer stream
	 * 
	 * @return CSVWriter
	 * @throws IOException
	 */

	public static CSVWriter createOutputCsv() throws IOException {
		File file = new File(filePath + "output.csv");
		FileWriter outputfile = new FileWriter(file);
		CSVWriter writer = new CSVWriter(outputfile);
		String[] header = { "No.of.Threads", "JvmCpuLoad", "SystemCpuLoad", "JavaCpuProcessTime" };
		writer.writeNext(header);

		return writer;
	}

}
