import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;


public class GetModel {
	public static void main(String[] args) throws Exception {
		if(args.length != 4){
			System.out.println("Usage: java -jar GetModel.jar <input>train.arff <input>dev.arff <output>paramSet.txt");
			return;
		}

		String trainPath = args[0]; // input train fitxategia
		String devPath = args[1]; // input dev fitxategia
		String paramPath = args[2]; // output param fitxategia

		GetModel.getModel(trainPath, devPath, paramPath);
	}
	public static void getModel(String trainPath, String devPath, String paramPath) throws Exception {
		
		ConverterUtils.DataSource ds = new ConverterUtils.DataSource(trainPath);
		ConverterUtils.DataSource testDs = new ConverterUtils.DataSource(devPath);

		Instances test = testDs.getDataSet();
		Instances data = ds.getDataSet();
		data.setClassIndex(0);
		test.setClassIndex(0);

		int minClassIndex = Utils.minIndex(data.attributeStats(0).nominalCounts);

		Kernel[] kernels = new Kernel[] {new PolyKernel(), new NormalizedPolyKernel()};

		Kernel bestKernel = kernels[0];
		double bestExponent = Double.MIN_VALUE;
		double bestScore = Double.MIN_VALUE;

		for (Kernel kernel : kernels) {
			for (int i = 0; i < 100; i= i + 2) {
				double exponent = 1.0 + ((double) i / 100);
				double result = evaluateModel(data, test, kernel, exponent, minClassIndex);

				System.out.printf("Kernel: %s, exponent: %f, result: %f\n", kernel.getClass().getSimpleName(), exponent,result);
				if (result > bestScore) {
					bestScore = result;
					bestKernel = kernel;
					bestExponent = exponent;
				}
			}
		}

		System.out.printf("Best kernel: %s, best exponent: %f, achieved min class f-measure: %f", bestKernel.getClass().getSimpleName(), bestExponent, bestScore);


		try (FileWriter fw = new FileWriter(paramPath)){
			fw.write(String.format("%s -E %f",
					bestKernel.getClass().getCanonicalName(), bestExponent
			));
		}
	}


	private static double evaluateModel(Instances train, Instances test, Kernel kernel, double exponent, int minClassIndex) throws Exception {
		SMO cls = new SMO();

		kernel.setOptions(new String[]{"-E", Double.toString(exponent)});
		cls.setKernel(kernel);

		cls.buildClassifier(train);


		Evaluation e = new Evaluation(train);
		e.evaluateModel(cls, test);


		return e.fMeasure(minClassIndex);
	}


}
