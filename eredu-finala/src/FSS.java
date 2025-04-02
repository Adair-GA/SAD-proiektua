import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;
import java.util.Random;

public class FSS {
	public static void main(String[] args) throws Exception {
		ConverterUtils.DataSource ds = new ConverterUtils.DataSource("datuak/train_BoW.arff");
		ConverterUtils.DataSource testDs = new ConverterUtils.DataSource("datuak/dev_BoW.arff");

		Instances test = testDs.getDataSet();
		Instances data = ds.getDataSet();
		data.setClassIndex(0);
		test.setClassIndex(0);

		int minClassIndex = Utils.minIndex(data.attributeStats(data.classIndex()).nominalCounts);
		Kernel[] kernels = new Kernel[] {new PolyKernel(), new NormalizedPolyKernel()};

//		for (int i = -50; i < 50; i = i + 10) {
//			double tolerance = 1.0e-3;
//			tolerance += i * 2e-5;
//			if (tolerance <= 0 ){
//				continue;
//			}
//			double result = evaluateModel(data, test, tolerance, new NormalizedPolyKernel(), minClassIndex);
//			System.out.printf("Tolerance: %f, result: %f\n", tolerance, result);
//		}

		Kernel bestKernel = null;
		double bestExponent = Double.MIN_VALUE;
		double bestScore = Double.MIN_VALUE;

		for (Kernel kernel : kernels) {
			for (int i = -50; i < 50; i++) {
				double exponent = 1.0 + ((double) i / 100);
				double result = evaluateModel(data, test, kernel, exponent);

				System.out.printf("Kernel: %s, exponent: %f, result: %f\n", kernel.getClass().getSimpleName(), exponent,result);
				if (result > bestScore) {
					bestScore = result;
					bestKernel = kernel;
					bestExponent = exponent;
				}
			}
		}

		System.out.printf("Best kernel: %s, best exponent: %f, achieved weighted f-measure: %f", bestKernel.getClass().getSimpleName(), bestExponent, bestScore);
	}


	private static double evaluateModel(Instances train, Instances test, Kernel kernel, double exponent) throws Exception {
		SMO cls = new SMO();

		kernel.setOptions(new String[]{"-E", Double.toString(exponent)});
		cls.setKernel(kernel);

		cls.buildClassifier(train);


		Evaluation e = new Evaluation(train);
		e.evaluateModel(cls, test);


		return e.weightedFMeasure();
	}


}
