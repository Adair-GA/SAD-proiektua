import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;


public class GetModel {
	public static void main(String[] args) throws Exception {
		if(args.length != 3){
			System.out.println("Usage: java -jar GetModel.jar <input>train.arff <input>dev.arff <output>model");
		}
		
		ConverterUtils.DataSource ds = new ConverterUtils.DataSource(args[0]);
		ConverterUtils.DataSource testDs = new ConverterUtils.DataSource(args[1]);

		Instances test = testDs.getDataSet();
		Instances data = ds.getDataSet();
		data.setClassIndex(0);
		test.setClassIndex(0);

		Kernel[] kernels = new Kernel[] {new PolyKernel(), new NormalizedPolyKernel()};

		Kernel bestKernel = kernels[0];
		double bestExponent = Double.MIN_VALUE;
		double bestScore = Double.MIN_VALUE;

		for (Kernel kernel : kernels) {
			for (int i = 0; i < 100; i= i + 2) {
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

		SMO classifier = new SMO();
		bestKernel.setOptions(new String[]{"-E", Double.toString(bestExponent)});

		classifier.setKernel(bestKernel);

		classifier.buildClassifier(data);


		SerializationHelper.write(args[2], classifier);
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
