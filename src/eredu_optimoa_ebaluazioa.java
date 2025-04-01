import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class eredu_optimoa_ebaluazioa {
    public static void main(String[] args) {
        if(args.length < 3){
            System.out.println("java -jar eredu_optimoa_ebaluazioa <trainDev.arff><model><kalitate_estimazioa.txt>");
        }
        String trainDevPath = args[0];
        String modelPath = args[1]; // creo
        String estimazioaPath = args[2];

        try{
            DataSource src = new DataSource(trainDevPath);
            Instances data = src.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // DE MOMENTO: Luego SVM
            Classifier model = (Classifier) SerializationHelper.read(modelPath); 
            
            int repeKop = 10; 
            double[] accuracies = new double[repeKop];

            for(int i = 0; i < repeKop; i++){
                Resample resample = new Resample();
                resample.setRandomSeed(new Random(1).nextInt());
                resample.setNoReplacement(true);
                resample.setSampleSizePercent(70);
                resample.setInputFormat(data);

                Instances trainData = Filter.useFilter(data, resample);
                Instances testData = new Instances(data);

                for(int j = 0; j < data.numInstances(); j++){
                    if(!trainData.contains(data.instance(j))){
                        testData.add(data.instance(j));
                    }
                }

                model.buildClassifier(trainData);

                Evaluation eval = new Evaluation(trainData);
                eval.evaluateModel(model, testData);

                accuracies[i] = eval.pctCorrect();
            }

            double mean = batezbestekoa_kalkulatu(accuracies);
            double stdev = stdv_kalkulatu(accuracies, mean);

            try(BufferedWriter writer = new BufferedWriter(new FileWriter(estimazioaPath))){
                writer.write("Batez bestekoa: " + mean + "\n");
                writer.write("Desbiderapen estandarra: " + stdev + "\n");
            }

            System.out.println("Ebaluazioa osatu da:" + estimazioaPath);

            
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    private static double batezbestekoa_kalkulatu(double[] balioak){
        double sum = 0;
        for(double b: balioak){
            sum += b;
        }
        return sum / balioak.length;
    }

    private static double stdv_kalkulatu(double[] balioak, double mean){
        double sum = 0;
        for(double b: balioak){
            sum += Math.pow(b - mean,2);
        }
        return Math.sqrt(sum / balioak.length);
    }
}
