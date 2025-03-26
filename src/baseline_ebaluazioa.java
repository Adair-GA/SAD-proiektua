import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class baseline_ebaluazioa {
    public static void main(String[] args){
        if(args.length < 3){
            System.out.println("java -jar baseline_ebaluazioa.jar <train.arff><dev.arff><Baseline.model><EvaluationBaseline.txt>");
            return;
        }
        String trainPath = args[0];
        String devPath = args[1];
        String modelPath = args[2];
        String ebaluazioaPath = args[3];

        try{
            DataSource srcTrain = new DataSource(trainPath);
            Instances train = srcTrain.getDataSet();
            train.setClassIndex(0);

            DataSource srcDev = new DataSource(devPath);
            Instances dev = srcDev.getDataSet();
            dev.setClassIndex(0);

            LinearRegression model = new LinearRegression();
            model.buildClassifier(train);

            SerializationHelper.write(modelPath, model);
            System.out.println("Eredua hemen gorde da: " + modelPath);
           
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, dev);
            
            //Klase minoritarioa bilatu:
			AttributeStats stats = train.attributeStats(train.classIndex());
			int minClassIndex = -1;
			int minClassCount = Integer.MAX_VALUE;
			for(int i = 0; i < stats.nominalCounts.length; i++) {
				if(stats.nominalCounts[i] < minClassCount) {
					minClassCount = stats.nominalCounts[i];
					minClassIndex = i;
				}
			}

            double recallMinClass = eval.recall(minClassIndex); // Klase minoritarioaren recall

            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
			String exekuzioData = sdf.format(new Date());

            try(PrintWriter writer = new PrintWriter(new FileWriter(ebaluazioaPath))){ 
				writer.println("==== Linear Regression ebaluazioa ====");
				writer.println("Exekuzio data: " + exekuzioData);

				writer.println("\n--- Nahasmen matrizea ---");
				writer.println(eval.toMatrixString());
				
				writer.println("\n--- Precision klasearen balio bakoitzeko ---");
		        writer.println(eval.toClassDetailsString());

		        writer.println("\n--- Weighted Average ---");
		        writer.println("Precision: " + eval.weightedPrecision());
		        writer.println("Recall: " + eval.weightedRecall());
		        writer.println("F-Measure: " + eval.weightedFMeasure());
                writer.println("Klase minoritarioaren Recall: "+ recallMinClass);
		        writer.println("\nEbaluazio-emaitzak amaituta.");
			}
            System.out.println("Ebaluazio osatua. Emaitzak gorde dira hemen: " + ebaluazioaPath);

        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
