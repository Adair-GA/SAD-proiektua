import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class baseline_eredua_ebaluazioa {
    public static void main(String[] args){
        if(args.length < 3){
            System.out.println("java -jar baseline_eredua_ebaluazioa.jar <train.arff><dev.arff><baseline.model><EvaluationBaseline.txt>");
            return;
        }
        String trainPath = args[0]; // input train (BoW eta atributu hautapena eginda) fitxategia
        String devPath = args[1]; // input dev (BoW eta atributu hautapena eginda) fitxategia
        String modelPath = args[2]; // Eredua gordeko den .model fitxategia
        String ebaluazioaPath = args[3]; // Kalitatea estimazioa gordeko den txt fitxategia

        try{
            double start = System.currentTimeMillis(); 
            
            // train datuak kargatu: 
            DataSource srcTrain = new DataSource(trainPath);
            Instances train = srcTrain.getDataSet();
            train.setClassIndex(0);
            // dev datuak kargatu:
            DataSource srcDev = new DataSource(devPath);
            Instances dev = srcDev.getDataSet();
            dev.setClassIndex(0);
            
            // Logistic Regression sailkatzailea sortu:
            Logistic model = new Logistic() ; 
            // sailkatzailea entrenatu train multzoarekin:
            model.buildClassifier(train);
            
            // Eredua gorde:
            SerializationHelper.write(modelPath, model);
            System.out.println("Eredua hemen gorde da: " + modelPath);
           
            // Ereduaren ebaluazioa egin:
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, dev);
            
            //Klase minoritarioa bilatu:
			AttributeStats stats = train.attributeStats(train.classIndex());
			String minClassName = "";
            int minClassIndex = -1;
			int minClassCount = Integer.MAX_VALUE;
			for(int i = 0; i < stats.nominalCounts.length; i++) {
				if(stats.nominalCounts[i] < minClassCount) {
					minClassCount = stats.nominalCounts[i];
					minClassIndex = i;
                    minClassName = train.classAttribute().value(i);
				}
			}

            // Klase minoritarioaren recall:
            double recallMinClass = eval.recall(minClassIndex); 
            // Klase minoritarioaren f-Measure
            double fMeasureMinClass = eval.fMeasure(minClassIndex); 

            // Exekuzio-data lortu:
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
			String exekuzioData = sdf.format(new Date());

            try(PrintWriter writer = new PrintWriter(new FileWriter(ebaluazioaPath))){ 
				writer.println("==== Linear Regression ebaluazioa ====");
				writer.println("Exekuzio data: " + exekuzioData);

				writer.println("\n--- Nahasmen matrizea ---");
				writer.println(eval.toMatrixString());
				
		        writer.println(eval.toClassDetailsString());

		        writer.println("\n--- Weighted Average ---");
		        writer.println("Precision: " + eval.weightedPrecision());
		        writer.println("Recall: " + eval.weightedRecall());
		        writer.println("F-Measure: " + eval.weightedFMeasure());
                
                writer.println("\n--- Klase Minoritarioa: " + minClassName + " ---");
                writer.println("Recall: "+ recallMinClass);
                writer.println("F-Measure: "+ fMeasureMinClass);

                writer.println("Exekuzio denbora: " + (System.currentTimeMillis() - start) / 1000 + " segundotan.");
			}
            System.out.println("Ebaluazio osatua. Emaitzak gorde dira: " + ebaluazioaPath);

        }catch(Exception e){
            e.printStackTrace();
        }
    }
}