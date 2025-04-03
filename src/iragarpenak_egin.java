import java.io.FileWriter;
import java.io.PrintWriter;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class iragarpenak_egin {
    public static void main(String[] args){
        if(args.length < 3){
            System.out.println("java -jar baseline_iragarpenak_egin.jar <test_blind.arff><.model><iragarpenak.txt>");
            return;
        }
        String testPath = args[0];
        String modelPath = args[1];
        String iragarpenakPath = args[2];

        try{ 
            // Eredua kargatu:
        	Classifier model = (Classifier) SerializationHelper.read(modelPath);
        	
        	//test blind kargatu:
        	DataSource src = new DataSource(testPath);
        	Instances blindData = src.getDataSet();
            blindData.setClassIndex(0);

            try (PrintWriter writer = new PrintWriter(new FileWriter(iragarpenakPath))) {
            	writer.println("==== Linear Regression Iragarpenak ====");
        
                //Instantzia bakoitzaren iragarpena egin:
                for(int i = 0; i < blindData.numInstances(); i++) {
                	double predIndex = model.classifyInstance(blindData.instance(i));
                	String predictedClass = blindData.classAttribute().value((int) predIndex);
                	
                	writer.printf("Instantzia %d: Iragarpena = %s \n", i + 1, predictedClass); 
                }   
            } 
            System.out.println("Iragarpenak gorde dira: " + iragarpenakPath);
            
        }catch(Exception e){
            e.printStackTrace();
        }
    }    
}