import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class arffToBoW {
    public static void main(String[] args) {
        /* 
        if (args.length < 3) {
            System.out.println("Erabilpena: java -jar arffToBoW.jar <train.arff> <dev.arff> <test_blind.arff> <hiztegia.arff>");
            System.exit(1);
        }

        String trainPath = args[0];
        String devPath = args[1];
        String testPath = args[2];
        String hiztegiaPath = args[3];
        */

        String trainPath = "C:/Users/User/Documents/uni/3/EHES/Proiektua/train.arff";
        String devPath = "C:/Users/User/Documents/uni/3/EHES/Proiektua/dev.arff";
        String testPath = "C:/Users/User/Documents/uni/3/EHES/Proiektua/test_blind.arff";
        
        Instances train = datuakKargatu(trainPath);
        Instances dev = datuakKargatu(devPath);
        Instances test = datuakKargatu(testPath);

        if (train == null || dev == null || test == null) {
            System.out.println("Datuak ezin izan dira kargatu.");
            System.exit(1);
        }
        System.out.println("Datuak kargatu dira.");

        StringToWordVector stwv = new StringToWordVector();

        try {
            stwv.setInputFormat(train);
            Instances trainBoW = Filter.useFilter(train, stwv);
            System.out.println("Train BoW sortu da.");

            Instances devBoW = Filter.useFilter(dev, stwv);
            System.out.println("Dev BoW sortu da.");
            Instances testBoW = Filter.useFilter(test, stwv);
            System.out.println("Test_Blind BoW sortu da.");

            saveInstances(trainBoW, trainPath);
            saveInstances(devBoW, devPath);
            saveInstances(testBoW, testPath);
            
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

    }
    public static Instances datuakKargatu(String path){
        try{
            DataSource source = new DataSource(path);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(0);
            }
            return data;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }  

    public static void saveInstances(Instances data, String filePath) throws Exception {
        String path = filePath.replace(".arff", "_BoW.arff");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        System.out.println(path);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}
