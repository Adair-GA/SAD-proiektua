import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class arffToBoW {
    public static void main(String[] args) {
        
        if (args.length < 2) {
            System.out.println("Erabilpena: java -jar arffToBoW.jar <train.arff> <hiztegia.arff>");
            System.exit(1);
        }

        String trainPath = args[0];
        String hiztegiaPath = args[1];
        
        Instances train = datuakKargatu(trainPath);

        if (train == null) {
            System.out.println("Datuak ezin izan dira kargatu.");
            System.exit(1);
        }
        System.out.println("Datuak kargatu dira.");

        StringToWordVector stwv = new StringToWordVector();
        stwv.setDictionaryFileToSaveTo(new File(hiztegiaPath));
        stwv.setLowerCaseTokens(true);
        

        try {
            stwv.setInputFormat(train);
            Instances trainBoW = Filter.useFilter(train, stwv);
            System.out.println("Train BoW sortu da.");

            if (train == null) {
                System.out.println("Errorea BoW sortzean.");
                System.exit(1);
            }

            saveInstances(trainBoW, trainPath);
            System.out.println("Train BoW gordeta. \n");
            
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
