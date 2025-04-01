import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
public class arffEgokitu{
    public static void main(String[] args) {
        
        if (args.length < 2){
            System.out.println("Erabilpena: java -jar arffEgokitu.jar <Data.arff> <hiztegia.arff>");
            System.exit(1);
        }

        String arffPath = args[0];
        String hiztegiaPath = args[1];

        Instances data = datuakKargatu(arffPath);

        if (data == null) {
            System.out.println("Datuak ezin izan dira kargatu.");
            System.exit(1);
        }
        System.out.println("Datuak kargatu dira.");

        FixedDictionaryStringToWordVector fstwv = new FixedDictionaryStringToWordVector();
        fstwv.setDictionaryFile(new File(hiztegiaPath));

        fstwv.setLowerCaseTokens(true);
        
        fstwv.setTFTransform(true);
        fstwv.setIDFTransform(true);
        fstwv.setOutputWordCounts(true); // Dokumentuan hitzaren agerpen kopurua
        
        try {
            fstwv.setInputFormat(data);
            Instances dataBoW = Filter.useFilter(data, fstwv);

            if (dataBoW == null) {
                System.out.println("Errorea BoW sortzean.");
                System.exit(1);
            }
            System.out.println("BoW sortu da.");

            if (dataBoW == null) {
                System.out.println("Errorea atributuak egokitzean.");
                System.exit(1);
            }
            System.out.println("Atributuak egokitu dira. \n");

            saveInstances(dataBoW, arffPath);
            System.out.println("BoW gordeta. \n");

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        
    }
    public static Instances datuakKargatu(String path){
        try{
            DataSource source = new DataSource(path);
            Instances data = source.getDataSet();
            data.setClassIndex(0);
            return data;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }  

    public static void saveInstances(Instances data, String filePath) throws Exception {
        String path = filePath.replace(".arff", "_as_BoW.arff");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        System.out.println(path);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}