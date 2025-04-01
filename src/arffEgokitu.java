import java.io.File;
import java.util.ArrayList;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;

public class arffEgokitu{
    public static void main(String[] args) {
        
        if (args.length < 3){
            System.out.println("Erabilpena: java -jar arffEgokitu.jar <Data.arff> <Headers.arff> <hiztegia.arff>");
            System.exit(1);
        }

        String arffPath = args[0];
        String HeadersPath = args[1];
        String hiztegiaPath = args[2];

        Instances data = datuakKargatu(arffPath);
        Instances Headers = datuakKargatu(HeadersPath);

        if (Headers == null) {
            System.out.println("Headers datuak ezin izan dira kargatu.");
            System.exit(1);
        }

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

            dataBoW = atributuakEgokitu(dataBoW, Headers);
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

    public static Instances atributuakEgokitu(Instances data, Instances Headers) {
        try{
            List <String> trainHeaders = new ArrayList<String>();
            for (int i = 0; i < Headers.numAttributes(); i++) {
                trainHeaders.add(Headers.attribute(i).name());
            }

            List <Integer> ezabatzekoak = new ArrayList<Integer>();
            for (int i = 0; i < data.numAttributes(); i++) {
                if (!trainHeaders.contains(data.attribute(i).name())) {
                    ezabatzekoak.add(i);
                }
            }
            // Ezabatzeko atributuak array batean bildu
            int[] ezabatuArray = ezabatzekoak.stream().mapToInt(Integer::intValue).toArray(); 

            if (!ezabatzekoak.isEmpty()) {
                Remove removeFilter = new Remove();
                removeFilter.setAttributeIndicesArray(ezabatuArray);
                removeFilter.setInputFormat(data);
                data = Filter.useFilter(data, removeFilter);
            }
            return data;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }
}