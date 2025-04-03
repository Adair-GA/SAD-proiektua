import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

public class arffEgokitu_Ngram{
    public static void main(String[] args) {
        
        if (args.length < 2){
            System.out.println("Erabilpena: java -jar arffEgokitu_Ngram.jar <Data.arff> <hiztegia_Ngram.arff>");
            System.exit(1);
        }

        String arffPath = args[0]; // dev edo test input fitxategia
        String hiztegiaPath = args[1]; // input n-gramen hiztegia

        egokitu(arffPath, hiztegiaPath);
    }

    public static void egokitu(String arffPath, String hiztegiaPath) {
        
        // arff-tik datuak kargatu:
        Instances data = datuakKargatu(arffPath);

        // Datuak kargatu diren egiaztatu:
        if (data == null) {
            System.out.println("Datuak ezin izan dira kargatu.");
            System.exit(1);
        }
        System.out.println("Datuak kargatu dira.");

        // FixedDictionaryStringToWordVector sortu datuak n-grametan bihurtzeko:
        FixedDictionaryStringToWordVector fstwv = new FixedDictionaryStringToWordVector();
        fstwv.setDictionaryFile(new File(hiztegiaPath));
        fstwv.setLowerCaseTokens(true);

        try {
            fstwv.setInputFormat(data);
            
            // N-gramen filtroa datuetan aplikatu:
            Instances dataNgram = Filter.useFilter(data, fstwv);
            
            // Bihurketa arrakastatsua izan den egiaztatu:
            if (dataNgram == null) {
                System.out.println("Errorea N-gram sortzean.");
                System.exit(1);
            }
            System.out.println("N-gramak sortu dira.");

            // N-grametan bihurtutako datuak arff fitxategi berri batean gorde:
            saveInstances(dataNgram, arffPath);
            System.out.println("N-gramak gordeta. \n");

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
        // N-gramen arff fitxategirako irteera path-a sortu: 
        String path = filePath.replace(".arff", "_Ngram.arff");
        // Instantziak gorde arff-an:
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        System.out.println(path);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}