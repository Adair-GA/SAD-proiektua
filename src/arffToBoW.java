import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Dictionary;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Reorder;
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
        
        // stwv.setDictionaryFileToSaveTo(new File(hiztegiaPath));
        stwv.setLowerCaseTokens(true);
        
        stwv.setTFTransform(true);
        stwv.setIDFTransform(true);
        stwv.setOutputWordCounts(true); // Dokumentuan hitzaren agerpen kopurua

        try {
            stwv.setInputFormat(train);
            Instances trainBoW = Filter.useFilter(train, stwv);
            if (trainBoW == null) {
                System.out.println("Errorea BoW sortzean.");
                System.exit(1);
            }
            System.out.println("Train BoW sortu da. \n");

            Instances trainBerria = atributuHautapena(trainBoW);
            if (trainBerria == null) {
                System.out.println("Errorea atributu hautapenean.");
                System.exit(1);
            }
            System.out.println("Atributu hautapena egin da. \n");

            Instances trainHeaders = new Instances(trainBerria);
            trainHeaders.delete(); 
            saveInstances(trainHeaders, trainPath.replace(".arff", "_Headers.arff"));
            System.out.println("Train Headers gordeta. \n");

            saveInstances(trainBerria, trainPath);
            System.out.println("Train BoW gordeta. \n");

            // Temporal, funciona pero no es del todo correcto
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(hiztegiaPath))) {
                // Recorrer todos los atributos y guardarlos como diccionario
                for (int i = 0; i < trainBerria.numAttributes(); i++) { // Excluye la clase si es el primer atributo
                    writer.write(trainBerria.attribute(i).name());
                    writer.newLine();
                }
                System.out.println("Diccionario guardado en: " + hiztegiaPath);
            }
            
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

    public static Instances atributuHautapena(Instances data){
        AttributeSelection as = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(3000); 
        ranker.setThreshold(-1.7976931348623157E308); 
        // El valor minimo para que este threshold afecete a los atributos es mayor que 0.0, en ese punto se seleccionan 744 atributos
        // Sin embargo, esto no parece que devuelva valores optimos en la ebaluacion (por lo menos con baseline), 
        // los atributos optimos parecen estar sobre los 3000
        as.setEvaluator(eval);
        as.setSearch(ranker);
        try{
            as.setInputFormat(data);
            Instances newData = Filter.useFilter(data, as);
            System.out.println("newData: " + newData.numAttributes() + " atributu ditu atributu hautapena egin eta gero.");

            // Klase berriaren posizioa gorde
            int originalClassIndex = data.classIndex();
            int newClassIndex = newData.classIndex();

            // Klasearen posizioa aldatzekotan, birmoldatu
            if (newClassIndex != originalClassIndex) {
                System.out.println("Klasea posizioz aldatu da, birmoldatzen...");

                // Atributuen ordena eraiki klasea lehen posizioan jarriz
                StringBuilder order = new StringBuilder();
                order.append(newClassIndex + 1).append(",");
                for (int i = 0; i < newData.numAttributes(); i++) {
                    if (i != newClassIndex) {
                        order.append(i + 1).append(",");
                    }
                }
                
                // Reorder erabili atributu hautapena egitean azken posizioan jarri den klasea 0 posizioan jartzeko
                Reorder reorder = new Reorder();
                reorder.setAttributeIndices(order.toString());
                reorder.setInputFormat(newData);
                Instances finalData = Filter.useFilter(newData, reorder);
                finalData.setClassIndex(finalData.numAttributes() - 1);

                System.out.println("Birmoldaketa egin da, kalsea 0 posizioan dago.");
                return finalData;
            }

            return newData;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }
}
