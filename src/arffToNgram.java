import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Map;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class arffToNgram {
    public static void main(String[] args) {
        
        if (args.length < 2) {
            System.out.println("Erabilpena: java -jar arffToNgram.jar <train.arff> <hiztegia.arff>");
            System.exit(1);
        }

        String trainPath = args[0]; // input train fitxategia
        String hiztegiaPath = args[1]; // hiztegia gordetzeko output fitxategia

        arffNgram(trainPath, hiztegiaPath);
    }

    public static void arffNgram(String trainPath, String hiztegiaPath) {
        // train-eko datuak kargatu:
        Instances train = datuakKargatu(trainPath);
        // Datuak kargatu diren egiaztatu:
        if (train == null) {
            System.out.println("Datuak ezin izan dira kargatu.");
            System.exit(1);
        }
        System.out.println("Datuak kargatu dira.");
        
        try {
            // Tokenizer-a sortu testua n-grama bihurtzeko:
            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(1); // n-gramen gutxieneko tamaina
            tokenizer.setNGramMaxSize(2); // n-gramen gehienezko tamaina
            tokenizer.setDelimiters(" \\W"); // Mugatzaileak (hutsune eta karaktere ez-alfanumerikoak)

            // StringToWordVector sortu instantziak n-grametan prozesatzeko:
            StringToWordVector stwv = new StringToWordVector();
            stwv.setTokenizer(tokenizer); // tokenizer-a ezarri
            stwv.setDictionaryFileToSaveTo(new File(hiztegiaPath)); // n-gramak gordetzeko hiztegia ezarri
            stwv.setLowerCaseTokens(true); // token guztiak letra xehe bihurtu
            stwv.setInputFormat(train); 

            // N-gramen filtroa train multzoari ezarri:
            Instances trainNgram = Filter.useFilter(train, stwv);
            System.out.println("Bihurketa N-grametara eginda.");

            // N-gram instantzietan atributu hautapena egin:
            Instances trainBerria = atributuHautapena(trainNgram);
            if (trainBerria == null) {
                System.out.println("Errorea atributu hautapenean.");
                System.exit(1);
            }
            System.out.println("Atributu hautapena egin da. \n");

            // Instantzia berriak n-gramekin gorde arff fitxategi batean:
            saveInstances(trainBerria, trainPath);
            System.out.println("Train N-gram gordeta. \n");

            hiztegiaEgokitu(trainBerria, hiztegiaPath);
            
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

    public static Instances atributuHautapena(Instances data){
        AttributeSelection as = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(3000); 
        ranker.setThreshold(0.0); 
        // Threshold-ak atributuei eragiteko gutxieneko balioa 0.0 baino handiagoa da, 
        // puntu horretan 744 atributu hautatzen dira.
        // Hala ere, horrek ez dirudi ebaluazioan balio optimoak itzultzen dituenik (gutxienez baselinearekin)        
        // atributu optimoak 3000 inguruan daudela dirudi
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

                System.out.println("Birmoldaketa egin da, klasea 0 posizioan dago.");
                return finalData;
            }

            return newData;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    public static void hiztegiaEgokitu(Instances headers, String hiztegiaPath){
        try{
            // Hiztegiaren atributuak eta haien agerpen kopurua gordetzeko mapa:
            Map <String, Integer> atributuHizt = new HashMap<String, Integer>();
            // Hiztegia irakurtzeko:
            BufferedReader br = new BufferedReader(new FileReader(hiztegiaPath));
            // Fitxategia sortu hiztegi egokitua gordetzeko:
            BufferedWriter wr = new BufferedWriter(new FileWriter(hiztegiaPath.replace(".arff", "_egokitua.arff")));
            
            String line;
            line = br.readLine();  // Doc kopurua adierazten du
            wr.write(line);
            wr.newLine();
            
            // Hiztegia irakurri esta HashMap batean gorde:
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                String atributua = parts[0];
                Integer kop = Integer.parseInt(parts[1]);
                atributuHizt.put(atributua, kop);
            }
            br.close(); 

            for (int i = 0; i < headers.numAttributes(); i++) {
                String atributua = headers.attribute(i).name(); // Atributuaren izena hartu
                if (atributuHizt.containsKey(atributua)) { // Hiztegian badago
                    wr.write(atributua + "," + atributuHizt.get(atributua)); // Hiztegi egokituan idatzi
                    wr.newLine();
                }    
            }
            wr.close();

            // Jatorrizko hiztegiaren fitxategia ezabatu:
            File hiztegiaFile = new File(hiztegiaPath);
            hiztegiaFile.delete(); 

            System.out.println("Hiztegia egokitu da.");

        }catch (Exception e){
            e.printStackTrace();
            System.out.println("Errorea hiztegia egokitzen.");
            System.exit(1);
        }
    }
}
