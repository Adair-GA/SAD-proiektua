import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class arffToNgram {
    public static void main(String[] args) {
        
        if (args.length < 2) {
            System.out.println("Erabilpena: java -jar arffToNgram.jar <train.arff> <hiztegia.arff>");
            System.exit(1);
        }

        String trainPath = args[0]; // train multzoaren arff
        String hiztegiaPath = args[1]; // hiztegia gordetzeko arff
        
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
            stwv.setLowerCaseTokens(true); // token guztiak letra xehe bihutu
            stwv.setInputFormat(train); 

            // N-gramen filtroa train multzoan aplikatu:
            Instances trainNgram = Filter.useFilter(train, stwv);
            System.out.println("Bihurketa N-grametara eginda.");
            // Instantzia berriak n-gramekin gorde arff batean:
            saveInstances(trainNgram, trainPath);
            System.out.println("Train N-gram gordeta. \n");
            
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
