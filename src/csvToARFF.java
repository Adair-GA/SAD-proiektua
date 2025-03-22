import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

public class csvToARFF {
    public static void main(String[] args) throws Exception {
        
        if (args.length < 2) {
            System.out.println("Erabilpena: java -jar csvToARFFmal.jar <input.csv> <output.arff>");
            System.exit(1);
        }

        String iPath = args[0];
        String oPath = args[1];
        String icPath = iPath.replace(".csv", "_clean.csv");

        CleanCSV(iPath, icPath);

        PrintWriter writer = new PrintWriter(oPath);
        writer.print("@RELATION tweetSentiment\n\n");
        writer.print("@attribute Topic string\n" + 
                        "@attribute Sentiment {positive, neutral, negative, irrelevant}\n" + 
                        "@attribute TweetId string\n" + 
                        "@attribute TweetDate string\n" + 
                        "@attribute TweetText string\n");
        writer.print("\n@data\n");

        BufferedReader br = new BufferedReader(new FileReader(icPath));
        String line;
        line = br.readLine();  // Lehenengo lerroan header-ak gordeta daude, hau ez idatzi
        while ((line = br.readLine()) != null) {
            writer.print(line + "\n");
        }
        br.close();
        writer.close();

        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(oPath));
        Instances data = loader.getDataSet();

        data.deleteAttributeAt(3);
        data.deleteAttributeAt(2);
        data.deleteAttributeAt(0);
        data.setClassIndex(0);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(oPath));
        saver.writeBatch();
    }

    public static void CleanCSV(String iPath, String icPath) {

        List<String> hasieraHitzak = Arrays.asList("Topic","google", "microsoft", "twitter", "apple");

        try (BufferedReader br = new BufferedReader(new FileReader(iPath));
             BufferedWriter bw = new BufferedWriter(new FileWriter(icPath))) {
            
            String line;
            while ((line = br.readLine()) != null) {
                // Eliminar comillas dobles, dentro de un string de csv se representan con dos comillas dobles
                line = line.replace("\"\"", ""); 

                String firstWord = "";
                // Usa epacio como split tambien, porque a veces la estructura esta tan mal que no hay comas
                String[] parts = line.split("[ ,]"); 
                if (parts.length > 4) {
                     // Mira la primera palabra, si no esta en la lista de palabras la estructura de la linea no esta del todo bien
                    firstWord = parts[0].replace("\"", "").trim(); 
                    if (hasieraHitzak.contains(firstWord)) {
                        // Si la primera palabra es Topic, eso es que esta leyendo la linea de headers, 
                        // le quito las comillas porque creo que a weka no le gustan
                        if (firstWord.equals("Topic")) { 
                            line = line.replace("\"", "");
                        }
                        
                        // test_blind tiene UNKNOWN para missingClass, peroa weka no le gusta, asi que lo cambio por ?, que si lo pilla automaticamente
                        if (parts.length > 1 && parts[1].equals("\"UNKNOWN\"")) {
                            parts[1] = "?"; // Reemplaza UNKNOWN con ?
                        }
                        line = String.join(",", parts);
                        bw.write(line);
                        bw.newLine();
                    }
                }
            }
            bw.close();
            System.out.println("Archivo CSV limpiado correctamente.");

        } catch (IOException e) {
            System.out.println("Errorea CSV fitxategian: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
