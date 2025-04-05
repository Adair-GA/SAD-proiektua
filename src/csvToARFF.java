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
    }

    public static void csvToArff(String iPath, String oPath) throws Exception {
        String icPath = iPath.replace(".csv", "_clean.csv");
        CleanCSV(iPath, icPath);

        PrintWriter writer = new PrintWriter(oPath);
        writer.print("@RELATION tweetSentiment\n\n");
        writer.print("@attribute Sentiment {positive, neutral, negative, irrelevant}\n" + 
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

        data.setClassIndex(0);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(oPath));
        saver.writeBatch();
    }

    public static void CleanCSV(String iPath, String icPath) {

        List<String> klaseHitzak = Arrays.asList("\"Sentiment\"","\"negative\"", "\"positive\"", "\"neutral\"", "\"irrelevant\"", "\"UNKNOWN\"");

        try (BufferedReader br = new BufferedReader(new FileReader(iPath));
             BufferedWriter bw = new BufferedWriter(new FileWriter(icPath))) {
            
            String line;
            while ((line = br.readLine()) != null) {
                line = line.replace("\"\"", "");
                String[] parts = line.split(",");
                String lerroa;
                if (parts.length > 1){
                    for (int i = 0; i < parts.length; i++){
                        if (klaseHitzak.contains(parts[i])){
                            lerroa = parts[i];
                            if (lerroa.contains("\"UNKNOWN\"")){
                                lerroa = lerroa.replace("\"UNKNOWN\"", "?");
                            }
                            lerroa = lerroa + "," + parts[i+3].replaceAll("[^a-zA-Z , \"]", "");
                            for (int j = i+4; j < parts.length; j++){
                                lerroa = lerroa + " " + parts[j].replaceAll("[^a-zA-Z ]", "");
                            }
                            bw.write(lerroa);
                            bw.newLine();
                        }
                    }
                }
            }
            bw.close();
            System.out.println("CSV fitxategia gorde da.");

        } catch (IOException e) {
            System.out.println("Errorea CSV fitxategian: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
