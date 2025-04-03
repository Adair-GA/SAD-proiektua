import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.FileReader;

public class train_and_dev {
    public static void main(String[] args) {
        if(args.length < 3){
            System.out.println("java -jar train_and_dev.jar <train.csv><dev.csv><train_dev.csv>");
            System.exit(1);
        }
        String trainPath = args[0]; // input train fitxategia
        String devPath = args[1]; // input dev fitxategia
        String mergedPath = args[2]; // output train_dev fitxategia

        try{
            // BufferedWriter sortu train eta dev batuta idazteko:
            BufferedWriter writer = new BufferedWriter(new FileWriter(mergedPath));
            // BufferedReader-ak sortu train eta dev fitxategiak irakurtzeko:
            BufferedReader reader1 = new BufferedReader(new FileReader(trainPath));
            BufferedReader reader2 = new BufferedReader(new FileReader(devPath));
            
            // train fitxategiaren goiburua irakurri eta idatzi output fitxategian:
            String header = reader1.readLine();
            writer.write(header + "\n");

            String line;
            // train fitxategiko lerro guztiak irakurri eta output-fitxategian idatzi:
            while((line = reader1.readLine()) != null){
                writer.write(line + "\n");
            }
            reader1.close(); // train fitxategiaren reader-a itxi

            reader2.readLine();
            // dev fitxategiko lerro guztiak irakurri eta output-fitxategian idatzi:
            while ((line = reader2.readLine()) != null) {
                writer.write(line + "\n");
            }
            reader2.close(); // dev fitxategiaren reader-a itxi

            writer.close(); // output fitxategiaren writer-a itxi
            System.out.println("Train eta Dev batu dira.");
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
