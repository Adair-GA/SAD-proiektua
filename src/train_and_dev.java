import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.FileReader;

public class train_and_dev {
    public static void main(String[] args) {
        if(args.length < 3){
            System.out.println("java -jar train_and_dev <train.csv><dev.csv><merged.csv>");
            System.exit(1);
        }
        String trainPath = args[0];
        String devPath = args[1];
        String mergedPath = args[2];

        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(mergedPath));
            BufferedReader reader1 = new BufferedReader(new FileReader(trainPath));
            BufferedReader reader2 = new BufferedReader(new FileReader(devPath));

            String header = reader1.readLine();
            writer.write(header + "\n");

            String line;
            while((line = reader1.readLine()) != null){
                writer.write(line + "\n");
            }
            reader1.close();

            reader2.readLine();
            while ((line = reader2.readLine()) != null) {
                writer.write(line + "\n");
            }
            reader2.close();

            writer.close();
            System.out.println("Train eta Dev batuta.");
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
