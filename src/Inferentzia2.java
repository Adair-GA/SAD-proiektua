public class Inferentzia2 {
    public static void main(String[] args) {
        if (args.length < 8) {
            System.out.println("Erabilpena: java -jar Inferentzia.jar <train.arff> <dev.arff> <train_dev.arff> <train_dev.csv> <hiztegia.arff> <modelPath> <paramSet.txt> <kalitate_estimazioa.txt>");
            System.exit(1);
        }

        String trainPath = args[0]; // input trainBoW fitxategia
        String devPath = args[1]; // input devBoW fitxategia
        String trainDevPath = args[2]; // output train_devBoW fitxategia
        String trainDevCSVPath = args[3]; // output train_dev fitxategia CSV formatuan
        String hiztegiaPath = args[4]; // hiztegia gordetzeko output fitxategia
        String modelPath = args[5]; // output model fitxategia
        String paramSetPath = args[6]; // output param fitxategia
        String kalitateEstimazioaPath = args[7]; // output kalitate estimazioa fitxategia

        try{
            GetModel.getModel(trainPath, devPath, paramSetPath);
            csvToARFF.csvToArff(trainDevCSVPath, trainDevPath);

            eredu_optimoa_RHO_ebal.eredu_optimoa_sortu(trainDevPath, paramSetPath, hiztegiaPath, modelPath, kalitateEstimazioaPath);       
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(1);
        }
        System.out.println("Modeloa lortu da.");
    }
    
}
