public class Inferentzia {
    public static void main(String[] args) {
        if (args.length < 10) {
            System.out.println("Erabilpena: java -jar Inferentzia.jar <train.arff> <dev.arff> <train_dev.arff> <train.csv> <dev.csv> <train_dev.csv> <hiztegia.arff> <modelPath> <paramSet.txt> <kalitate_estimazioa.txt>");
            System.exit(1);
        }

        String trainPath = args[0]; // input trainBoW fitxategia
        String devPath = args[1]; // input devBoW fitxategia
        String trainDevPath = args[2]; // output train_devBoW fitxategia
        String trainCSVPath = args[3]; // input train fitxategia CSV formatuan
        String devCSVPath = args[4]; // input dev fitxategia CSV formatuan
        String trainDevCSVPath = args[5]; // output train_dev fitxategia CSV formatuan
        String hiztegiaPath = args[6]; // hiztegia gordetzeko output fitxategia
        String modelPath = args[7]; // output model fitxategia
        String paramSetPath = args[8]; // output param fitxategia
        String kalitateEstimazioaPath = args[9]; // output kalitate estimazioa fitxategia

        try{
            GetModel.getModel(trainPath, devPath, paramSetPath);
            
            train_and_dev.mergeTrainAndDev(trainCSVPath, devCSVPath, trainDevCSVPath);
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
