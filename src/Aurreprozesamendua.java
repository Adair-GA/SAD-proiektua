public class Aurreprozesamendua {
    public static void main(String[] args) {
        if (args.length < 5) {
            System.out.println("Erabilpena: java -jar Aurreprozesamendua.jar <train.csv> <dev.csv> <test.csv> <arff/path/> <hiztegia.arff>");
            System.exit(1);
        }

        String trainPath = args[0]; // input train fitxategia
        String devPath = args[1]; // input dev fitxategia
        String testPath = args[2]; // input test fitxategia
        String arffPath = args[3]; // output ARFF fitxategia
        String hiztegiaPath = args[4]; // hiztegia gordetzeko output fitxategia

        // CSV fitxategiak ARFF formatura bihurtu
        String trainArffPath = arffPath + "train.arff";
        String devArffPath = arffPath + "dev.arff";
        String testArffPath = arffPath + "test_blind.arff";

        try{
            csvToARFF.csvToArff(trainPath, trainArffPath);
            csvToARFF.csvToArff(devPath, devArffPath);
            csvToARFF.csvToArff(testPath, testArffPath);

            arffToBoW.arffBoW(trainArffPath, hiztegiaPath);
            hiztegiaPath = hiztegiaPath.replace(".arff", "_egokitua.arff");
            arffEgokitu.egokitu(devArffPath, hiztegiaPath);
            arffEgokitu.egokitu(testArffPath, hiztegiaPath);
        }catch (Exception e){
            e.printStackTrace();
            System.exit(1);
        }
    }
}
