import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class eredu_optimoa_ebaluazioa {
    public static void main(String[] args) {
        if(args.length < 3){
            System.out.println("java -jar eredu_optimoa_ebaluazioa <trainDev.arff><parametroak.txt><kalitate_estimazioa.txt>");
        }
        String trainDevPath = args[0]; // train eta dev batuta dituen arff
        String modelPath = args[1]; // Entrenatutako eredua
        String estimazioaPath = args[2]; // Kalitatea estimazioa gordeko den output-a

        try{
            // train eta dev batuta dituen arff kargatu:
            DataSource src = new DataSource(trainDevPath);
            Instances data = src.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Crear un modelo y meterle los parámetro que me den
            // de train_dev.arff hacer atributu hautapena
            // entrenar el modelo
            
            // DE MOMENTO: Luego SVM
            // Entrenatutako eredua kargatu:
            Classifier model = (Classifier) SerializationHelper.read(modelPath); 
            
            int repeKop = 10; // Errepikapen kopurua eredua ebaluatzeko
            double[] accuracies = new double[repeKop]; // Errepikapen bakoitzaren zehaztasuna gordetzeko array-a 

            // Ebaluazioa egin Repeated Hold-out erabiliz: 
            for(int i = 0; i < repeKop; i++){
                // Stratified Hold-Out erabili:
                Resample resample = new Resample();
                resample.setRandomSeed(new Random(1).nextInt());
                resample.setNoReplacement(true);
                resample.setSampleSizePercent(70);
                resample.setInputFormat(data);

                // Resample ezarri train multzoa lortzeko:
                Instances trainData = Filter.useFilter(data, resample);
                
                // ATRIBUTU HAUTAPENA a TRAIN, y luego a DEV

                // test multzoa entrenamendu multzoan dauden instantzien osagarria da:
                Instances devData = new Instances(data);

                for(int j = 0; j < data.numInstances(); j++){
                    if(!trainData.contains(data.instance(j))){
                        devData.add(data.instance(j));
                    }
                }

                // Eredua entrenatu train multzoarekin:
                model.buildClassifier(trainData);
                
                // evaluación con train con el atributu hautapena hecho (y dev)
                
                // Eredua ebaluatu test multzoa erabiliz:
                Evaluation eval = new Evaluation(trainData);
                eval.evaluateModel(model, devData);

                // Ereduaren zehaztasuna gorde errepikapen honetarako:
                accuracies[i] = eval.pctCorrect();
            }

            // Errepikapenetan lortutako zehaztapenen batez bestekoa 
            // eta desbiderapen estandarra kalkulatu:
            double mean = batezbestekoa_kalkulatu(accuracies);
            double stdev = stdv_kalkulatu(accuracies, mean);

            // Ebaluazioaren emaitzak gorde:
            try(BufferedWriter writer = new BufferedWriter(new FileWriter(estimazioaPath))){
                writer.write("Batez bestekoa: " + mean + "\n");
                writer.write("Desbiderapen estandarra: " + stdev + "\n");
            }

            System.out.println("Ebaluazioa osatu da:" + estimazioaPath);

            
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    private static double batezbestekoa_kalkulatu(double[] balioak){
        double sum = 0;
        // array-eko balio guztiak gehitu:
        for(double b: balioak){
            sum += b;
        }
        // batura elementu kopuruagatik zatitu batez bestekoa lortzeko:
        return sum / balioak.length;
    }

    private static double stdv_kalkulatu(double[] balioak, double mean){
        double sum = 0;
        // Balio bakoitzaren eta batez bestekoaren arteko aldeak ber bi batu
        for(double b: balioak){
            sum += Math.pow(b - mean,2);
        }
        // alde horien batez bestekoaren erro karratua itzuli:
        return Math.sqrt(sum / balioak.length);
    }
}
