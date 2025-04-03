import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Constructor;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.NormalizedPolyKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class eredu_optimoa_ebaluazioa {
    public static void main(String[] args) {
        if(args.length < 4){
            System.out.println("java -jar eredu_optimoa_ebaluazioa <train_dev.arff><parametro_optimoak.txt><hiztegia.arff><Eredu_Optimoa.model><kalitate_estimazioa.txt>");
        }
        String trainDevPath = args[0]; // train eta dev batuta dituen arff fitxategia
        String paramOptimoakPath = args[1]; // parametro optimoak gordeta dituen txt fitxategia
        String hiztegiaPath = args[2]; // output hiztegia BoW formatua sortzeko
        String modelPath = args[3]; // Eredu optimoa gordeko den .model fitxategia
        String estimazioaPath = args[4]; // Kalitatea estimazioa gordeko den txt fitxategia


        try{
            // train eta dev bateratutako datuak kargatu:
            DataSource src = new DataSource(trainDevPath);
            Instances data = src.getDataSet();
            data.setClassIndex(0);

            // Parametro optimoak irakurri:
            BufferedReader br = new BufferedReader(new FileReader(paramOptimoakPath));
            String line = br.readLine(); 
            br.close();

            if (line == null || line.isEmpty()) {
                throw new IOException("Parametroen txt-a hutsik dago edo ez da baliozkoa.");
            }
            // Parametro optimoen informazioa banatu eta prozesatu:
            String[] parts = line.split(" -E ");
            String kernelName = parts[0].trim();
            String exponentStr = parts[1].trim();
            double exponent = 1.0; // Balio lehenetsia
            if(!exponentStr.isEmpty()){
                exponent = Double.parseDouble(exponentStr.replace(",", "."));
            }

            SMO smo = new SMO();

            try{
                // Kernel klasearen izena erabiliz kernel instantzia sortu:
                Class<?> kernelClass = Class.forName(kernelName);
                Constructor<?> constructor = kernelClass.getConstructor();
                Object kernelObject = constructor.newInstance();
                
                if (kernelObject instanceof Kernel){
                    Kernel kernel = (Kernel) kernelObject;
                    if (kernel instanceof NormalizedPolyKernel){
                        ((NormalizedPolyKernel) kernel).setExponent(exponent);
                    } else if (kernel instanceof PolyKernel){
                        ((PolyKernel) kernel).setExponent(exponent);
                    } else {
                        throw new IllegalArgumentException();
                    }
                    // Kernel-a ezarri SMO-ari:
                    smo.setKernel(kernel);
                } else {
                    throw new IllegalArgumentException("Kernel mota ez da bateragarria SMO-rekin.");
                }

            }catch(Exception e){
                e.printStackTrace();
                System.out.println("Errorea kernel-a kargatzean");
                return;
            }
            
            // Atributu hautapena
            arffToBoW.arffBoW(trainDevPath, hiztegiaPath);
            Instances trainDevData = datuakKargatu(trainDevPath.replace(".arff", "_as_BoW.arff"));
            
            // SMO eredua entrenatu:
            smo.buildClassifier(trainDevData);

            // Eredua gorde:
            SerializationHelper.write(modelPath, smo);
            
            int repeKop = 10; // Errepikapen kopurua eredua ebaluatzeko
            double[] recallCminBalioak = new double[repeKop]; // Klase minoritarioaren recall
            double[] accuracies = new double[repeKop]; 
            double[] fmeasures = new double[repeKop]; 
            double[] precisions = new double[repeKop]; 
            double[] recalls = new double[repeKop]; 

            // Repeated Hold-out ebaluazioa: 
            for(int i = 0; i < repeKop; i++){
                // Stratified Hold-Out erabili:
                Resample resample = new Resample();
                resample.setRandomSeed(i); 
                resample.setNoReplacement(true);
                resample.setSampleSizePercent(70);
                resample.setInputFormat(data);

                // train datuak sortu:
                Instances trainData = Filter.useFilter(data, resample);
                trainData.setClassIndex(0);

                // Atributu hautapena train:
                String trainPath = "Ebaluazio_eredu_optimoa/repeated_HO_train_" + i + ".arff";
                String hizPath = "Ebaluazio_eredu_optimoa/repeated_HO_hiztegi_" + i + ".arff";
                saveInstances(trainData, trainPath);
                arffToBoW.arffBoW(trainPath, hizPath);
                Instances trainBoW = datuakKargatu(trainPath.replace(".arff", "_as_BoW.arff"));

                // test multzoa sortu:
                Instances devData = new Instances(data);
                for(int j = 0; j < data.numInstances(); j++){
                    if(!trainData.contains(data.instance(j))){
                        devData.add(data.instance(j));
                    }
                }
                devData.setClassIndex(0);

                // Atributu hautapena dev:
                String devPath = "Ebaluazio_eredu_optimoa/repeated_HO_dev_" + i + ".arff";
                saveInstances(devData, devPath);
                arffEgokitu.egokitu(devPath, hizPath.replace(".arff", "_egokitua.arff")); 
                Instances devBoW = datuakKargatu(devPath.replace(".arff", "_as_BoW.arff"));

                // train-eko klase minoritarioa aurkitu:
                AttributeStats stats = trainBoW.attributeStats(trainBoW.classIndex());
                int minClassIndex = -1;
                int minClassCount = Integer.MAX_VALUE;
                for(int j = 0; j < stats.nominalCounts.length; j++) {
                	if(stats.nominalCounts[j] < minClassCount) {
                		minClassCount = stats.nominalCounts[j];
                		minClassIndex = j;
                	}
                }
                
                // Eredua entrenatu eta ebaluatu:
                smo.buildClassifier(trainBoW);
                Evaluation eval = new Evaluation(trainBoW);
                eval.evaluateModel(smo, devBoW);
                
                // Ebaluazio metrikak kalkulatu:
    			recallCminBalioak[i] = eval.recall(minClassIndex);
                accuracies[i] = eval.pctCorrect();                
                fmeasures[i] = eval.weightedFMeasure();
                precisions[i] = eval.weightedPrecision();
                recalls[i] = eval.weightedRecall();
                System.out.println(i + " Hold-Out eginda.");
            }

            // Batez bestekoak eta desbiderapen estandarrak kalkulatu:
            double meanRmin = avg_kalkulatu(recallCminBalioak);
            double stdevRmin = stdev_kalkulatu(recallCminBalioak, meanRmin);

            double meanA = avg_kalkulatu(accuracies);
            double stdevA = stdev_kalkulatu(accuracies, meanA);

            double meanF = avg_kalkulatu(fmeasures);
            double stdevF = stdev_kalkulatu(fmeasures, meanF);

            double meanP = avg_kalkulatu(precisions);
            double stdevP = stdev_kalkulatu(precisions, meanP);

            double meanR = avg_kalkulatu(recalls);
            double stdevR = stdev_kalkulatu(recalls, meanR);

            // Emaitzak gorde:
            try(BufferedWriter writer = new BufferedWriter(new FileWriter(estimazioaPath))){
                writer.write("Recall Klase minoritarioa - Batez bestekoa : " + meanRmin + "\n");
                writer.write("Recall Klase minoritarioa- Desbiderapen estandarra : " + stdevRmin + "\n");
                writer.write("\n");

                writer.write("Accuracy - Batez bestekoa : " + meanA + "\n");
                writer.write("Accuracy - Desbiderapen estandarra : " + stdevA + "\n");
                writer.write("\n");

                writer.write("F-Measure - Batez bestekoa : " + meanF + "\n");
                writer.write("F-Measure - Desbiderapen estandarra : " + stdevF + "\n");
                writer.write("\n");

                writer.write("Precision - Batez bestekoa : " + meanP + "\n");
                writer.write("Precision - Desbiderapen estandarra : " + stdevP + "\n");
                writer.write("\n");

                writer.write("Recall - Batez bestekoa : " + meanR + "\n");
                writer.write("Recall - Desbiderapen estandarra : " + stdevR + "\n");
            }
            System.out.println("Ebaluazioa osatu da:" + estimazioaPath);
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    private static Instances datuakKargatu(String path){
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

    private static void saveInstances(Instances data, String filePath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        System.out.println(filePath);
        saver.setFile(new File(filePath));
        saver.writeBatch();
    }

    private static double avg_kalkulatu(double[] balioak){
        double sum = 0;
        // array-eko balio guztiak gehitu:
        for(double b: balioak){
            sum += b;
        }
        // batura elementu kopuruagatik zatitu batez bestekoa lortzeko:
        return sum / balioak.length;
    }

    private static double stdev_kalkulatu(double[] balioak, double mean){
        double sum = 0;
        // Balio bakoitzaren eta batez bestekoaren arteko aldeak ber bi batu
        for(double b: balioak){
            sum += Math.pow(b - mean,2);
        }
        // alde horien batez bestekoaren erro karratua itzuli:
        return Math.sqrt(sum / balioak.length);
    }
}
