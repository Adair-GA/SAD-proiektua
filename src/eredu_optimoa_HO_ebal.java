import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
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

public class eredu_optimoa_HO_ebal {
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
   
            Resample resample = new Resample();
            resample.setRandomSeed(1); 
            resample.setNoReplacement(true);
            resample.setInvertSelection(false);
            resample.setSampleSizePercent(70);
            resample.setInputFormat(data);

            // train datuak sortu:
            Instances trainData = Filter.useFilter(data, resample);
            trainData.setClassIndex(0);

            // Atributu hautapena train:
            String trainPath = "Ebaluazio_eredu_optimoa/HO_train.arff";
            String hizPath = "Ebaluazio_eredu_optimoa/HO_hiztegi.arff";
            saveInstances(trainData, trainPath);
            arffToBoW.arffBoW(trainPath, hizPath);
            Instances trainBoW = datuakKargatu(trainPath.replace(".arff", "_as_BoW.arff"));

            // dev multzoa sortu:
            Resample resample2 = new Resample();
            resample2.setRandomSeed(1); 
            resample2.setNoReplacement(true);
            resample2.setInvertSelection(true);
            resample2.setSampleSizePercent(70);
            resample2.setInputFormat(data);
            Instances devData = Filter.useFilter(data, resample2);
            devData.setClassIndex(0);

            // Atributu hautapena dev:
            String devPath = "Ebaluazio_eredu_optimoa/HO_dev.arff";
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
            double recallMin = eval.recall(minClassIndex);
            double fMeasureMin = eval.fMeasure(minClassIndex);
            double acc = eval.pctCorrect();
            double fMeasure = eval.weightedFMeasure();
            double precision = eval.weightedPrecision();
            double recall = eval.weightedRecall();

            // Emaitzak gorde:
            try(PrintWriter writer = new PrintWriter(new FileWriter(estimazioaPath))){
                writer.println("\n--- Nahasmen matrizea ---");
				writer.println(eval.toMatrixString());
		        
                writer.println(eval.toClassDetailsString());

                writer.println("Recall Klase minoritarioa: " + recallMin + "\n");
                writer.println("F-Measure Klase minoritarioa: " + fMeasureMin + "\n\n");

                writer.println("Accuracy: " + acc + "\n");
                writer.println("F-Measure: " + fMeasure + "\n");
                writer.println("Precision: " + precision + "\n");
                writer.println("Recall: " + recall + "\n");
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
}