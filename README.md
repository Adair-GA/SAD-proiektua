
# Tweet Sentiment SVM Sailkatzailea

## Taldekideak
Laura Meihua Caballero Pascual  
Unai Rodríguez Cubillo  
Adair Gondan Alonso.

## Proiektua hedatzeko instrukzioak
Hurrengo .jar fitxategiak exekutatu behar dira:
1. Aurreprozesamendua:

```bash
 java -jar Aurreprozesamendua.jar <train.csv> <dev.csv> <test.csv> <arff/path/> <hiztegia.arff>
```
2. Inferentzia
Inferentzia egiterako orduan bi aukera posible daude:

Train eta dev bakarrik existitzen dira csv fitxategi independente bezala:
```bash
java -jar Inferentzia.jar <train.arff> <dev.arff> <train_dev.arff> <train.csv> <dev.csv> <train_dev.csv> <hiztegia.arff> <modelPath> <paramSet.txt> <kalitate_estimazioa.txt>
```
CSV fitxategi bat existitzen da datu sorta osoarekin (train + dev):
```bash
java -jar Inferentzia2.jar <train.arff> <dev.arff> <train_dev.arff> <train_dev.csv> <hiztegia.arff> <modelPath> <paramSet.txt> <kalitate_estimazioa.txt>
```
3. Sailkapenak
```bash
java -jar Iragarpenak.jar <test_blind.arff><.model><iragarpenak.txt>
```

## Aurre-baldintzak / Ondorengo-baldintzak

1. Aurreprozesamendua
- CSV fitxategiak hurrengo patroia jarraitu behar dute:
    "Topic, Sentiment, TweetID, TweetDate, TweetText"
- Emaitzak adierazitako path-an gordeko dira, hurrengo izenekin:
```bash
train + _as_BoW.arff, dev + _as_BoW.arff, test + _as_BoW.arff, hiztegia + egokitua.arff
```

2. Inferentzia
- Datuak aurreprazesamenduan adierazitako egitura jarraitu behar dute
- Klase atributua 0 posizioan egon behar da
- Eredua, parametro optimoak eta kalitatearen estimazioa adierazitako path-etan gordeko dira

3. Sailkapenak
- Test aurreprazesamenduan adierazitako egitura jarraitu behar du
- Eredua entrenatzeko erabili diren datuak aurreprazesamenduan adierazitako egitura jarraitu behar dute

## Menpekotasunak
- mtj-1.0.4.jar:
 JDK1.5 - JDK8 

 JDK9-tik aurrera erabiltzea arazoak sor ditzake
 - weka.jar
WEKA 3.8.x Java 8-rekin exekutatzeko diseinatuta dago.

Bertsio berriagoetan funtziona dezake (Java 11, 17), baina ofizialki ez dute bermatzen Java 8 baino bertsio bateragarritasun osoa.

## Erabilpenaren adibidea
Hurrengo datuak erabiliko dira: 
```bash
tweetSentiment.train.csv, tweetSentiment.dev.csv, tweetSentiment.test_blind.csv
```
1. Aurreprozesamendua:
```bash
java -jar Aurreprozesamendua.jar "Datuak/CSV/tweetSentiment.train.csv" "Datuak/CSV/tweetSentiment.dev.csv" "Datuak/CSV/tweetSentiment.test_blind.csv" "Datuak/ARFF/" "Datuak/ARFF/hiztegia.arff"
```
2. Inferentzia:
Suposatuko da erabiltzaileak ez daukala datu sorta osoaren CSV fitxategia, beraz programak hau automatikoki sortuko du train.csv eta dev.csv batuz
```bash
java -jar Inferentzia.jar "Datuak/ARFF/train_as_BoW.arff" "Datuak/ARFF/dev_as_BoW.arff" "Datuak/ARFF/train_dev.arff" "Datuak/CSV/tweetSentiment.train.csv" "Datuak/CSV/tweetSentiment.dev.csv" "Datuak/CSV/tweetSentiment.train_dev.csv" "Datuak/ARFF/dictionary.arff" "Datuak/model" "Datuak/ps.txt" "Datuak/Emaitzak.txt"
``` 
3. Sailkapenak
```bash
java -jar Iragarpenak.jar "Datuak/ARFF/test_blind_as_BoW.arff" "Emaitzak/Ereduak/Eredu_Optimoa_RHO.model" "Emaitzak/Sailkapenak/Eredu_optimoa_predictions.txt"
```

