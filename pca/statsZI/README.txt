1) Download JSAP-2.1.jar from http://www.martiansoftware.com/jsap/ and place in statsZI folder.

2) From within the statsZI folder, run the following two commands:

javac -d . -cp JSAP-2.1.jar statistics/*.java utility/*.java
jar cfm statsZI.jar manifest.txt utility/*.class statistics/*.class

3) Test with:

java -jar statsZI.jar --help

java -jar -Xmx5G statsZI.jar --beginDemo=0 --endDemo=1 --numPerDemo=1 --msmsFolder=example/data/ --statsFolder=example/stats/