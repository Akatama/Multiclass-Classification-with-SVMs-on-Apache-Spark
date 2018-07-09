# Multiclass-Classification-with-SVMs-on-Apache-Spark
Note: To run the code in the manner prescribed here, you need SBT and Apache Spark. The versions used in creating this project was Apache Spark 2.3.0 and Hadoop 2.9.0. Also note that you will need to have your data in HDFS, and if it is not in the same path as the program in here, you will need to change it. There are also various Spark Settings that you will need to set. How you set these will have a great impact on performance. Once you make those changes, then from the file you project sbt is in, you will need to execute 'sbt package'

To run these programs:
Single computer, all cores:

~/spark-2.1.0/bin/spark-submit --class "org.apache.spark.mllib.classification.OVAJimmy" --master local[*] target/scala-2.11/jimmyproject_2.11-1.0.jar


~/spark-2.1.0/bin/spark-submit --class "org.apache.spark.mllib.classification.OVOJimmy" --master local[*] target/scala-2.11/jimmyproject_2.11-1.0.jar

~/spark-2.1.0/bin/spark-submit --class "org.apache.spark.mllib.classification.CBTSJimmy" --master local[*] target/scala-2.11/jimmyproject_2.11-1.0.jar


Spark Network:

~/spark-2.3.0/bin/spark-submit --class "org.apache.spark.mllib.classification.OVAJimmy" --master spark://192.168.0.45:7077 target/scala-2.11/jimmyproject_2.11-1.0.jar

~/spark-2.3.0/bin/spark-submit --class "org.apache.spark.mllib.classification.OVOJimmy" --master spark://192.168.0.45:7077 target/scala-2.11/jimmyproject_2.11-1.0.jar


~/spark-2.3.0/bin/spark-submit --class "org.apache.spark.mllib.classification.CBTSJimmy" --master spark://192.168.0.45:7077 target/scala-2.11/jimmyproject_2.11-1.0.jar
