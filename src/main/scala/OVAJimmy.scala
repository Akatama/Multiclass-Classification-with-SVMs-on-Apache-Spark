package org.apache.spark.mllib.classification

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object OVAJimmy
{
	def main(args: Array[String])
	{
		val ss = SparkSession
		  .builder()
		  .appName("OVAJimmy")
		  .getOrCreate()
		  import ss.implicits._
		//this stuff just creates RDDs that are RDD[LabeledPoint]
		//val trainTextRDD = ss.sparkContext.textFile("file:///home/hduser/Desktop/JimmyProject/pendigits.tra")
		//val trainRDD = trainTextRDD.map(line => line.split(",").map(_.trim.toDouble)).map(v => LabeledPoint(v(16),Vectors.dense(v.dropRight(1).toArray))).cache()

		//val testTextRDD = ss.sparkContext.textFile("file:///home/hduser/Desktop/JimmyProject/pendigits.tes")
		//val testRDD = testTextRDD.map(line => line.split(",").map(_.trim.toDouble)).map(v => LabeledPoint(v(16),Vectors.dense(v.dropRight(1).toArray)))
		

		//load the data. Note that the Minst8m dataset for this project is in LibSVM format
		//can be found at: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist8m
		//Also note that it is being loaded in from HDFS. Depending on where you are running this, some of the text might be different
   		val loadTime = System.nanoTime

   		val dataPath = "hdfs://cloudlabMaster:9000/user/hduser/Jimmy/Data/mnist8m/mnist8m.scale"

   		println("Loading Data")

   		val dataRDD : RDD[LabeledPoint] = MLUtils.loadLibSVMFile(ss.sparkContext, dataPath)
   		val splitTime = System.nanoTime

   		println("Done Loading Data, splitting it")


   		val splits = dataRDD.randomSplit(Array(0.5, 0.5))


   		val (trainRDD, testRDD) = (splits(0), splits(1))

		println("Data split, starting training")




   		//start recording time
   		val startTime = System.nanoTime

   		//This calls SMOVA, t is an extension of SVMWithSGD
   		//numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double
   		val model = SVMOVA.train(trainRDD, 250, 4.0, 0.01, 0.25)

   		val trainTime = System.nanoTime

   		//get the prediction from the extension
   		val predictionAndLabel = testRDD.map(p => (model.predict(p.features), p.label))

   		val trainPrediction = trainRDD.map(p => (model.predict(p.features), p.label))

   		

   		//calculate accuracy
   		//val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testRDD.count()

   		//show the score and the actual label
   		/*val scoreAndLabels = trainRDD.map
   		{
   			point => val score = model.predict(point.features)
   			(score, point.label)
   		}*/

   		// Instantiate metrics object
		val metrics = new MulticlassMetrics(predictionAndLabel)

		val testTime = System.nanoTime

		val trainMetrics = new MulticlassMetrics(trainPrediction)


		val loadingTimeTaken = (splitTime - loadTime) * 1E-9
		val splittingTimeTaken = (startTime - splitTime) * 1E-9
   		val trainingTimeTaken = (trainTime - startTime) * 1E-9
   		val testingTimeTaken = (testTime - trainTime) * 1E-9




   		println(/*"\n\n\nConfusion matrix:\n" + metrics.confusionMatrix + */
   			"\n\nTrain accuracy: " + trainMetrics.accuracy +
   			"\n\nAccuracy: " + metrics.accuracy +
   			"\n\nTime taken for loading data: " + loadingTimeTaken + " seconds " +
   			"\n\nTime take for splitting data: " + splittingTimeTaken + " seconds " +
   			"\n\nTime taken for training: " + trainingTimeTaken + " seconds " +
   			"\n\nTime taken for testing: " + testingTimeTaken + " seconds")

   		println("\n\n")

   		val labels = metrics.labels
		labels.foreach { l =>
  			println(s"Precision($l) = " + metrics.precision(l))
		}

		println("\n\n")

		// Recall by label
		labels.foreach { l =>
  			println(s"Recall($l) = " + metrics.recall(l))
		}

		println("\n\n")

		// False positive rate by label
		labels.foreach { l =>
  			println(s"FPR($l) = " + metrics.falsePositiveRate(l))
		}

		println("\n\n")

		// F-measure by label
		labels.foreach { l =>
  			println(s"F1-Score($l) = " + metrics.fMeasure(l))
		}

		println("\n\n")

		ss.stop()

	}

	//this class extends SVMModel
	class SVMOVAModel(classModels: Array[(SVMModel, Int)]) extends ClassificationModel with Serializable
	{
		//get the index (like for an array/vector)
		//RDDs can be sorted, and so do have an order. This order is used to create the index with .zipWithIndex()


		//this is the proper prediction class
		override def predict(testData: RDD[Vector]): RDD[Double] =
		{
			
			//mapPartitions() provides for the initialization to be done once per worker task/thread/partition instead of once per RDD data element
			//also, it is apparently better for heavyweight initialization because it does not do it for each row, instead each partition
			testData.mapPartitions
			{
				test => test.map(pred => predictPoint(pred, classModels))
			}

		}

		//this one needs to be overridden even though it is unused because ClassificationModel is a trait (abstract class)
		override def predict(testData: Vector): Double = predictPoint(testData, classModels)

		//This predicts a single point from an already trained model
		def predictPoint(testData: Vector, models: Array[(SVMModel, Int)]): Double =
		models
		//gets the prediction of the testData (a number between 0 and 1) and puts it in a tuple with the class classNumber
		.map { case (classModel, classNumber) => (classModel.predict(testData), classNumber)}
		//gets the max prediciton score (ex: if class 1 has a score of .5 and class 2 has a score of .8 and there are no other classes )
		//class 2 wins
		.maxBy { case (score, classNumber) => score}
		//takes the second tuple element (our prediction for what class it belongs too because it had the most positive result for a class)
		._2

	}

	//the object that outputs an SVMOVAModel
	//only trains the models
	object SVMOVA
	{
	/**
   	* Train a Multiclass SVM model given an RDD of (label, features) pairs,
   	* using One-vs-Rest method - create one SVMModel per class with SVMWithSGD.
   	*
 	* @param input RDD of (label, array of features) pairs.
   	* @param numIterations Number of iterations of gradient descent to run.
   	* @param stepSize Step size to be used for each iteration of gradient descent.
   	* @param regParam Regularization parameter.
  	* @param miniBatchFraction Fraction of data to be used per iteration.
   	*/
		def train(input: RDD[LabeledPoint], numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double): SVMOVAModel = 
		{
			println("starting training")
			
			//find the number of classes
			val numClass = input.map(_.label).max.toInt
			//this is a loop. Iterating through the RDD[LabeledPoint]. All our models will be put into this classModels variable
			var totalTime = 0.0
			val classModels = (0 to numClass).map
			//needs to map based on classID, which is the Label of the LabeledPoint
			{classID =>
			println("class: " + classID)
			val startTime = System.nanoTime
			//maps the label from the classNum to 1.0 if label == classID, otherwise maps it to 0.0
			//this is necessary because the labels in SVMs in Apache Spark need to be scaled from 0.0 to 1.0
			//as SVMs only technically do binary classification. We are creating a set for Multiclass classificaiton
			val inputProjection = input.map { case LabeledPoint(label, features) =>
			LabeledPoint(if (label == classID) 1.0 else 0.0, features)}.persist(StorageLevel.MEMORY_ONLY)
			//train the model for that particular class ID
			val model = SVMWithSGD.train(inputProjection, numIterations, stepSize, regParam, miniBatchFraction)
			val trainTime = System.nanoTime
			val timeTaken = (trainTime - startTime) * 1E-9
			println("Time taken: " + timeTaken)
			totalTime += timeTaken
			inputProjection.unpersist(true)
			model.clearThreshold()
			//create an array of these models
			(model, classID)

			}.toArray
			//create an SVMOVAModel from the classModels
			println("\n\nAVG time taken per classifier: " + totalTime/10)
			new SVMOVAModel(classModels)
		}
	}
}