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

object OVOJimmy
{
	def main(args: Array[String])
	{
		val ss = SparkSession
		  .builder()
		  .appName("OVOJimmy")
		  .getOrCreate()
		  import ss.implicits._
		//this stuff just creates RDDs that are RDD[LabeledPoint]
		/*val trainTextRDD = ss.sparkContext.textFile("file:///home/hduser/Desktop/JimmyProject/pendigits.tra")
		val trainRDD = trainTextRDD.map(line => line.split(",").map(_.trim.toDouble)).map(v => LabeledPoint(v(16),Vectors.dense(v.dropRight(1).toArray))).cache()

		val testTextRDD = ss.sparkContext.textFile("file:///home/hduser/Desktop/JimmyProject/pendigits.tes")
		val testRDD = testTextRDD.map(line => line.split(",").map(_.trim.toDouble)).map(v => LabeledPoint(v(16),Vectors.dense(v.dropRight(1).toArray)))*/
		
		//these commented out lines of code will be useful in the future when using data that is in LibSVM format

		/*
		String pathTra = "file:///pendigits.tra"
   		String pathTes = "file:///pendigits.tes"
   		
   		JavaRDD<LabeledPoint> dataTra = MLUtils.loadLibSVMFile(sc, pathTra)
   		JavaRDD<LabeledPoint> dataTes = MLUtils.loadLibSVMFile(sc, pathTes)  
   		*/

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

   		//This calls SVMOVO it is an extension of SVMWithSGD
   		//numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double
   		val model = SVMOVO.train(trainRDD, 400, 4.0, 1E-5, 0.5)

   		print("\n\n")


   		val trainTime = System.nanoTime

   		//get the prediction from the extension
   		val predictionAndLabel = testRDD.map(p => (model.predict(p.features), p.label))
   		val trainPrediction = trainRDD.map(p => (model.predict(p.features), p.label))

   		val testTime = System.nanoTime

   		//calculate accuracy
   		//val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testRDD.count()

   		//show the score and the actual label
   		/*val scoreAndLabels = trainRDD.map
   		{
   			point => val score = model.predict(point.features)
   			(score, point.label)
   		}//._1*/

   		//predictionAndLabel.collect().foreach(println)

   		// Instantiate metrics object
   		
		val metrics = new MulticlassMetrics(predictionAndLabel)

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
	class SVMOVOModel(classModels: Array[(SVMModel, Int, Int)]) extends ClassificationModel with Serializable
	{
		//get the index (like for an array/vector)
		//RDDs can be sorted, and so do have an order. This order is used to create the index with .zipWithIndex()
		val broadcastZip = classModels.zipWithIndex


		//this is the proper prediction class
		override def predict(testData: RDD[Vector]): RDD[(Double)] =
		{
			//broadcoast the index so it can be put back together properly when everything is recieved
			val broadcast = testData.sparkContext.broadcast(broadcastZip)
			
			//mapPartitions() provides for the initialization to be done once per worker task/thread/partition instead of once per RDD data element
			//also, it is apparently better for heavyweight initialization because it does not do it for each row, instead each partition
			testData.mapPartitions
			{
				//using an iterator on the broadcast value(s)
				//these are the classes
				iter => val w = broadcast.value
				//for each broadcast.value (which is based on the Index of the input RDD)
				//call predictPoint
				iter.map(v => predictPoint(v, w))
			}

		}

		//this one needs to be overridden even though it is unused because ClassificationModel is a trait (abstract class)
		override def predict(testData: Vector): Double = predictPoint(testData, broadcastZip)

		//This predicts a single point from an already trained model
		def predictPoint(testData: Vector, models: Array[((SVMModel, Int, Int), Int)]): Double =
		{
			val scoredModels = models.map { case ((model, posLabel, negLabel), broadcast) => (posLabel.toDouble, negLabel.toDouble, model.predict(testData))}
			val filteredModelsPos = scoredModels.filter(_._3>=0.0).map{case (posLabel, negLabel, score) => (posLabel, score)}
			val filteredModelsNeg = scoredModels.filter(_._3< 0.0).map{case (posLabel, negLabel, score) => (negLabel, score)}
			val filteredModels = filteredModelsPos.union(filteredModelsNeg)

			val countModels = filteredModels.groupBy(_._1).map(x => (x._1, x._2.size))
			//countModels.foreach{case (posLabel, count) => println(s"( $posLabel, $count )")}
			//println(countModels.mkString(" "))
			//println("\n\n")
			countModels.maxBy { case (posLabel, count) => count}
			//takes the second tuple element (our prediction for what class it belongs too because it had the most positive result for a class)
			._1
		}

	}

	//the object that outputs an SVMOVOModel
	//only trains the models
	object SVMOVO
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
		
		def train(input: RDD[LabeledPoint], numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double): SVMOVOModel = 
		{
			//find the number of classes
			val numClass = input.map(_.label).max.toInt
			//this is a loop. Iterating through the RDD[LabeledPoint]. All our models will be put into this classModels variable
			var totalTime = 0.0
			val classModels = (0 to numClass-1).map
			//I apparently need to use two maps here in order to make the n(n-1)/2 SVM classifiers
			//Note the fact that we have posLabel and negLabel
			{posLabel =>
				(posLabel+1 to numClass).map
				{negLabel =>
				//filters out all classes that aren't equal to posLabel or negLabel
				//then unions them into one RDD
				//Might be possible to do this in one class, with the benefit of not having to use union
				//(look into making a function that you can use inside the filter function)
				println(posLabel + " vs " + negLabel)
				val startTime = System.nanoTime
				val posData = input.filter(_.label==posLabel)
				val negData = input.filter(_.label==negLabel)

				val filteredData = posData.union(negData)


				val inputProjection = filteredData.map { case LabeledPoint(label, features) =>
				LabeledPoint(if (label == posLabel) 1.0 else 0.0, features)}.persist()
				//train the model for those particular class IDs

				val model = SVMWithSGD.train(inputProjection, numIterations, stepSize, regParam, miniBatchFraction)
				val trainTime = System.nanoTime
				val timeTaken = (trainTime - startTime) * 1E-9
				println("Time taken: " + timeTaken)
				totalTime += timeTaken
				inputProjection.unpersist(true)
				model.clearThreshold()

				//create an array of these models
				(model, posLabel, negLabel)

				}.toArray
			}.toArray

			//as a result of the nested loop, I was getting the inner array to be IndexedSeq, as that is the type for the loop
			//to get rid of that, I used .toArray at the end of both looops
			//then I used flatten to make the Array[Array] into just an Array
			//there might be a better way to go about this
			println("\n\nAVG time taken per classifier: " + totalTime/45)
			val classModelsFlat = classModels.flatten
			//create an SVMOVOModel from the classModels
			new SVMOVOModel(classModelsFlat)
		}
	}
}