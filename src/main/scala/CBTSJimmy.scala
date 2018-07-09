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
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

object CBTSJimmy
{
	def main(args: Array[String])
	{
		val ss = SparkSession
		  .builder()
		  .appName("CBTSJimmy")
		  .getOrCreate()
		  import ss.implicits._

		//create spark context here using val sc = ss.sparkContext
		//this stuff just creates RDDs that are RDD[LabeledPoint]
		//val trainTextRDD = ss.sparkContext.textFile("file:///home/hduser/Desktop/JimmyProject/pendigits.tra")
		//val trainRDD = trainTextRDD.map(line => line.split(",").map(_.trim.toDouble)).map(v => LabeledPoint(v(16),Vectors.dense(v.dropRight(1).toArray))).cache()

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

   		//numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double
   		val model = CBTSSVM.train(trainRDD, 500, 4.0, 1E-6, 0.50)


   		val trainTime = System.nanoTime

   		//get the prediction from the extension
   		val predictionAndLabel = testRDD.map(p => (model.predict(p.features), p.label))

   		val trainPrediction = trainRDD.map(p => (model.predict(p.features), p.label))

   		//testRDD.take(10).foreach(println)
   		//predictionAndLabel.take(10).foreach(println)

   		val testTime = System.nanoTime

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
	class CBTSSVMModel(classModels: Array[(SVMModel, Int, Int, Int, Int)]) extends ClassificationModel with Serializable
	{
		//get the index (like for an array/vector)
		//RDDs can be sorted, and so do have an order. This order is used to create the index with .zipWithIndex()
		val broadcastZip = classModels.zipWithIndex


		//this is the proper prediction class
		override def predict(testData: RDD[Vector]): RDD[Double] =
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
		//Maybe write a better prediction
		def predictPoint(testData: Vector, models: Array[((SVMModel, Int, Int, Int, Int), Int)]): Double =
		{
			var index = 0;
			var nextIndex = 0;
			while(nextIndex != -1)
			{
				var prediction = models(index)._1._1.predict(testData)
				if(prediction >= 0.0)
				{
					nextIndex = models(index)._1._2
				}
				else
				{
					nextIndex = models(index)._1._3
				}

				if(nextIndex == -1)
				{
					if(prediction >= 0.0)
						index = models(index)._1._4
					else
						index = models(index)._1._5
				}
				else
				{
					index = nextIndex
				}
			}
			return index;
		}

	}

	//the object that outputs an CBTSSVMModel
	//only trains the models
	object CBTSSVM
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
		def train(input: RDD[LabeledPoint], numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double): CBTSSVMModel = 
		{

			val startTime = System.nanoTime
			val parsedData = input.map(p => p.features).persist()
			val numClusters = 2
			val clusters = KMeans.train(parsedData, numClusters, numIterations)
			val clusterTime = System.nanoTime


			/*
			val WSSSE = clusters.computeCost(parsedData)
			println("Within Set Sum of Squared Errors = " + WSSSE)
			*/

			val clusterTimeTaken = (clusterTime - startTime) * 1E-9
			println("Cluster time taken: " + clusterTimeTaken)

			//find the number of classes
			val numClass = input.map(_.label).max.toInt

			//this is a loop. We are trying to figure out which cluster each class belongs to
			val clusterClass = (0 to numClass).map
			//needs to map based on classID, which is the Label of the LabeledPoint
			{classID =>
				val inputByClass = input.filter(_.label==classID)
				val inputByFeatures = inputByClass.map(p => p.features).persist()
				val prediction = clusters.predict(inputByFeatures)

				val test = prediction.map(x => (x, 1L)).reduceByKey(_ + _)

				val popularCluster = prediction.map(x => (x, 1L)).reduceByKey(_ + _).reduce((x, y) => if(x._2 > y._2) x else y)
				val lessPopularCluster = prediction.map(x => (x, 1L)).reduceByKey(_ + _).reduce((x, y) => if(x._2 < y._2) x else y)
				//the third element is the SSE of the class based on the cluster it belongs to
				(classID, popularCluster._1, lessPopularCluster._2 * lessPopularCluster._2)


			}.toArray

			//seperate out the cluser1 values
			val cluster1 = clusterClass.filter(_._2 == 1).map{case (classID, cluster, error) => (classID, error)}.sortBy(_._2)
			.map{case (classID, error) => classID}
			//seperate out the cluster0 values (maybe don't need)
			val cluster0 = clusterClass.filter(_._2 == 0).map{case (classID, cluster, error) => (classID, error)}.sortBy(_._2)
			.map{case (classID, error) => classID}


			val firstTrain = System.nanoTime
			val sepTimeTaken = (firstTrain - clusterTime) * 1E-9


			println("Initial Seperation Time Taken" + sepTimeTaken)
			//filtering the results based on the array cluster1
			//might be better to check to see which array is larger to use to make the dataset
			//so that the side with the more values is the 1.0 value and the side with less values is 0.0
			val firstInput = input.map { case LabeledPoint(label, features) =>
				LabeledPoint(if (cluster1 contains label) 1.0 else 0.0, features)}.cache()

			val firstModel = SVMWithSGD.train(firstInput, numIterations, stepSize, regParam, miniBatchFraction)

			val trainTimeFirst = System.nanoTime

			val firstTimeTaken = (trainTimeFirst - firstTrain) * 1E-9

			println("First model took: " + firstTimeTaken)

			firstModel.clearThreshold()
			//create the array we will use for our tree
			//note that the way this works is [SVMMODEl, leftside Index, rightside Index, posLabel, negLabel]
			//-1 in leftside result or rightside result tells us that there are no more SVMs to look at on that side
			//-1 in posLabel or negLabel says that this SVM does not result in a classification decision
			//if posLabel or negLabel has a different value than -1, than that is what the classification decision results in

			//rightIndex is caluclated by figuring that the left tree needs to create n-1 SVMs
			//and therefore, the rightIndex would be (n-1)+1 for the right node
			val rightIndex = cluster1.length
			//takes n-1 SVMs, but our numClass val ignores that there are 10 classes (value says 9)
			val treeSVMs = new Array[(SVMModel, Int, Int, Int, Int)](numClass)
			treeSVMs(0) = (firstModel, 1, rightIndex, -1, -1)
			//holds the uncreated branches
			val leftInput = input.filter(cluster1 contains _.label)
			val rightInput = input.filter(cluster0 contains _.label)
			val index = 1

			val posModels = trainHelper(treeSVMs, leftInput, index, cluster1, numIterations, stepSize, regParam, miniBatchFraction)
			val negModels = trainHelper(posModels, rightInput, rightIndex, cluster0, numIterations, stepSize, regParam, miniBatchFraction)

			val modelsTime = System.nanoTime

			val modelsTimeTaken = (modelsTime - trainTimeFirst) * 1E-9

			val avgTimeTaken = (modelsTimeTaken + firstTimeTaken) / 9

			println("Remaining modes took: " + modelsTimeTaken)

			println("Avg time taken per model: " + avgTimeTaken)
			
			//negModels.foreach{case (model, posIndex, negIndex, posLabel, negLabel) => println(s"( $posIndex, $negIndex, $posLabel, $negLabel )")}

			//create an SVMOVAModel from the classModels
			new CBTSSVMModel(negModels)
		}

		def trainHelper(models: Array[(SVMModel, Int, Int, Int, Int)], data: RDD[LabeledPoint], index: Int, 
			classes: Array[Int], numIterations: Int, stepSize: Double, regParam : Double, miniBatchFraction: Double) 
			: Array[(SVMModel, Int, Int, Int, Int)] =
		{
			//train one last SVM
			if(classes.length == 2)
			{
				val class1 = classes(0)
				val dataProjection = data.map { case LabeledPoint(label, features) =>
					LabeledPoint(if (class1 == label) 1.0 else 0.0, features)}.cache()
				val newModel = SVMWithSGD.train(dataProjection, numIterations, stepSize, regParam, miniBatchFraction)
				newModel.clearThreshold()
				models(index) = (newModel, -1, -1, class1, classes(1))

				return models

			}
			else
			{
				//split data and classes in half
				val splitPoint = if(classes.length % 2 == 0) classes.length / 2 else (classes.length / 2) + 1
				val splitClass = classes.splitAt(splitPoint)
				val class1 = splitClass._1
				val class0 = splitClass._2

				val dataProjection = data.map { case LabeledPoint(label, features) =>
					LabeledPoint(if (class1 contains label) 1.0 else 0.0, features)}.cache()

				val posIndex = if(class1.length > 1) index+1 else -1
				val negIndex = if(class0.length > 1) class1.length + index else -1
				val posLabel = if(class1.length > 1) -1 else class1(0)
				val negLabel = if(class0.length > 1) -1 else class0(0)

				val newModel = SVMWithSGD.train(dataProjection, numIterations, stepSize, regParam, miniBatchFraction)

				newModel.clearThreshold()
				models(index) = (newModel, posIndex, negIndex, posLabel, negLabel)

				val data1 = data.filter(class1 contains _.label)
				val data0 = data.filter(class0 contains _.label)

				val posModels = trainHelper(models, data1, posIndex, class1, numIterations, stepSize, regParam, miniBatchFraction)
				if(class0.length > 1)
				{
					return trainHelper(posModels, data0, negIndex, class0, numIterations, stepSize, regParam, miniBatchFraction)
				}
				return posModels
			}
			
		}
	}
}