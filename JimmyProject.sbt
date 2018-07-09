name := "JimmyProject"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += "Spark Packages Repo" at "https://dl.bintray.com/spark-packages/maven/"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.1.0",
  "org.apache.spark" %% "spark-sql" % "2.1.0",
  "org.apache.hadoop" % "hadoop-hdfs" % "2.6.0",
  "org.apache.spark" %% "spark-mllib" % "2.1.0"
)
