package es.jarias.pgm2018.run

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification._

import es.jarias.pgm2018.experiment._
import org.apache.log4j.{Level, Logger, LogManager, PropertyConfigurator}



object EvalDecomp {

  def main(args: Array[String]) {

  implicit def averagedBayesDecomposable(s: AveragedBayesNetClassificationModel) = new DecomposableAveragedBayesNetClassificationModel(s)
  implicit def bayesNetDecomposable(s: BayesNetClassificationModel) = new DecomposableBayesNetClassificationModel(s)
  implicit def randomForestDecomposable(s: RandomForestClassificationModel) = new DecomposableRandomForestClassificationModel(s)
  implicit def decisionTreeDecomposable(s: DecisionTreeClassificationModel) = new DecomposableDecisionTreeClassificationModel(s)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val numArgs = 5
    if (args.length != numArgs )
      throw new Exception(f"Wrong args number ${args.length} required ${numArgs}. Usage: EvalDecomp path outPath filename algorithm parallelism")

    // Get arguments
    val path = args(0)
    val outPath = args(1)
    val name = args(2)
    val algorithm = args(3)
    val parallelism = args(4).toInt

    val outName = name.split("\\.")(0)

    val spark = SparkSession
      .builder()
      .appName(f"Bias Variance $name $algorithm")
      .config("spark.driver.maxResultSize", "4096")
      .getOrCreate()

    val df = spark.read.load( f"$path%s/$name%s" )

    val conf = algorithm.split("-")

    val estimatorName = conf(0)
    val params = conf.tail
      .map(p => p.split("\\."))
      .map {case Array(name, value) => (name, value)}
      .toMap

  val result = 
    estimatorName match {

      case "aode" => 
        val est = 
          new AodeClassifier()

        Decomposition.getBiasVariance(
          df, 
          est, 
          averagedBayesDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "aodeVoting" => 
        val est = 
          new AodeClassifier()

        Decomposition.getBiasVariance(
          df, 
          est, 
          averagedBayesDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "a2de" => 
        val est = 
          new A2deClassifier()

        Decomposition.getBiasVariance(
          df, 
          est, 
          averagedBayesDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "sa2de" => 
        val est = 
          new SA2deClassifier()

        Decomposition.getBiasVariance(
          df, 
          est, 
          averagedBayesDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "naivebayes" => 
        val est = 
          new DiscreteNaiveBayesClassifier()

        Decomposition.getBiasVariance(
          df, 
          est, 
          bayesNetDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "rkdb" => 
        val est = 
          new RandomKDependentBayesClassifier()
            .setK(params("k").toInt)
            .setNumModels(params("numModels").toInt)
            .setSubsamplingRate(0.6)

        Decomposition.getBiasVariance(
          df, 
          est, 
          averagedBayesDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "kdb" => 
        val est = 
          new KDependentBayesClassifier()
            .setK(params("k").toInt)

        Decomposition.getBiasVariance(
          df, 
          est, 
          bayesNetDecomposable,
          discretize = true,
          parallelism = parallelism
        )

      case "decisiontree" =>
        val est = 
          new DecisionTreeClassifier()
                .setMaxDepth(params("depth").toInt)

        Decomposition.getBiasVariance(
          df, 
          est, 
          decisionTreeDecomposable,
          discretize = false,
          parallelism = parallelism
        )

      case "randomforest" =>
        val est = 
          new RandomForestClassifier()
            .setNumTrees(params("trees").toInt)
            .setMaxDepth(params("depth").toInt)

        Decomposition.getBiasVariance(
          df, 
          est, 
          randomForestDecomposable,
          discretize = false,
          parallelism = parallelism
        )
      }

    // val estimator = new RandomForestClassifier()
    //           .setNumTrees(2)
    //           .setMaxDepth(1)

    // val modelsMethodName = getmodelsMethodName(algorithm)

    result
      .withColumn("algorithm", lit(f"$algorithm"))
      .withColumn("db", lit(f"$outName"))
      // .show
    .write
    .format("csv")
    .save(f"$outPath/$algorithm-$outName.csv")
  }


  // Parse the algorithm string and create the estimator
  // Format estimator-param1:value1-param2:value2...
  //
  def getEstimator[
    FeaturesType,
    E <: Classifier[FeaturesType, E, M], 
    M <: ClassificationModel[FeaturesType, M]
  ](algorithm: String): Classifier[FeaturesType, E, M] = {

    val conf = algorithm.split("-")

    val estimator = conf(0)
    val params = conf.tail
      .map(p => p.split("\\."))
      .map {case Array(name, value) => (name, value)}
      .toMap

    estimator match {

      case "aode" => 
        new AodeClassifier()
          .asInstanceOf[Classifier[FeaturesType, E, M]] 

      case "a2de" => 
        new A2deClassifier()
          .asInstanceOf[Classifier[FeaturesType, E, M]] 

      case "naivebayes" => 
        new DiscreteNaiveBayesClassifier()
          .asInstanceOf[Classifier[FeaturesType, E, M]] 

      case "kdb" => 
        new KDependentBayesClassifier()
          .setK(params("k").toInt)
          .asInstanceOf[Classifier[FeaturesType, E, M]] 

      case "decisiontree" =>
        new DecisionTreeClassifier()
              .setMaxDepth(params("depth").toInt)
              .asInstanceOf[Classifier[FeaturesType, E, M]]

      case "randomforest" =>
        new RandomForestClassifier()
              .setNumTrees(params("trees").toInt)
              .setMaxDepth(params("depth").toInt)
              .asInstanceOf[Classifier[FeaturesType, E, M]]

      }

  }

  def getmodelsMethodName(algorithm: String): String = {
      val conf = algorithm.split("-")
      val estimator = conf(0)  

      estimator match {
        case "aode" => 
          "models"

        case "a2de" => 
          "models"

        case "naivebayes" => 
          ""

        case "kdb" => 
          ""

        case "decisiontree" =>
          ""

        case "randomforest" =>
          "trees"

      } 
  }
}