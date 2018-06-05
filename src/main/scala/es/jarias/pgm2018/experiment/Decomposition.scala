package es.jarias.pgm2018.experiment

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, QuantileDiscretizer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.{Classifier, ClassificationModel, DecomposableClassificationModel}
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel, DecisionTreeClassificationModel}
import org.apache.spark.ml.classification.AodeClassifier
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification._

import scala.collection.parallel._
import org.apache.log4j.{Level, Logger, LogManager, PropertyConfigurator}

object Decomposition {

  def getBiasVariance[
    FeaturesType,
    E <: Classifier[FeaturesType, E, M], 
    M <: ClassificationModel[FeaturesType, M],
    DM <: DecomposableClassificationModel[FeaturesType, M]
    ](
      df: DataFrame, 
      estimator: Classifier[FeaturesType, E, M], 
      converter: M => DM,
      discretize: Boolean = true,
      parallelism: Int = 1): DataFrame = {

    val log = LogManager.getLogger("experiment")
    log.setLevel(Level.DEBUG)
    log.debug("Start")

    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext 
    import spark.implicits._

    val label = df.columns.last

    log.debug("Clean data pipeline (cached)")
    val labelIndexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("indexedLabel")

    // val pipClean = new Pipeline().setStages(Array(labelIndexer))

    // val dfPreprocessed = pipClean
    //   .fit(df)
    //   .transform(df)
    //   .withColumn("id", monotonically_increasing_id)
    //   .cache()

    log.debug("Create preprocessing pipeline")

    // val featuresDiscrete = df.schema.toArray.init
    //   .filter { case StructField(name, fieldType, _, _) => (fieldType != DoubleType) && (fieldType != FloatType)  && (fieldType != StringType)}
    //   .map { case StructField(name, fieldType, _, _) => name }

    val featuresString = df.schema.toArray.init
      .filter { case StructField(name, fieldType, _, _) => (fieldType == StringType) }
      .map { case StructField(name, fieldType, _, _) => name }

    val featuresContinuous = df.schema.toArray.init
      .filter { case StructField(name, fieldType, _, _) => (fieldType == DoubleType) || (fieldType == FloatType) || (fieldType == IntegerType) || (fieldType == LongType)}
      .map { case StructField(name, fieldType, _, _) => name }


    val featuresIndexed = featuresString.map(feat => f"${feat}_indexed")
    val featuresDiscretized = featuresContinuous.map(feat => f"${feat}_disc")

    val indexers = featuresString.map(feat =>
     new StringIndexer()
      .setInputCol(feat)
      .setOutputCol(f"${feat}_indexed") 
    ) 

    val discretizer = new QuantileDiscretizer()
      .setInputCols(featuresContinuous)
      .setOutputCols(featuresDiscretized)
      .setNumBuckets(4)

    val inputFeatures =
      if (discretize)
        featuresDiscretized ++ featuresIndexed 
      else
        featuresContinuous ++ featuresIndexed

    val assembler = new VectorAssembler()
      .setInputCols(inputFeatures)
      .setOutputCol("features")

    val featureIndexer = new VectorIndexer()
      .setInputCol(assembler.getOutputCol)
      .setOutputCol("indexedFeatures")
      .setMaxCategories(15) // features with > 4 distinct values are treated as continuous.

    // Leak
    val pipClean = new Pipeline().setStages(indexers ++ Array(labelIndexer, discretizer, assembler, featureIndexer))

    val dfPreprocessed = pipClean
      .fit(df)
      .transform(df)
      .withColumn("id", monotonically_increasing_id)
      .cache()

    // val pipPreStages = 
    // if (discretize)
    //   Array(discretizer, assembler, featureIndexer)
    // else
    //   Array(assembler, featureIndexer)

    log.debug("Start Repeated Cross Validation")
    // val pipPre = new Pipeline().setStages(pipPreStages)

    val schema = dfPreprocessed.schema
    val schemaPost = StructType(Vector(
      StructField("id", LongType, false), 
      StructField("model", IntegerType, false), 
      StructField("label", DoubleType, false), 
      StructField("prediction", IntegerType, false)
    ))

    val seeds = Vector(3592, 9806, 4240, 8439, 1594, 3098, 1187, 5893, 5712, 7573)

    val repeats = seeds.zipWithIndex.flatMap { case (seed, repeat) => 
      MLUtils.kFold(dfPreprocessed.rdd, 3, seed)
        .zipWithIndex
        .map{ case ((training, validation), fold) => (training, validation, repeat, fold)}
    }

    val repeatsPar = repeats.par
    repeatsPar.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(parallelism))

    val evaluations = repeatsPar.map { case (training, validation, repeat, fold) => {

      log.debug(f"Starting Repeat repeat: $repeat,  fold: $fold")

      val trainingDataset = spark.createDataFrame(training, schema)
      
      // val pre = pipPre.fit(trainingDataset)
      val model = estimator
        .setLabelCol(labelIndexer.getOutputCol)
        .setFeaturesCol(featureIndexer.getOutputCol)
        // .fit(pre.transform(trainingDataset))
        .fit(trainingDataset)
      
      val validationDataset = spark.createDataFrame(validation, schema)
      // val validationPreDataset = pre.transform(validationDataset).cache()

      converter(model)
        // .transform(validationPreDataset)
        .transformBySubmodels(validationDataset)
        .withColumn("predictionByModels", explode($"prediction"))
        .select($"id",  $"predictionByModels._2".as("model"), $"indexedLabel".as("label"), $"predictionByModels._1".as("prediction"))
        
        .localCheckpoint(eager = true)
        .rdd

    } }

    log.debug("Finished crossval")

    val rddEvaluations = sc.union(evaluations.toVector).cache

    val dfEval = spark.createDataFrame(rddEvaluations, schemaPost)
      .groupBy("id", "model")
      .agg(first("label").as("label"), collect_list("prediction").as("prediction"))

    val mode: Seq[Int] => Double = _.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
    val modeUdf = udf(mode)

    val error = (groundTruth: Double, predictions: Seq[Int]) => (predictions.map( p => if (p == groundTruth) 0 else 1 ).sum.toDouble / predictions.length)
    val errorUdf = udf(error)

    val bias = (groundTruth: Double, central: Double, error: Double) => if (central == groundTruth) 0.0 else error
    val biasUdf = udf(bias)

    val variance = (groundTruth: Double, central: Double, error: Double) => if (central == groundTruth) error else 0.0
    val varianceUdf = udf(variance)

    dfEval
      .withColumn("central", modeUdf(col("prediction")))
      .withColumn("error", errorUdf(col("label"), col("prediction")))
      .withColumn("bias", biasUdf(col("label"), col("central"), col("error")))
      .withColumn("variance", varianceUdf(col("label"), col("central"), col("error")))
      .groupBy("model")
      .agg(avg("error"), avg("bias"), avg("variance"))
  }
}