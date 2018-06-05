package org.apache.spark.ml.classification

import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl._
import org.apache.spark.ml.tree._


trait DecomposableClassificationModel[FeatureType, M <: ClassificationModel[FeatureType, M]] extends ClassificationModel[FeatureType, M] {

  def transformBySubmodels(dataset: Dataset[_]): DataFrame
}

// AveragedBayesNetClassificationModel

class DecomposableAveragedBayesNetClassificationModel(
    override val uid: String,
    override val models: Array[BayesNetClassificationModel],
    override val numClasses: Int
) extends AveragedBayesNetClassificationModel(uid, models, numClasses)
  with DecomposableClassificationModel[Vector, AveragedBayesNetClassificationModel] {

  def this(m: AveragedBayesNetClassificationModel) = {
    this(m.uid, m.models, m.numClasses) 

    this.setFeaturesCol(m.getFeaturesCol)
  }

  def transformBySubmodels(dataset: Dataset[_]): DataFrame =  {
      
    val predictUDF = udf { (features: Vector) =>

      val modelsPosteriors = getModelsClassPosterior(features)

        val ensemblePosteriors: Array[Double] = Array.range(0, numClasses).map { c =>
          modelsPosteriors.map(p => p(c)).sum
        }

      val modelsPredictions = modelsPosteriors.map(v => v.indexOf(v.max))
      val ensemblePredictions = ensemblePosteriors.indexOf(ensemblePosteriors.max)

      // Vote!
      // val modelsPredictionsSum = modelsPredictions.groupBy(x => x).mapValues(_.length).toVector.sortBy(-_._2)
      // val ensemblePredictions = modelsPredictionsSum(0)._1

      // Return posteriors for brierscore:
      modelsPosteriors.zipWithIndex :+ (ensemblePosteriors, -1)
      // modelsPredictions.zipWithIndex :+ (ensemblePredictions, -1)
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }
}

// BayesNetClassificationModel

class DecomposableBayesNetClassificationModel(
    override val uid: String,
    override val classPriors: ProbTable,
    override val nodes: Array[BncNode],
    override val numClasses: Int
) extends BayesNetClassificationModel(uid, classPriors, nodes, numClasses)
  with DecomposableClassificationModel[Vector, BayesNetClassificationModel] {

  def this(m: BayesNetClassificationModel) = {
        this(m.uid, m.classPriors, m.nodes, m.numClasses) 

        this.setFeaturesCol(m.getFeaturesCol)
  }

  def transformBySubmodels(dataset: Dataset[_]): DataFrame =  {

    val predictUDF = udf { (features: Vector) =>
    
        val posterior = getClassPosterior(features)
        val prediction = posterior.indexOf(posterior.max)

        Array((prediction, -1))
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }
}


// Random Forest

class DecomposableRandomForestClassificationModel(
    override val uid: String,
    private val _trees: Array[DecisionTreeClassificationModel],
    override val numFeatures: Int,
    override val numClasses: Int
) extends RandomForestClassificationModel(uid, _trees, numFeatures, numClasses)
  with DecomposableClassificationModel[Vector, RandomForestClassificationModel] {

  def this(m: RandomForestClassificationModel) = {
    this(m.uid, m.trees, m.numFeatures, m.numClasses) 

    this.setFeaturesCol(m.getFeaturesCol)
  }

  def transformBySubmodels(dataset: Dataset[_]): DataFrame =  {

    val predictUDF = udf { (features: Vector) =>

        val treeVotes = _trees.view.map { tree =>
            tree.rootNode.predictImpl(features).impurityStats.stats
        }

        val ensembleVotes = Array.range(0, numClasses).map { c =>
           treeVotes.map(p => p(c)).sum
        }

        val treePredictions = treeVotes.map( p => p.indexOf(p.max) )
        val ensemblePredictions = ensembleVotes.indexOf(ensembleVotes.max)

        treePredictions.zipWithIndex :+ (ensemblePredictions, -1)
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }
}

// Decision Tree

class DecomposableDecisionTreeClassificationModel(
    override val uid: String,
    override val rootNode: Node,
    override val numFeatures: Int,
    override val numClasses: Int
) extends DecisionTreeClassificationModel(uid, rootNode, numFeatures, numClasses)
  with DecomposableClassificationModel[Vector, DecisionTreeClassificationModel] {

  def this(m: DecisionTreeClassificationModel) = {
    this(m.uid, m.rootNode, m.numFeatures, m.numClasses) 

    this.setFeaturesCol(m.getFeaturesCol)
  }

  def transformBySubmodels(dataset: Dataset[_]): DataFrame =  {

    val predictUDF = udf { (features: Vector) =>

        val votes = rootNode.predictImpl(features).impurityStats.stats.clone()
        val prediction = votes.indexOf(votes.max)

        Array((prediction, -1))
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }
}