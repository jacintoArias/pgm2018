# Bayesian Network Classifiers Under the Ensemble Perspective
### Supplementary data and code

This work is currently under review for the PGM2018 conference.


## Source Code and reproducibility

The experiments have been carried out using Apache Spark and are programmed using the scala API. 

A spark cluster is needed to reproduce the results, however your can launch the provided dockerized example that may be suitable for small datasets.

### Dependencies

This work depends on the [spark-bnc package](http://github.com/jacintoarias/spark-bnc) and Apache Spark core, SQL and ml libraries.

### Building

This project is build with sbt. A dockerized script is available to build locally without installing any dependency (just docker!), see `/scripts/build.sh`.

If you have experience with `sbt` just use your prefered method.

### Running

#### Dockerized

We provide a dockerized spark distribution along a proper run script to launch the experiments, see `scripts/runDecomposition.sh`. 

Note: Only the environment is dockerized so you have to build the sources before running the software.

#### On Apache Spark

I am assuming that you have working experience with Apache Spark. Generate the `.jar` using the build instructions and run the package using the main class `org.es.jarias.pgm2018.run.EvalDecomp`.

You will need to generate and pass along the `spark-bnc` distribution jar to spark using the `--jars` option, taking a look ath the `scripts/runDecomposition.sh` might help develop your own. The `spark-bnc` jar will be available under the `.caching` directory if you use our dockerized build script.

### Configuration

The main class executes the bias and variance error decomposition experiment described in the paper. For that it takes an input dataset (in parquet format) and outputs a csv file with the following format:

`model, error, bias, variance, algorithm, database`

where model is a numeric identifier of the ensemble submodel (-1 for the ensemble), error bias and variance the numeric value for such metric and algorithm and database the names for the corresponding input.

The main class (and running script) receive the following parameters:

- `inputPath`: folder where the data rests (can be HDFS)
- `outputPath`: folder where the results will ne written (can be HDFS)
- `filename`: the name of the database in the input path. Must be a parquet data file (the name will be appendend in results).
- `algorithm`: the algorithm to evaluate, currently the program can configure a set of algoriths (described below)
- `parallelism`: the program performs each repeat and fold of the experiment concurrently you can controll the level of parallelism.

Available algorithms, some can be parametrized by swaping {Type} with an actual literal value:

- `naivebayes`
- `kdb-k.{Int}`
- `a1de`
- `a2de`
- `rkdb-numModels.{In}-k.{Int}`
- `decisiontree-depth.{Int}`
- `randomforest-numTrees.{Int}-depth{Int}`


### Contributing

If you plan to contribute or extend our research, first of all: thank you!. Please contact me as I can provide you with additional details about the interals of the code.

