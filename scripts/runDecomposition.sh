#!/bin/bash

# Run in dockerized spark
# Args requires are input_path output_path database algorithm parallelism_level
#

docker run \
    --rm \
    -v $(pwd):/home/work/project \
    jacintoarias/docker-sparkdev \
    spark-submit \
    --jars .staging/**/spark-bnc/target/scala-2.11/spark-bnc_*.jar \
    --class es.jarias.pgm2018.run.EvalDecomp \
    target/scala-2.11/jariaspgm2018_*-0.1.0.jar \
    $@
