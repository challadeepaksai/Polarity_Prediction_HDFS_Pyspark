#!/bin/bash

# Usage: ./run_spark.sh path/to/your_script.py

if [ -z "$1" ]; then
    echo "Error: No script provided."
    echo "Usage: ./run_spark.sh <path_to_python_script>"
    exit 1
fi

SCRIPT_PATH="$1"
PYTHON_ENV="/home/deepakchalla/pyspark_env/bin/python"

spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.executor.instances=2 \
  --conf spark.executor.memory=5g \
  --executor-cores 7 \
  --driver-memory 2g \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="$PYTHON_ENV" \
  --conf spark.executorEnv.PYSPARK_PYTHON="$PYTHON_ENV" \
  "$SCRIPT_PATH"
