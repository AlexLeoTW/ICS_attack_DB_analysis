#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

DATASET_DIR="../../2_classes/"
STEP_SIZE_LIST="1 5 10 15 20 25 30 35 40"
RNN_UNIT_SIZE_LIST="10 20 30 40 50 60"
FILE_LIST=$(find "$DATASET_DIR" -maxdepth 1 -name "*.csv" -type f | sort -V)

for file in $FILE_LIST; do
  for step_size in $STEP_SIZE_LIST; do
    for unit in $RNN_UNIT_SIZE_LIST; do
      for (( i = 0; i < 10; i++ )); do
        python RNN.py --step $step_size --unit $unit $file
      done
    done
  done
done
