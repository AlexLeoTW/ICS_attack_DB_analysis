#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
DATASET_DIR="../../2_classes/"
STEP_SIZE_LIST="1 5 10 15 20 25 30 35 40"
RNN_UNIT_SIZE_LIST="10 20 30 40 50 60"
ANN_LIST=("SimpleRNN.py" "LSTM.py" "GRU.py")
PYTHON_CMD="python"
FILE_LIST=$(find "$DATASET_DIR" -maxdepth 1 -name "*.csv" -type f | sort -V)

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash

BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for file in $FILE_LIST; do
    for step_size in $STEP_SIZE_LIST; do
      for unit in $RNN_UNIT_SIZE_LIST; do
        for ann_py in ${ANN_LIST[*]}; do
          for (( i = 0; i < 10; i++ )); do
            echo "$PYTHON_CMD $ann_py --step $step_size --unit $unit $file" >> $JOB_FILE
            echo "sed --in-place '7,8d' \$BASENAME" >> $JOB_FILE
          done
        done
      done
    done
  done
}

if [[ ! -f $JOB_FILE ]]; then
  printf "\n\nJob file \"${JOB_FILE}\" not exist, create one.\n\n"
  new_job_file
fi

bash $JOB_FILE
