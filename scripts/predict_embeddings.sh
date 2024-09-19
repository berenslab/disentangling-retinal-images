#!/bin/bash

# bash script arguments: $1 experiment_folder $2 predict config file
# Example call: 
# sh scripts/predict_embeddings.sh outputs/expriment_folder configs/configs_predict/default.yaml

echo "experiment folder" $1

PROJECT_DIR=/absolute/path/to/code/
python_path=/path/to/python/

echo "encoder/discriminator inference"
$python_path $PROJECT_DIR/src/predict.py -s train -d $1 -c $2
$python_path $PROJECT_DIR/src/predict.py -s val -d $1 -c $2
$python_path $PROJECT_DIR/src/predict.py -s test -d $1 -c $2