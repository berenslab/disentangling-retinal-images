#!/bin/bash

# bash script arguments: $1 experiment_folder $2-$end subspace dimensions
# Subspace dimensions:
# [age, camera, identity] = [4, 12, 16]
# [ethnicity, camera, identity] = [8, 12, 16]
# Example call: 
# sh scripts/knn_eval_embeddings.sh outputs/expriment_folder 4 12 16

echo "experiment folder:" $1

PROJECT_DIR=/absolute/path/to/code/
python_path=/path/to/python/

echo "compute kNN on whole latent space..."
$python_path src/evaluation/compute_knn.py -d $1

echo "compute kNN on latent subspaces..."
start_dim=0
for (( i=2; i <= $#; i++ )); do
    end_dim=$((${!i}+$start_dim))
    echo "subspace $(($i-1)): start dim $start_dim, end_dim $end_dim"
    $python_path src/evaluation/compute_knn.py -d $1 -s $start_dim $end_dim -n subspace$(($i-1))
    start_dim=$end_dim
done

echo "compute knn performance"

num_subspaces=$(($#-1))
echo "evaluate $num_subspaces subspaces:"
$python_path src/evaluation/eval_knn.py -d $1 -ns $num_subspaces -k 30 -a patient_ethnicity camera age_groups