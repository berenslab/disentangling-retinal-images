#!/bin/bash
#SBATCH --cpus-per-task=16                # Number of CPU cores per task
#SBATCH --nodes=1                         # Ensure that all cores are on one machine
#SBATCH --partition=gpu-2080              # Request a specific partition for the resource allocation
#SBATCH --time=0-00:30                    # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                      # optionally type and number of gpus
#SBATCH --mem=100G                        # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logfiles/%j.out          # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logfiles/%j.err           # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=anonymous@domain.com  # Email to which notifications will be sent

# bash script arguments: $1 experiment_folder $2-$end subspace dimensions
# Subspace dimensions:
# [age, camera, identity] = [4, 12, 16]
# [ethnicity, camera, identity] = [8, 12, 16]
# Example call: 
# sh scripts/slurm/knn_eval_embeddings.sh outputs/expriment_folder 4 12 16

# Print info about current job.
scontrol show job $SLURM_JOB_ID 

echo "experiment folder" $1

PROJECT_DIR=/absolute/path/to/code/
python_path=/path/to/python/

# We need to mount the EyePACS folder on our slurm cluster.
echo "mount eyepacs folder"
sudo /opt/eyepacs/start_mount.sh

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

$python_path src/evaluation/eval_knn.py -d $1 -ns 3 -k 30 -a patient_ethnicity camera age_groups

# Unmount the EyePACS folder.
echo "unmount eyepacs folder"
sudo /opt/eyepacs/stop_mount.sh