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

# bash script arguments: $1 experiment_folder $2 config file
# Example call: 
# sh scripts/slurm/compute_image_quality.sh outputs/expriment_folder \
#   configs/configs_image_quality/default.yaml

# Print info about current job.
scontrol show job $SLURM_JOB_ID 

echo "experiment folder" $1

PROJECT_DIR=/absolute/path/to/code/
python_path=/path/to/python/

# We need to mount the EyePACS folder on our slurm cluster.
echo "mount eyepacs folder"
sudo /opt/eyepacs/start_mount.sh

echo "compute image quality (fid, kid)..."
$python_path $PROJECT_DIR/src/evaluation/eval_image_quality.py -d $1 -c $2

# Unmount the EyePACS folder.
echo "unmount eyepacs folder"
sudo /opt/eyepacs/stop_mount.sh