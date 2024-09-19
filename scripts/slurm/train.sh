#!/bin/bash
#SBATCH --cpus-per-task=16                # Number of CPU cores per task
#SBATCH --nodes=1                         # Ensure that all cores are on one machine
#SBATCH --partition=v100                  # Request a specific partition for the resource allocation
#SBATCH --time=3-00:00                    # Runtime in D-HH:MM
#SBATCH --gres=gpu:4                      # optionally type and number of gpus
#SBATCH --mem=100G                        # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logfiles/%j.out          # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logfiles/%j.err           # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=anonymous@domain.com  # Email to which notifications will be sent

# bash script arguments: $1 train config file
# Example call: 
# sbatch scripts/slurm/train.sh configs/configs_train/age_camera_w_dcor.yaml

# Print info about current job.
scontrol show job $SLURM_JOB_ID 

PROJECT_DIR=/absolute/path/to/code/
python_path=/path/to/python/

# We need to mount the EyePACS folder on our slurm cluster.
echo "mount eyepacs folder"
sudo /opt/eyepacs/start_mount.sh

echo "train model" 
$python_path $PROJECT_DIR/src/train.py -c $1

# Unmount the EyePACS folder.
echo "unmount eyepacs folder"
sudo /opt/eyepacs/stop_mount.sh