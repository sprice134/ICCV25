#!/bin/bash
#SBATCH --job-name=770             # Job name
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --mem=16000MB                   # Memory in MB
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=short               # Partition name

# Print some job information
echo "Running job on $SLURM_JOB_NODELIST"
echo "Requested resources:"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - GPUs: $SLURM_GPUS"
echo "  - Memory: $SLURM_MEM_PER_NODE"

# Activate the Python virtual environment
source /home/sprice/ICCV25/samEnv/bin/activate


# python /home/sprice/ICCV25/DualSight/multiImageDS_SR_V5.py
python /home/sprice/ICCV25/DualSight/ablationStudy/ablation_study.py
# python /home/sprice/ICCV25/DualSight/ablationStudy/testingPickle2.py
