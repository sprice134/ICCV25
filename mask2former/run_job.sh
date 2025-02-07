#!/bin/bash
#SBATCH --job-name=mask2FormerOptimal    # Job name
#SBATCH --cpus-per-task=48               # Number of CPU cores
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --mem=40000MB                   # Memory in MB
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=short               # Partition name

# Print some job information
echo "Running job on $SLURM_JOB_NODELIST"
echo "Requested resources:"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - GPUs: $SLURM_GPUS"
echo "  - Memory: $SLURM_MEM_PER_NODE"

# Activate the Python virtual environment
source /home/sprice/ICCV25/maskFormerEnv/bin/activate


# python compute_mis_commandLine_v2.py --node_counts 15 20 25 30 35 40 45 50 55 
# python compute_mis_commandLine_v2.py --node_counts 60 65 70 75 
# python compute_mis_commandLine_v2.py --node_counts 80 85 90 95
# python compute_mis_commandLine_v2.py --node_counts 100
# python compute_mis_commandLine_v2.py --node_counts 105
# python compute_mis_commandLine_v2.py --node_counts 110
# python compute_mis_commandLine_v2.py --node_counts 115
# python compute_mis_commandLine_v2.py --node_counts 120

# python misEvaluator.py --node_counts 55 --base_dir generated_graphs --output_dir mis_results_grouped_v3
# python misEvaluator.py --node_counts 60 --base_dir generated_graphs --output_dir mis_results_grouped_v3
# python misEvaluator.py --node_counts 65 --base_dir generated_graphs --output_dir mis_results_grouped_v3
python /home/sprice/ICCV25/mask2former/trainMask2FormerOptimal.py

# python compute_greedy_mis_commandline.py --node_counts 100 105 110 115 120 125 130 135 140 145 150
