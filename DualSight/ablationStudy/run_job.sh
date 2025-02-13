#!/bin/bash
#SBATCH --job-name=IDs_v2       # Job name
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH -C A100|V100|L40S               # A100 or V100 GPU
#SBATCH --mem=12000MB                   # Memory in MB
#SBATCH --time=6:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=short               # Partition name

# Print some job information
echo "Running job on $SLURM_JOB_NODELIST"
echo "Requested resources:"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - GPUs: $SLURM_GPUS"
echo "  - Memory: $SLURM_MEM_PER_NODE"

# Activate the Python virtual environment
source /home/sprice/ICCV25/maskFormerEnv/bin/activate


# python ablation_study_cl.py \
#         --BoxInclusion False \
#         # --MaskInclusion True \
#         --Model "YOLOv8 X-Large + Sam" \
#         --NumberOfPOIs 7 \
#         # --BoundingBoxDistortion "110%" \

# python /home/sprice/ICCV25/DualSight/multiImageDS_SR_V5.py
python /home/sprice/ICCV25/DualSight/ablationStudy/ablation_study.py
# python /home/sprice/ICCV25/DualSight/ablationStudy/testingPickle2.py


# python model_sam_inference.py \
#         --inference-pickle "../../savedInference/particle_yolov8x_inference.pkl" \
#         --images-dir "/home/sprice/ICCV25/datasets/powder/test" \
#         --output-pickle "../../savedInference/particle_yolov8x_dualsight_inference.pkl" \
#         --device "cuda"