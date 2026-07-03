#!/bin/bash
#SBATCH --job-name=bioclip_ood_36
#SBATCH --output=logs/bioclip_ood_36combos_%j.out
#SBATCH --error=logs/bioclip_ood_36combos_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"


# Activate virtual environment
source ~/myenv/bin/activate

# Verify GPU
nvidia-smi

# Set paths
DATASET_PATH="/home/svenkata/new_810"
OUTPUT_DIR="./bioclip_ood_all_combos_results"
BALANCE_STRATEGY="original"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Run BioClip OOD all combinations
echo "Starting BioClip OOD experiment..."
echo "Dataset: $DATASET_PATH"
echo "Balance: $BALANCE_STRATEGY"
echo "Output: $OUTPUT_DIR"
echo "Testing ALL 36 species combinations"
echo "=========================================="

python bioclip_ood_experiment.py \
    --dataset_path $DATASET_PATH \
    --balance_strategy $BALANCE_STRATEGY \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ Job completed successfully!"
    echo "All 36 combinations tested!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "=========================================="
    echo "❌ Job failed with exit code $?"
fi

echo "End time: $(date)"