#!/bin/bash
#SBATCH --job-name=zeroshot_ood
#SBATCH --output=logs/zeroshot_ood_%j.out
#SBATCH --error=logs/zeroshot_ood_%j.err
#SBATCH --time=08:00:00
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

# Set paths  (same as run_ood.sh)
DATASET_PATH=""
OUTPUT_DIR="./zeroshot_baseline_results"
BALANCE_STRATEGY="original"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Dataset:  $DATASET_PATH"
echo "Balance:  $BALANCE_STRATEGY"
echo "Output:   $OUTPUT_DIR"
echo "=========================================="

python zeroshot_runner.py \
    --mode ood \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --balance_strategy $BALANCE_STRATEGY

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully!"
    echo "All 36 OOD combinations tested!"
    echo "Results: $OUTPUT_DIR/zeroshot_ood_1_10/zeroshot_ood_results.json"
else
    echo "=========================================="
    echo "Job failed with exit code $?"
fi

echo "End time: $(date)"