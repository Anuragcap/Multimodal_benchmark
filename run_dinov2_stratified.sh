#!/bin/bash
#SBATCH --job-name=dinov2_strat_36
#SBATCH --output=logs/dinov2_stratified_36runs_%j.out
#SBATCH --error=logs/dinov2_stratified_36runs_%j.err
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
DATASET_PATH=""
OUTPUT_DIR="./dinov2_stratified_36runs_results"
BALANCE_STRATEGY="original"
NUM_RUNS=36

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Run DINOv2 stratified 36 runs
echo "Starting DINOv2 stratified experiment..."
echo "Dataset: $DATASET_PATH"
echo "Balance: $BALANCE_STRATEGY"
echo "Output: $OUTPUT_DIR"
echo "Number of runs: $NUM_RUNS"
echo "=========================================="

python dinov2_stratified_experiment.py \
    --dataset_path $DATASET_PATH \
    --balance_strategy $BALANCE_STRATEGY \
    --output_dir $OUTPUT_DIR \
    --num_runs $NUM_RUNS

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ Job completed successfully!"
    echo "All 36 runs completed!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "=========================================="
    echo "❌ Job failed with exit code $?"
fi

echo "End time: $(date)"