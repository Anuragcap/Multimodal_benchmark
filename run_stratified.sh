#!/bin/bash
#SBATCH --job-name=strat_3way_36
#SBATCH --output=logs/strat_3way_%j.out
#SBATCH --error=logs/strat_3way_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
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
OUTPUT_DIR="./stratified_3way_results"
BALANCE_STRATEGY="original"
NUM_RUNS=36

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Starting STRATIFIED 3-way experiments..."
echo "Dataset: $DATASET_PATH"
echo "Balance: $BALANCE_STRATEGY"
echo "Output: $OUTPUT_DIR"
echo "Runs:   $NUM_RUNS"
echo ""
echo "Models in each run:"
echo "  1. Text-Only (captions only)"
echo "  2. Single Modal (vision only)"
echo "  3. Multimodal (custom captions)"
echo ""


python stratified_frozen_runner.py \
    --dataset_path $DATASET_PATH \
    --balance_strategy $BALANCE_STRATEGY \
    --output_dir $OUTPUT_DIR \
    --num_runs $NUM_RUNS

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Results: $OUTPUT_DIR/stratified_frozen_3way_original/stratified_frozen_3way_results.json"
    echo "  2. Logs:    $OUTPUT_DIR/stratified_frozen_3way_original/stratified_experiment_log.txt"
else
    echo "=========================================="
    echo "Job failed with exit code $?"
fi

echo "End time: $(date)"