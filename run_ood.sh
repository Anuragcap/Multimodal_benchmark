#!/bin/bash
#SBATCH --job-name=ood_4way_36
#SBATCH --output=logs/ood_4way_%j.out
#SBATCH --error=logs/ood_4way_%j.err
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
OUTPUT_DIR="./ood_4way_results"
BALANCE_STRATEGY="original"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Starting OOD 4-way experiments with ALL 36 combinations..."
echo "Dataset: $DATASET_PATH"
echo "Balance: $BALANCE_STRATEGY"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Models in each combination:"
echo "  1. Text-Only (captions only)"
echo "  2. Single Modal (vision only)"
echo "  3. Multimodal (custom captions)"
echo "  4. Adversarial (DANN-fixed: sigmoid lambda 0->1, shallow discriminator)"
echo ""
echo "  (Leaking CLIP baseline removed; true zero-shot reported separately)"
echo "=========================================="

python adversarial_runner.py \
    --dataset_path $DATASET_PATH \
    --balance_strategy $BALANCE_STRATEGY \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Results: $OUTPUT_DIR/ood_adversarial_4way_original/ood_generalization_4way_results.json"
    echo "  2. Logs:    $OUTPUT_DIR/ood_adversarial_4way_original/ood_experiment_log.txt"
else
    echo "=========================================="
    echo "Job failed with exit code $?"
fi

echo "End time: $(date)"