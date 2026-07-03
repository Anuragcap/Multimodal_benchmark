#!/bin/bash
#SBATCH --job-name=sig2froz
#SBATCH --output=logs/sig2froz_%A_%a.out
#SBATCH --error=logs/sig2froz_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --array=0-3

echo "Array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $SLURM_NODELIST | $(date)"
source ~/myenv/bin/activate
nvidia-smi


CAPTIONS_810=""
CAPTIONS_3858=""
OUTPUT_DIR="./siglip2_frozen"
mkdir -p "$OUTPUT_DIR" logs

case $SLURM_ARRAY_TASK_ID in
  0) DS=810;  DP=;  SP=ood;        CAP="$CAPTIONS_810"  ;;
  1) DS=810;  DP=;  SP=stratified; CAP="$CAPTIONS_810"  ;;
  2) DS=3858; DP=;  SP=ood;        CAP="$CAPTIONS_3858" ;;
  3) DS=3858; DP=;  SP=stratified; CAP="$CAPTIONS_3858" ;;
esac

echo ">>> task $SLURM_ARRAY_TASK_ID : ds=$DS split=$SP | $(date)"
python siglip2_frozen_eval.py \
    --dataset_path "$DP" \
    --captions_file "$CAP" \
    --output_dir "$OUTPUT_DIR" \
    --balance_strategy original \
    --split_mode "$SP" \
    --ds_tag "$DS" \
    --n_runs 36
echo ">>> task $SLURM_ARRAY_TASK_ID done exit=$? | $(date)"