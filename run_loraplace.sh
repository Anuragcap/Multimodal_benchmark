#!/bin/bash
#SBATCH --job-name=loraplace
#SBATCH --output=logs/loraplace_%j.out
#SBATCH --error=logs/loraplace_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST | start $(date)"
source ~/myenv/bin/activate
nvidia-smi

CAPTIONS_810=""
CAPTIONS_3858=""
OUTPUT_DIR="./loraplace_results"
mkdir -p "$OUTPUT_DIR" logs

run () {  # $1=ds_tag  $2=dataset_path  $3=split  $4=captions
    echo ">>> ds=$1 split=$3 | $(date)"
    python lora_placement_runner.py \
        --dataset_path "$2" \
        --captions_file "$4" \
        --output_dir "$OUTPUT_DIR" \
        --balance_strategy original \
        --split_mode "$3" \
        --ds_tag "$1"
    echo ">>> done ds=$1 split=$3 exit=$? | $(date)"
}

run 810    ood        "$CAPTIONS_810"
run 810    stratified "$CAPTIONS_810"
run 3858  ood        "$CAPTIONS_3858"
run 3858  stratified "$CAPTIONS_3858"

echo "All LoRA-placement runs done | $(date)"
echo "Results: $OUTPUT_DIR/lora_placement_<split>_<ds>_original_final.json"