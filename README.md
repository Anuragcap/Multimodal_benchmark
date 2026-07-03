# Captive or Wild? A Multimodal Benchmark for Cross-Species Captivity Detection

A benchmark and codebase for automatically classifying wildlife images as **captive** or **wild**, designed to generalize to species never seen during training — supporting scalable enforcement against illegal wildlife trafficking.

## Overview

Existing captivity-detection approaches rely on species-specific models that require retraining for every new species, and typically use images alone. This repo addresses both limitations by:

- Generating **five-aspect environmental captions** (behavior, surroundings, background, lighting, vegetation) for each image using BLIP-2, capturing context cues (e.g., cages, enclosures) rather than species morphology.
- Pairing images with these captions to build **multimodal** (image + text) representations.
- Benchmarking vision and vision-language backbones — **CLIP, SigLIP\,2, BioCLIP, DINOv2** — under **zero-shot, few-shot, frozen-probe (image-only / text-only / multimodal), and LoRA** (text / image / both) adaptation.
- Evaluating under both **in-distribution** (species seen during training) and **out-of-distribution / species-disjoint** (species never seen during training) splits, across two paired, source-controlled datasets covering nine CITES Appendix I/II protected species.

**Key finding:** lightweight LoRA adaptation of the vision encoder consistently performs best, while a frozen multimodal probe (no training required) is the strongest training-free option — and the generated captions help most precisely in that low-data / frozen / cross-species regime.

## Datasets

| Dataset | Description | Captive | Wild | Total |
|---|---|---|---|---|
| **1:10 Ratio Dataset** (`3858`) | Captive images from web search; wild images from iNaturalist | 351 | 3,507 | 3,858 |
| **Same-Source Dataset** (`810`) | Both classes from Bing Images, controlling for platform artifacts | 351 | 459 | 810 |

Species: *Ateles geoffroyi, Canis lupus, Caracal caracal, Cervus elaphus, Chinchilla lanigera, Lemur catta, Pardofelis marmorata, Prionailurus bengalensis, Saguinus oedipus.*

Expected directory structure :

```
data/
├── 1_10_dataset/
│   ├── Ateles_geoffroyi/
│   │   ├── captive/*.jpg
│   │   └── wild/*.jpg
│   ├── Canis_lupus/
│   │   ├── captive/
│   │   └── wild/
│   └── ...
└── same_source_dataset/
    └── ... (same layout)
```

## Repository Structure

```
.
├── # Core: Data, Config, Utilities
├── config.py                              # Config dataclasses (Data/Model/Training), label mapping
├── utils.py                                # Logging, seeding, class weights, evaluation metrics
├── dataset.py                              # WildlifeDataset, transforms, dataloaders, caption formatting
├── stratified_random_splitter.py           # In-distribution splits (species present in train/val/test)
├── random_species_splitter.py              # Out-of-distribution splits (species-disjoint; 36 combos)
│
├── # Caption Generation (BLIP-2)
├── caption_generator.py                    # Environmental captions (behavior/surroundings/background/lighting/vegetation) — main benchmark
├── caption_generator_species_specific.py   # Identity-attribute captions (fur/body/head-face/markings/size) — ablation
├── caption_generator_generic.py            # Single open-ended caption ("Describe this image") — ablation
│
├── # CLIP Models & Experiments
├── multimodal.py / multimodal_stratified.py    # MultiModalCLIP: frozen CLIP, image+text concat + MLP head
├── single_modal.py / single_modal_stratified.py # Vision-only CLIP baselines (ResNet-50/101, ViT-B/16)
├── text_only.py                            # Text-only (caption-only) frozen baseline
├── lora_vision_multimodal.py               # LoRA model classes: VisionLoRA, TextOnlyLoRA, TextLoRA, MultimodalLoRA, BothLoRA
├── clip_zeroshot.py                        # True zero-shot CLIP (prompt ensembling, no training)
├── zeroshot_runner.py                      # Zero-shot CLIP: 36-run stratified + all-36-combo OOD, one model load
├── clip_presence_runner.py                 # 4-way: multimodal-frozen / text-only-LoRA / image-only-LoRA / both-LoRA
├── lora_vision_runner.py                   # 4-way: multimodal-frozen / vision-frozen / vision-LoRA / multimodal-LoRA
├── lora_placement_runner.py                # 4-way LoRA-placement ablation: frozen / vision-LoRA / text-LoRA / both-LoRA
├── fewshot_runner.py                       # K-shot (K=1,5,10,20) linear probe, vision-only vs multimodal, frozen CLIP
├── stratified_frozen_runner.py             # In-distribution frozen probes: text-only / single-modal / multimodal (36 seeds)
├── ood_frozen_runner.py                    # Out-of-distribution version of the above (36 species combos)
│
├── # SigLIP2 Models & Experiments
├── siglip2_backbone.py                     # SigLIP2 model classes: Frozen, ImageOnlyLoRA, TextOnlyLoRA, BothLoRA
├── siglip2_runner.py                       # 4-condition LoRA sweep: frozen / text-only-LoRA / image-only-LoRA / both-LoRA
├── siglip2_frozen_eval.py                  # SigLIP2 zero-shot, frozen vision/text probes, few-shot (k=1,5,10,20)
│
├── # BioCLIP / DINOv2 Baselines
├── bioclip_baseline.py                     # BioClipTrainer: frozen BioCLIP encoder + MLP head
├── bioclip_stratified_experiment.py        # BioCLIP in-distribution (36 seeds)
├── bioclip_ood_experiment.py               # BioCLIP out-of-distribution (36 species combos)
├── dinov2_baseline.py                      # DINOv2Trainer: frozen DINOv2 encoder + MLP head
├── dinov2_stratified_experiment.py         # DINOv2 in-distribution (36 seeds)
├── dinov2_ood_experiment.py                # DINOv2 out-of-distribution (36 species combos)
│
├── # SLURM Job Scripts
├── run_stratified.sh / run_ood.sh          # Text-only / single-modal / multimodal frozen probes (in-dist. / OOD)
├── run_clip_presence.sh                    # clip_presence_runner.py array job (810/3858 × ood/stratified)
├── run_siglip2.sh                          # siglip2_runner.py array job (810/3858 × ood/stratified)
├── run_siglip_frozen_eval.sh               # siglip2_frozen_eval.py array job (zero-shot/frozen/few-shot)
├── run_loraplace.sh                        # lora_placement_runner.py, all 4 (dataset × split) combos sequentially
├── run_fewshot.sh                          # fewshot_runner.py, all 4 (dataset × split) combos sequentially
├── run_bioclip_stratified.sh / run_bioclip_ood.sh   # BioCLIP in-distribution / OOD
├── run_dinov2_stratified.sh / run_dinov2.sh         # DINOv2 in-distribution / OOD
├── zeroshot_runner.sh / zeroshot_runnerood.sh       # Zero-shot CLIP (stratified / OOD)
│
└── data/                                    # 1:10 ratio (3858) and same-source (810) datasets
```

## Installation

Requirements: Python 3.8+, CUDA-capable GPU (experiments run on NVIDIA L40S)

```bash
pip install torch torchvision open_clip_torch transformers timm peft \
    scipy scikit-learn tqdm matplotlib pillow
```

## Usage

### Step 1 — Generate captions

Three caption strategies are available, matching the ablation in the paper (Section 5). All three share the same CLI and output schema (a JSON mapping image path → `{"captions": {...}, "label": ..., "species": ...}`):

**Environmental (main benchmark)** — five aspects describing the *scene*:

```bash
python caption_generator.py \
    --dataset_path data/1_10_dataset \
    --output_file ./captions_environmental.json \
    --balance_strategy original
```

**Species-specific / identity-attribute (ablation)** — same five-slot structure, but queries the *animal's appearance* (fur/coat, body shape, head/face, markings, size):

```bash
python caption_generator_species_specific.py \
    --dataset_path data/1_10_dataset \
    --output_file ./captions_species_specific.json \
    --balance_strategy original
```

**Generic (ablation)** — a single open-ended "Describe this image" prompt, stored under a `description` key:

```bash
python caption_generator_generic.py \
    --dataset_path data/1_10_dataset \
    --output_file ./captions_generic.json \
    --balance_strategy original
```

The environmental captions are used for the main results; all three are compared in the caption-strategy ablation, where environmental captions win on every out-of-distribution split.

### Step 2 — Zero-shot CLIP baseline (no training)

Single dataset run:

```bash
python clip_zeroshot.py \
    --dataset_path data/1_10_dataset \
    --output_dir ./zeroshot_results \
    --balance_strategy original
```

36-run stratified + all-36-combination OOD (loads CLIP once, reuses across runs):

```bash
# In-distribution
python zeroshot_runner.py --mode stratified \
    --dataset_path data/1_10_dataset --output_dir ./zeroshot_baseline_results \
    --balance_strategy 1:10 --num_runs 36

# Out-of-distribution (all 36 species-pair combinations)
python zeroshot_runner.py --mode ood \
    --dataset_path data/1_10_dataset --output_dir ./zeroshot_baseline_results \
    --balance_strategy 1:10
```

### Step 3 — CLIP frozen-probe / LoRA sweep

`clip_presence_runner.py` trains four conditions per run — **multimodal-frozen**, **text-only-LoRA**, **image-only-LoRA**, **both-LoRA** — and reports paired-comparison statistics (t-tests) across all runs:

```bash
python clip_presence_runner.py \
    --dataset_path data/1_10_dataset \
    --captions_file ./captions_environmental.json \
    --output_dir ./clip_presence_results \
    --balance_strategy original \
    --split_mode stratified \
    --ds_tag 3858
```

`--split_mode` is `stratified` (in-distribution, 36 seeds) or `ood` (species-disjoint, all 36 combinations). `--ds_tag` is just a label for the output filename (e.g. `810` or `3858`).

Related runners with the same CLI (`--dataset_path --captions_file --output_dir --balance_strategy --split_mode --ds_tag`):

- `lora_vision_runner.py` — multimodal-frozen / vision-frozen / vision-LoRA / multimodal-LoRA (isolates whether LoRA or multimodality drives the gain)
- `lora_placement_runner.py` — frozen / vision-LoRA / text-LoRA / both-LoRA (which tower benefits most from adaptation)
- `fewshot_runner.py` — add `--n_runs` optional; K-shot (K=1,5,10,20) linear probes, vision-only vs. multimodal

### Step 4 — SigLIP\,2 experiments

LoRA sweep (frozen / text-only-LoRA / image-only-LoRA / both-LoRA):

```bash
python siglip2_runner.py \
    --dataset_path data/1_10_dataset \
    --captions_file ./captions_environmental.json \
    --output_dir ./siglip2_results \
    --balance_strategy original \
    --split_mode ood \
    --ds_tag 3858
```

Zero-shot, frozen probe, and few-shot (K=1,5,10,20) evaluation:

```bash
python siglip2_frozen_eval.py \
    --dataset_path data/1_10_dataset \
    --captions_file ./captions_environmental.json \
    --output_dir ./siglip2_frozen \
    --balance_strategy original \
    --split_mode stratified \
    --ds_tag 3858 \
    --n_runs 36
```

### Step 5 — BioCLIP / DINOv2 baselines

```bash
# BioCLIP
python bioclip_stratified_experiment.py --dataset_path data/1_10_dataset --output_dir ./bioclip_stratified_36runs --balance_strategy original --num_runs 36
python bioclip_ood_experiment.py        --dataset_path data/1_10_dataset --output_dir ./bioclip_ood_all_combos    --balance_strategy original

# DINOv2
python dinov2_stratified_experiment.py --dataset_path data/1_10_dataset --output_dir ./dinov2_stratified_36runs --balance_strategy original --num_runs 36
python dinov2_ood_experiment.py        --dataset_path data/1_10_dataset --output_dir ./dinov2_ood_all_combos    --balance_strategy original
```

### Step 6 — Frozen-probe baselines (Text-only / Single-Modal / Multimodal)

In-distribution and out-of-distribution runs of the three frozen-probe conditions reported in the paper — Text-only, Single-Modal (vision-only), and Multimodal — each trained for 36 seeds (stratified) or across all 36 species-disjoint combinations (OOD):

```bash
python stratified_frozen_runner.py --dataset_path data/1_10_dataset --balance_strategy original --output_dir ./stratified_results --num_runs 36
python ood_frozen_runner.py        --dataset_path data/1_10_dataset --balance_strategy original --output_dir ./ood_results
```

### Running via SLURM

Each experiment has a matching `.sh` script (edit `DATASET_PATH` / `CAPTIONS_810` / `CAPTIONS_3858` at the top of each script first):

```bash
sbatch run_stratified.sh          # text-only / single-modal / multimodal frozen probes (in-distribution)
sbatch run_ood.sh                 # same, out-of-distribution
sbatch run_clip_presence.sh       # clip_presence_runner.py, array job over 810/3858 × ood/stratified
sbatch run_siglip2.sh             # siglip2_runner.py, array job
sbatch run_siglip_frozen_eval.sh  # siglip2_frozen_eval.py, array job
sbatch run_loraplace.sh           # lora_placement_runner.py, all 4 combos
sbatch run_fewshot.sh             # fewshot_runner.py, all 4 combos
sbatch run_bioclip_stratified.sh / run_bioclip_ood.sh
sbatch run_dinov2_stratified.sh / run_dinov2.sh
sbatch zeroshot_runner.sh / zeroshot_runnerood.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{Anonymous2026,
  title     = {Captive or Wild? A Multimodal Benchmark for Cross-Species Captivity Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```