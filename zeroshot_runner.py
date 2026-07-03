import os
import json
import argparse
import numpy as np
import torch
import open_clip
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from scipy import stats
from itertools import combinations

WILD_PROMPTS = [
    "a photo of a wild animal",
    "a photo of an animal in the wild",
    "a photo of an animal in its natural habitat",
    "a photo of a free animal in nature",
]

CAPTIVE_PROMPTS = [
    "a photo of a captive animal",
    "a photo of an animal in captivity",
    "a photo of an animal in a zoo",
    "a photo of an animal in an enclosure",
]

LABEL_NAMES = ["wild", "captive"]  # 0=wild, 1=captive

class ZeroShotCLIPEvaluator:
    """Load CLIP once, reuse across all 36 runs."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        self.class_embeddings = self._encode_class_prompts()

    def _encode_class_prompts(self) -> torch.Tensor:
        
        with torch.no_grad():
            embeddings = []
            for prompts in [WILD_PROMPTS, CAPTIVE_PROMPTS]:
                tokens = self.tokenizer(prompts).to(self.device)
                feats = self.clip_model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean = feats.mean(dim=0)
                mean = mean / mean.norm()
                embeddings.append(mean)
        return torch.stack(embeddings, dim=0)  # [2, D]

    @torch.no_grad()
    def evaluate_split(self, dataloader) -> Dict[str, Any]:
        
        all_predictions, all_labels, all_probs, all_species = [], [], [], []

        for batch in tqdm(dataloader, desc="Zero-shot inference", leave=False):
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            labels = batch['label']
            species = batch.get('species', ['unknown'] * len(labels))

            image_feats = self.clip_model.encode_image(images)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            similarity = image_feats @ self.class_embeddings.T   # [B, 2]
            probs = torch.softmax(similarity * 100.0, dim=-1)
            predictions = similarity.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(captive)
            all_species.extend(species)

        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probs': all_probs,
            'species': all_species,
        }




def run_single_zeroshot_experiment(
    evaluator: ZeroShotCLIPEvaluator,
    dataset_path: str,
    balance_strategy: str,
    run_id: int,
    seed: int,
    logger,
    split_mode: str,                        
    test_species_pair: Optional[tuple] = None,
) -> Dict[str, Any]:
    

    from dataset import prepare_dataset, create_data_transforms, WildlifeDataset
    from config import create_config
    from utils import set_seed, evaluate_model_performance
    from torch.utils.data import DataLoader

    logger.info(f"\nRun {run_id} | seed={seed} | mode={split_mode}"
                + (f" | test_species={test_species_pair}" if test_species_pair else ""))

    try:
        set_seed(seed)

        image_paths, labels, species_list = prepare_dataset(
            dataset_path, balance_strategy, logger
        )

        
        if split_mode == "stratified":
            from stratified_random_splitter import create_stratified_random_splits
            splits = create_stratified_random_splits(
                image_paths, labels, species_list,
                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                random_seed=seed, logger=logger
            )
        else:  # ood
            from random_species_splitter import create_random_species_splits
            splits = create_random_species_splits(
                image_paths, labels, species_list,
                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                random_seed=seed, logger=logger,
                test_species_override=test_species_pair
            )

        
        transforms_dict = create_data_transforms()
        test_paths, test_labels, test_species = splits['test']

        test_dataset = WildlifeDataset(
            image_paths=test_paths,
            labels=test_labels,
            species=test_species,
            transform=transforms_dict['test']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

        logger.info(f"  Test set: {len(test_dataset)} images | "
                    f"species: {sorted(set(test_species))}")

        
        raw = evaluator.evaluate_split(test_loader)

        
        metrics, report = evaluate_model_performance(
            raw['labels'], raw['predictions'], raw['probs'], LABEL_NAMES
        )

        # Per-species breakdown
        species_metrics = {}
        for sp in set(raw['species']):
            if sp == 'unknown':
                continue
            idx = [i for i, s in enumerate(raw['species']) if s == sp]
            sp_labels = [raw['labels'][i] for i in idx]
            sp_preds  = [raw['predictions'][i] for i in idx]
            sp_acc    = sum(p == l for p, l in zip(sp_preds, sp_labels)) / len(sp_labels)
            species_metrics[sp] = {'accuracy': sp_acc, 'samples': len(idx)}

        logger.info(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}  "
                    f"AUC={metrics['auc_roc']:.4f}  MCC={metrics['mcc']:.4f}")

        return {
            'run_id': run_id,
            'seed': seed,
            'split_mode': split_mode,
            'test_species': list(test_species_pair) if test_species_pair else None,
            'test_size': len(test_dataset),
            'overall_metrics': metrics,
            'species_metrics': species_metrics,
        }

    except Exception as e:
        logger.error(f"  Run {run_id} FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'run_id': run_id, 'seed': seed, 'error': str(e)}




def compute_zeroshot_statistics(results: List[Dict]) -> Dict:
    """Compute mean/std/CI across 36 runs."""
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc', 'mcc']
    stats_out = {}
    for m in metrics:
        values = [r['overall_metrics'][m] for r in results if 'overall_metrics' in r]
        if not values:
            continue
        arr = np.array(values)
        n = len(arr)
        se = arr.std() / np.sqrt(n)
        ci95 = se * stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
        stats_out[m] = {
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'min':  float(arr.min()),
            'max':  float(arr.max()),
            'ci95': float(ci95),
            'n':    n,
        }
    return stats_out




def run_zeroshot_stratified(dataset_path: str, balance_strategy: str,
                             output_base_dir: str, num_runs: int = 36):
    from utils import setup_logging

    experiment_name = f"zeroshot_stratified_{balance_strategy.replace(':', '_')}"
    output_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging("INFO", os.path.join(output_dir, 'zeroshot_stratified.log'))

    logger.info("=" * 80)
    logger.info("ZERO-SHOT CLIP BASELINE — STRATIFIED (36 RUNS)")
    logger.info("=" * 80)
    logger.info(f"Wild prompts:    {WILD_PROMPTS}")
    logger.info(f"Captive prompts: {CAPTIVE_PROMPTS}")
    logger.info(f"Splits: stratified random (same species in train+test)")
    logger.info(f"No training — pure cosine similarity")
    logger.info("=" * 80)

    evaluator = ZeroShotCLIPEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")

    results = {
        'experiment_type': 'zeroshot_clip_stratified',
        'num_runs': num_runs,
        'prompts': {'wild': WILD_PROMPTS, 'captive': CAPTIVE_PROMPTS},
        'start_time': datetime.now().isoformat(),
        'successful_experiments': [],
        'failed_experiments': [],
    }

    for i in range(num_runs):
        seed = 42 + i * 100
        result = run_single_zeroshot_experiment(
            evaluator, dataset_path, balance_strategy,
            run_id=i + 1, seed=seed, logger=logger,
            split_mode="stratified"
        )
        if 'error' in result:
            results['failed_experiments'].append(result)
        else:
            results['successful_experiments'].append(result)
        logger.info(f"Progress: {i+1}/{num_runs}")

    # Stats
    successful = results['successful_experiments']
    if len(successful) >= 3:
        results['statistical_analysis'] = compute_zeroshot_statistics(successful)

    results['end_time'] = datetime.now().isoformat()
    results['successful_runs'] = len(successful)
    results['failed_runs'] = len(results['failed_experiments'])

    # Save
    out_file = os.path.join(output_dir, 'zeroshot_stratified_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    _print_summary(results, logger, "STRATIFIED")
    logger.info(f"Results saved: {out_file}")
    return results




def run_zeroshot_ood(dataset_path: str, balance_strategy: str,
                     output_base_dir: str):
    from utils import setup_logging
    from dataset import prepare_dataset
    from random_species_splitter import get_all_species_combinations

    experiment_name = f"zeroshot_ood_{balance_strategy.replace(':', '_')}"
    output_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging("INFO", os.path.join(output_dir, 'zeroshot_ood.log'))

    logger.info("=" * 80)
    logger.info("ZERO-SHOT CLIP BASELINE — OOD ALL 36 COMBINATIONS")
    logger.info("=" * 80)
    logger.info(f"Wild prompts:    {WILD_PROMPTS}")
    logger.info(f"Captive prompts: {CAPTIVE_PROMPTS}")
    logger.info(f"Splits: OOD (test species never seen in training)")
    logger.info(f"No training — pure cosine similarity")
    logger.info("=" * 80)

    # Get all 36 species combinations (same as OOD runner)
    image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)
    all_combinations = get_all_species_combinations(species_list, n_test_species=2)

    logger.info(f"\nTesting all {len(all_combinations)} OOD combinations:")
    for i, combo in enumerate(all_combinations, 1):
        logger.info(f"  {i:2d}. {combo[0]} + {combo[1]}")

    evaluator = ZeroShotCLIPEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")

    results = {
        'experiment_type': 'zeroshot_clip_ood',
        'num_combinations': len(all_combinations),
        'prompts': {'wild': WILD_PROMPTS, 'captive': CAPTIVE_PROMPTS},
        'start_time': datetime.now().isoformat(),
        'successful_experiments': [],
        'failed_experiments': [],
    }

    for i, combo in enumerate(all_combinations):
        seed = 42 + i * 100  # same seeds as original OOD runner
        result = run_single_zeroshot_experiment(
            evaluator, dataset_path, balance_strategy,
            run_id=i + 1, seed=seed, logger=logger,
            split_mode="ood",
            test_species_pair=combo
        )
        if 'error' in result:
            results['failed_experiments'].append(result)
        else:
            results['successful_experiments'].append(result)
        logger.info(f"Progress: {i+1}/{len(all_combinations)} | combo: {combo}")

    # Stats
    successful = results['successful_experiments']
    if len(successful) >= 3:
        results['statistical_analysis'] = compute_zeroshot_statistics(successful)

    results['end_time'] = datetime.now().isoformat()
    results['successful_runs'] = len(successful)
    results['failed_runs'] = len(results['failed_experiments'])

    # Save
    out_file = os.path.join(output_dir, 'zeroshot_ood_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    _print_summary(results, logger, "OOD")
    logger.info(f"Results saved: {out_file}")
    return results




def _print_summary(results: Dict, logger, mode: str):
    logger.info("\n" + "=" * 80)
    logger.info(f"ZERO-SHOT CLIP BASELINE — {mode} SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successful: {results['successful_runs']} / "
                f"{results.get('num_runs', results.get('num_combinations', '?'))}")

    if 'statistical_analysis' in results:
        sa = results['statistical_analysis']
        logger.info("\nMETRIC             MEAN ± STD         95% CI")
        logger.info("-" * 50)
        for m in ['accuracy', 'f1_score', 'auc_roc', 'mcc']:
            if m in sa:
                d = sa[m]
                logger.info(f"{m:<18} {d['mean']:.4f} ± {d['std']:.4f}    "
                            f"[{d['mean']-d['ci95']:.4f}, {d['mean']+d['ci95']:.4f}]")
    logger.info("=" * 80)




def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot CLIP Baseline — 36 runs for Stratified or OOD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['stratified', 'ood'],
                        help='stratified = same species in train+test | ood = held-out species')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--balance_strategy', type=str, default='1:10',
                        choices=['original', '1:1', '1:10'])
    parser.add_argument('--num_runs', type=int, default=36,
                        help='Number of runs (stratified mode only; OOD always uses all 36 combos)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'stratified':
        run_zeroshot_stratified(
            args.dataset_path, args.balance_strategy,
            args.output_dir, args.num_runs
        )
    else:
        run_zeroshot_ood(
            args.dataset_path, args.balance_strategy,
            args.output_dir
        )


if __name__ == "__main__":
    main()