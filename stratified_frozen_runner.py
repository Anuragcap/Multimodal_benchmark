import os
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from scipy import stats


def run_single_stratified_experiment(dataset_path: str, balance_strategy: str,
                                     captions_file: str, output_dir: str,
                                     run_id: int, seed: int,
                                     experiment_logger) -> Dict[str, Any]:
    

    experiment_logger.info(f"\n🔬 Running Stratified Experiment {run_id} (seed={seed})")
    experiment_logger.info("   Mode: STRATIFIED (same species in train/test - NOT OOD)")
    experiment_logger.info("   3-way comparison: Text-Only → Single → Multimodal")
    experiment_logger.info("-" * 80)

    try:
        from dataset import prepare_dataset, create_data_transforms
        from stratified_random_splitter import create_stratified_random_splits
        from single_modal import SingleModalTrainer
        from multimodal import MultiModalTrainer
        from text_only import TextOnlyTrainer
        from config import create_config
        from utils import setup_logging, set_seed, load_captions_data
        import open_clip
        from dataset import create_dataloaders as create_single_dataloaders

        set_seed(seed)

        # Prepare dataset
        image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, experiment_logger)

        # Create STRATIFIED splits (NOT OOD!)
        splits = create_stratified_random_splits(
            image_paths, labels, species_list,
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
            random_seed=seed, logger=experiment_logger
        )

        unique_species = sorted(set(species_list))
        train_sp = set(splits['train'][2])
        val_sp = set(splits['val'][2])
        test_sp = set(splits['test'][2])

        experiment_logger.info(f"\n📊 SPECIES DISTRIBUTION FOR RUN {run_id}:")
        experiment_logger.info(f"  Train Species ({len(train_sp)}): {sorted(train_sp)}")
        experiment_logger.info(f"  Val Species   ({len(val_sp)}): {sorted(val_sp)}")
        experiment_logger.info(f"  Test Species  ({len(test_sp)}): {sorted(test_sp)}")

        # Check overlap (should be HIGH for stratified splits)
        overlap_count = len(train_sp & test_sp)
        total_species = len(unique_species)

        experiment_logger.info(f"  Species overlap: {overlap_count}/{total_species} ({100*overlap_count/total_species:.1f}%) ✅ EXPECTED")

        # Create config
        config = create_config(
            dataset_path=dataset_path,
            output_dir=output_dir,
            balance_strategy=balance_strategy,
            batch_size=16,
            num_epochs=15,
            learning_rate=1e-4,
            device="auto"
        )

        # Load captions
        captions_data = load_captions_data(captions_file, experiment_logger)
        if captions_data is None:
            raise ValueError("Could not load captions")

        transforms_dict = create_data_transforms()
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')

        # Filter for captions
        def filter_split_for_captions(split_data):
            paths, labels, species = split_data
            filtered_paths, filtered_labels, filtered_species = [], [], []
            for p, l, s in zip(paths, labels, species):
                if p in captions_data:
                    filtered_paths.append(p)
                    filtered_labels.append(l)
                    filtered_species.append(s)
            return filtered_paths, filtered_labels, filtered_species

        filtered_splits = {
            'train': filter_split_for_captions(splits['train']),
            'val': filter_split_for_captions(splits['val']),
            'test': filter_split_for_captions(splits['test'])
        }

        
        experiment_logger.info(f"\n📝 Training Text-Only Baseline...")

        text_only_dataloaders = create_single_dataloaders(
            filtered_splits, transforms_dict, config.data.batch_size,
            config.data.num_workers, captions_data, clip_tokenizer
        )

        text_only_trainer = TextOnlyTrainer(config, experiment_logger)
        text_only_trainer.setup_training(filtered_splits['train'][1])
        text_only_training = text_only_trainer.train(
            text_only_dataloaders['train'], text_only_dataloaders['val']
        )
        text_only_results = text_only_trainer.evaluate(text_only_dataloaders['test'])

        to = text_only_results['overall_metrics']
        experiment_logger.info(f"  ✓ Text-Only Test: Acc={to['accuracy']:.4f}, F1={to['f1_score']:.4f}, MCC={to['mcc']:.4f}")

        
        single_dataloaders = create_single_dataloaders(
            filtered_splits, transforms_dict, config.data.batch_size, config.data.num_workers
        )

        experiment_logger.info(f"\n🖼️ Training Baseline Single Modal (Vision-Only)...")
        single_trainer = SingleModalTrainer(config, experiment_logger)
        single_trainer.setup_training(filtered_splits['train'][1])
        single_training = single_trainer.train(single_dataloaders['train'], single_dataloaders['val'])
        single_results = single_trainer.evaluate(single_dataloaders['test'])

        sm = single_results['overall_metrics']
        experiment_logger.info(f"  ✓ Single Modal Test: Acc={sm['accuracy']:.4f}, F1={sm['f1_score']:.4f}, MCC={sm['mcc']:.4f}")
        experiment_logger.info(f"    vs Text-Only: {sm['accuracy'] - to['accuracy']:+.4f}")

        
        multimodal_dataloaders = create_single_dataloaders(
            filtered_splits, transforms_dict, config.data.batch_size,
            config.data.num_workers, captions_data, clip_tokenizer
        )

        experiment_logger.info(f"\n📄 Training Baseline Multimodal (Custom Captions)...")
        multimodal_trainer = MultiModalTrainer(config, experiment_logger)
        multimodal_trainer.setup_training(filtered_splits['train'][1])
        multimodal_training = multimodal_trainer.train(multimodal_dataloaders['train'], multimodal_dataloaders['val'])
        multimodal_results = multimodal_trainer.evaluate(multimodal_dataloaders['test'])

        mm = multimodal_results['overall_metrics']
        experiment_logger.info(f"  ✓ Multimodal Test: Acc={mm['accuracy']:.4f}, F1={mm['f1_score']:.4f}, MCC={mm['mcc']:.4f}")
        experiment_logger.info(f"    vs Single Modal: {mm['accuracy'] - sm['accuracy']:+.4f}")

        
        experiment_logger.info(f"\n📈 STRATIFIED RUN {run_id} - 3-WAY COMPARISON:")
        experiment_logger.info(f"  1. Text-Only:        {to['accuracy']:.4f}")
        experiment_logger.info(f"  2. Single Modal:     {sm['accuracy']:.4f} ({sm['accuracy'] - to['accuracy']:+.4f})")
        experiment_logger.info(f"  3. Multimodal:       {mm['accuracy']:.4f} ({mm['accuracy'] - sm['accuracy']:+.4f})")
        experiment_logger.info(f"  Total Improvement:   {mm['accuracy'] - to['accuracy']:+.4f}")

        # Compile results
        experiment_result = {
            'run_id': run_id,
            'seed': seed,
            'split_type': 'stratified_random',
            'species_splits': {
                'train_species': sorted(train_sp),
                'val_species': sorted(val_sp),
                'test_species': sorted(test_sp),
                'total_species': len(unique_species),
                'train_test_overlap_count': overlap_count,
                'train_test_overlap_percent': 100 * overlap_count / total_species,
                'split_verification': 'stratified_with_overlap'
            },
            'dataset_sizes': {
                'total_images': len(image_paths),
                'train_images': len(filtered_splits['train'][0]),
                'val_images': len(filtered_splits['val'][0]),
                'test_images': len(filtered_splits['test'][0])
            },
            'text_only_baseline': {
                'test_metrics': text_only_results['overall_metrics'],
                'training_info': {
                    'best_epoch': text_only_training['best_epoch'],
                    'best_val_loss': text_only_training['best_val_loss']
                }
            },
            'single_modal_baseline': {
                'test_metrics': single_results['overall_metrics'],
                'training_info': {
                    'best_epoch': single_training['best_epoch'],
                    'best_val_loss': single_training['best_val_loss']
                }
            },
            'multimodal_baseline': {
                'test_metrics': multimodal_results['overall_metrics'],
                'training_info': {
                    'best_epoch': multimodal_training['best_epoch'],
                    'best_val_loss': multimodal_training['best_val_loss']
                }
            }
        }

        experiment_logger.info(f"✅ Stratified experiment {run_id} completed successfully")
        return experiment_result

    except Exception as e:
        experiment_logger.error(f"❌ Experiment {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'run_id': run_id, 'seed': seed, 'error': str(e)}


def compute_stratified_statistics(successful_experiments: List[Dict]) -> Dict[str, Any]:
    

    # Extract metrics for all three models
    text_only_acc = [exp['text_only_baseline']['test_metrics']['accuracy'] for exp in successful_experiments]
    single_acc = [exp['single_modal_baseline']['test_metrics']['accuracy'] for exp in successful_experiments]
    multimodal_acc = [exp['multimodal_baseline']['test_metrics']['accuracy'] for exp in successful_experiments]

    text_only_f1 = [exp['text_only_baseline']['test_metrics']['f1_score'] for exp in successful_experiments]
    single_f1 = [exp['single_modal_baseline']['test_metrics']['f1_score'] for exp in successful_experiments]
    multimodal_f1 = [exp['multimodal_baseline']['test_metrics']['f1_score'] for exp in successful_experiments]

    text_only_mcc = [exp['text_only_baseline']['test_metrics']['mcc'] for exp in successful_experiments]
    single_mcc = [exp['single_modal_baseline']['test_metrics']['mcc'] for exp in successful_experiments]
    multimodal_mcc = [exp['multimodal_baseline']['test_metrics']['mcc'] for exp in successful_experiments]

    def analyze_paired_metric(baseline_vals, test_vals, metric_name):
        baseline_vals = np.array(baseline_vals)
        test_vals = np.array(test_vals)

        improvements = test_vals - baseline_vals
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements, ddof=1)

        t_stat, p_value = stats.ttest_rel(test_vals, baseline_vals)
        cohens_d = mean_improvement / std_improvement if std_improvement > 0 else 0

        n = len(improvements)
        se = std_improvement / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n - 1, loc=mean_improvement, scale=se)

        better_count = np.sum(test_vals > baseline_vals)
        equal_count = np.sum(test_vals == baseline_vals)
        worse_count = np.sum(test_vals < baseline_vals)

        return {
            f'{metric_name}_baseline_mean': float(np.mean(baseline_vals)),
            f'{metric_name}_test_mean': float(np.mean(test_vals)),
            f'{metric_name}_mean_improvement': float(mean_improvement),
            f'{metric_name}_relative_improvement_pct': float((mean_improvement / np.mean(baseline_vals)) * 100),
            f'{metric_name}_std_improvement': float(std_improvement),
            f'{metric_name}_effect_size_cohens_d': float(cohens_d),
            f'{metric_name}_p_value': float(p_value),
            f'{metric_name}_t_statistic': float(t_stat),
            f'{metric_name}_confidence_interval_95': [float(ci_95[0]), float(ci_95[1])],
            f'{metric_name}_test_better_count': int(better_count),
            f'{metric_name}_test_equal_count': int(equal_count),
            f'{metric_name}_test_worse_count': int(worse_count),
            f'{metric_name}_test_better_rate': float(better_count / n),
            f'{metric_name}_significant_at_05': str(p_value < 0.05),
            f'{metric_name}_significant_at_01': str(p_value < 0.01),
            f'{metric_name}_significant_at_001': str(p_value < 0.001)
        }

    # All 3-way comparisons
    single_vs_text_acc = analyze_paired_metric(text_only_acc, single_acc, 'accuracy')
    single_vs_text_f1 = analyze_paired_metric(text_only_f1, single_f1, 'f1_score')
    single_vs_text_mcc = analyze_paired_metric(text_only_mcc, single_mcc, 'mcc')

    multimodal_vs_single_acc = analyze_paired_metric(single_acc, multimodal_acc, 'accuracy')
    multimodal_vs_single_f1 = analyze_paired_metric(single_f1, multimodal_f1, 'f1_score')
    multimodal_vs_single_mcc = analyze_paired_metric(single_mcc, multimodal_mcc, 'mcc')

    multimodal_vs_text_acc = analyze_paired_metric(text_only_acc, multimodal_acc, 'accuracy')
    multimodal_vs_text_f1 = analyze_paired_metric(text_only_f1, multimodal_f1, 'f1_score')
    multimodal_vs_text_mcc = analyze_paired_metric(text_only_mcc, multimodal_mcc, 'mcc')

    return {
        'num_experiments': len(successful_experiments),
        'single_modal_vs_text_only': {
            'accuracy': single_vs_text_acc,
            'f1_score': single_vs_text_f1,
            'mcc': single_vs_text_mcc
        },
        'multimodal_vs_single_modal': {
            'accuracy': multimodal_vs_single_acc,
            'f1_score': multimodal_vs_single_f1,
            'mcc': multimodal_vs_single_mcc
        },
        'multimodal_vs_text_only_total': {
            'accuracy': multimodal_vs_text_acc,
            'f1_score': multimodal_vs_text_f1,
            'mcc': multimodal_vs_text_mcc
        }
    }


def run_stratified_experiment(dataset_path: str, balance_strategy: str,
                              output_base_dir: str, num_runs: int = 10) -> Dict[str, Any]:
    

    experiment_name = f"stratified_frozen_3way_{balance_strategy.replace(':', '_')}"
    output_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    from utils import setup_logging
    log_file = os.path.join(output_dir, 'stratified_experiment_log.txt')
    logger = setup_logging("INFO", log_file)

    logger.info(f"🎯 STRATIFIED EXPERIMENT (NON-OOD) - 3-WAY COMPARISON")
    logger.info(f"Method: {num_runs} runs with STRATIFIED random splits")
    logger.info(f"  1. Text-Only (captions only)")
    logger.info(f"  2. Single Modal (vision only)")
    logger.info(f"  3. Multimodal (custom captions)")
    
    logger.info("=" * 100)

    # Handle captions
    logger.info("\n🔍 Setting up captions...")

    ood_captions = f"1on10_ood_results/ood_frozen_3way_{balance_strategy.replace(':', '_')}/ood_experiment_captions.json"
    captions_file = os.path.join(output_dir, 'stratified_experiment_captions.json')

    if os.path.exists(ood_captions):
        logger.info(f"♻️ Reusing existing captions from OOD experiment: {ood_captions}")
        captions_file = ood_captions
    elif os.path.exists(captions_file):
        logger.info(f"✅ Using existing captions from {captions_file}")
    else:
        logger.info(f"Generating new captions...")
        import subprocess
        import sys

        caption_cmd = [
            sys.executable, 'caption_generator.py',
            '--dataset_path', dataset_path,
            '--output_file', captions_file,
            '--balance_strategy', balance_strategy,
            '--device', 'auto'
        ]

        try:
            subprocess.run(caption_cmd, check=True)
            logger.info(f"✅ Captions generated and saved to {captions_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Caption generation failed: {e}")
            raise

    # Run experiments
    experiment_results = {
        'experiment_name': experiment_name,
        'split_type': 'stratified_random',
        'dataset_path': dataset_path,
        'balance_strategy': balance_strategy,
        'num_runs': num_runs,
        'start_time': datetime.now().isoformat(),
        'successful_experiments': [],
        'failed_experiments': []
    }

    for i in range(num_runs):
        seed = 42 + i * 100
        result = run_single_stratified_experiment(
            dataset_path, balance_strategy, captions_file,
            output_dir, i + 1, seed, logger
        )

        if 'error' in result:
            experiment_results['failed_experiments'].append(result)
        else:
            experiment_results['successful_experiments'].append(result)

    # Statistical analysis
    successful_runs = len(experiment_results['successful_experiments'])
    if successful_runs >= 3:
        logger.info(f"\n📊 Computing statistical analysis...")
        stats_results = compute_stratified_statistics(experiment_results['successful_experiments'])
        experiment_results['statistical_analysis'] = stats_results

    # Save results
    experiment_results['end_time'] = datetime.now().isoformat()
    experiment_results['successful_runs'] = successful_runs
    experiment_results['failed_runs'] = len(experiment_results['failed_experiments'])

    results_file = os.path.join(output_dir, 'stratified_frozen_3way_results.json')
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 100)
    logger.info("STRATIFIED 3-WAY COMPARISON RESULTS")
    logger.info("=" * 100)
    logger.info(f"Successful: {successful_runs}/{num_runs}")

    if 'statistical_analysis' in experiment_results:
        stats_data = experiment_results['statistical_analysis']

        logger.info(f"\n🎯 STATISTICAL ANALYSIS:")
        for comparison in ['single_modal_vs_text_only', 'multimodal_vs_single_modal',
                           'multimodal_vs_text_only_total']:
            logger.info(f"\n{comparison.upper()}:")
            for metric in ['accuracy', 'f1_score', 'mcc']:
                data = stats_data[comparison][metric]
                logger.info(f"  {metric}: {data[f'{metric}_mean_improvement']:+.4f} "
                            f"(p={data[f'{metric}_p_value']:.4f})")

    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info("=" * 100)

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run stratified experiments with 3-way comparison (no adversarial)")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--balance_strategy', type=str, default='1:10', help='Balance strategy')
    parser.add_argument('--output_dir', type=str, default='./stratified_results_3way', help='Output directory')
    parser.add_argument('--num_runs', type=int, default=36, help='Number of experimental runs')

    args = parser.parse_args()

    run_stratified_experiment(
        args.dataset_path,
        args.balance_strategy,
        args.output_dir,
        args.num_runs
    )


if __name__ == "__main__":
    main()