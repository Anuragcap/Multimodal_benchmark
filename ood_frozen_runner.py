import os
import json
import argparse
import sys
import subprocess
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from scipy import stats


def run_single_ood_experiment(dataset_path: str, balance_strategy: str,
                               captions_file: str, output_dir: str,
                               run_id: int, seed: int,
                               experiment_logger,
                               test_species_pair: tuple = None) -> Dict[str, Any]:
    

    experiment_logger.info(f"\n🔬 Running OOD Experiment {run_id} (seed={seed})")
    if test_species_pair:
        experiment_logger.info(f"   Test Species: {test_species_pair}")
    experiment_logger.info("   3-way comparison: Text-Only → Single → Multimodal")
    experiment_logger.info("-" * 80)

    try:
        from dataset import prepare_dataset, create_data_transforms
        from random_species_splitter import create_random_species_splits, get_all_species_combinations
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

        # Create species-aware OOD splits
        splits = create_random_species_splits(
            image_paths, labels, species_list,
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
            random_seed=seed, logger=experiment_logger,
            test_species_override=test_species_pair
        )

        # Species distribution for logging / overlap check
        unique_species = sorted(set(species_list))
        train_sp = set(splits['train'][2])
        val_sp = set(splits['val'][2])
        test_sp = set(splits['test'][2])

        experiment_logger.info(f"📊 SPECIES SPLITS FOR RUN {run_id}:")
        experiment_logger.info(f"  Train Species ({len(train_sp)}): {sorted(train_sp)}")
        experiment_logger.info(f"  Val Species   ({len(val_sp)}): {sorted(val_sp)}")
        experiment_logger.info(f"  Test Species  ({len(test_sp)}): {sorted(test_sp)}")

        # Verify no overlap (critical for OOD)
        if (train_sp & test_sp) or (train_sp & val_sp) or (val_sp & test_sp):
            raise ValueError("Species overlap detected in OOD splits!")

        # Load config and captions
        config = create_config(
            dataset_path=dataset_path,
            output_dir=output_dir,
            balance_strategy=balance_strategy,
            batch_size=16,
            num_epochs=15,
            learning_rate=1e-4,
            device="auto"
        )

        captions_data = load_captions_data(captions_file, experiment_logger)
        if captions_data is None:
            raise ValueError("Failed to load captions")

        clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
        transforms_dict = create_data_transforms()

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

        
        experiment_logger.info(f"\n🖼️ Training Single Modal (Vision-Only)...")
        single_dataloaders = create_single_dataloaders(
            filtered_splits, transforms_dict, config.data.batch_size, config.data.num_workers
        )

        single_trainer = SingleModalTrainer(config, experiment_logger)
        single_trainer.setup_training(filtered_splits['train'][1])
        single_training = single_trainer.train(single_dataloaders['train'], single_dataloaders['val'])
        single_results = single_trainer.evaluate(single_dataloaders['test'])

        sm = single_results['overall_metrics']
        experiment_logger.info(f"  ✓ Single Modal: Acc={sm['accuracy']:.4f}, F1={sm['f1_score']:.4f}, MCC={sm['mcc']:.4f}")
        experiment_logger.info(f"    vs Text-Only: {sm['accuracy'] - to['accuracy']:+.4f}")

        
        experiment_logger.info(f"\n📄 Training Multimodal (Custom Captions)...")
        multimodal_dataloaders = create_single_dataloaders(
            filtered_splits, transforms_dict, config.data.batch_size,
            config.data.num_workers, captions_data, clip_tokenizer
        )

        multimodal_trainer = MultiModalTrainer(config, experiment_logger)
        multimodal_trainer.setup_training(filtered_splits['train'][1])
        multimodal_training = multimodal_trainer.train(
            multimodal_dataloaders['train'], multimodal_dataloaders['val']
        )
        multimodal_results = multimodal_trainer.evaluate(multimodal_dataloaders['test'])

        mm = multimodal_results['overall_metrics']
        experiment_logger.info(f"  ✓ Multimodal: Acc={mm['accuracy']:.4f}, F1={mm['f1_score']:.4f}, MCC={mm['mcc']:.4f}")
        experiment_logger.info(f"    vs Single Modal: {mm['accuracy'] - sm['accuracy']:+.4f}")

        
        experiment_logger.info(f"\n📈 OOD RUN {run_id} COMPLETE - 3-WAY COMPARISON:")
        if test_species_pair:
            experiment_logger.info(f"  Test Species: {test_species_pair}")
        experiment_logger.info(f"  1. Text-Only:        {to['accuracy']:.4f}")
        experiment_logger.info(f"  2. Single Modal:     {sm['accuracy']:.4f} ({sm['accuracy'] - to['accuracy']:+.4f})")
        experiment_logger.info(f"  3. Multimodal:       {mm['accuracy']:.4f} ({mm['accuracy'] - sm['accuracy']:+.4f})")
        experiment_logger.info(f"  Total Improvement:   {mm['accuracy'] - to['accuracy']:+.4f} ({(mm['accuracy'] - to['accuracy'])/to['accuracy']*100:+.2f}%)")

        # Compile results
        experiment_result = {
            'run_id': run_id,
            'test_species_pair': list(test_species_pair) if test_species_pair else None,
            'seed': seed,
            'species_splits': {
                'train_species': sorted(train_sp),
                'val_species': sorted(val_sp),
                'test_species': sorted(test_sp),
                'no_overlap': True
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
            'multimodal_custom_captions': {
                'test_metrics': multimodal_results['overall_metrics'],
                'training_info': {
                    'best_epoch': multimodal_training['best_epoch'],
                    'best_val_loss': multimodal_training['best_val_loss']
                }
            }
        }

        experiment_logger.info(f"✅ OOD experiment {run_id} completed successfully")
        return experiment_result

    except Exception as e:
        experiment_logger.error(f"❌ Experiment {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_ood_statistics(results: List[Dict]) -> Dict[str, Any]:
    

    def analyze_paired_metric(baseline, test, metric_name):
        """Analyze paired comparisons"""
        baseline_arr = np.array(baseline)
        test_arr = np.array(test)
        improvements = test_arr - baseline_arr

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(test_arr, baseline_arr)

        # Effect size (Cohen's d for paired samples)
        mean_diff = np.mean(improvements)
        std_diff = np.std(improvements, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        # Confidence interval
        se = std_diff / np.sqrt(len(improvements))
        ci_95 = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)

        # Count improvements
        better = np.sum(improvements > 0)
        equal = np.sum(improvements == 0)
        worse = np.sum(improvements < 0)

        return {
            f'{metric_name}_baseline_mean': float(np.mean(baseline_arr)),
            f'{metric_name}_test_mean': float(np.mean(test_arr)),
            f'{metric_name}_mean_improvement': float(mean_diff),
            f'{metric_name}_relative_improvement_pct': float(100 * mean_diff / np.mean(baseline_arr)),
            f'{metric_name}_p_value': float(p_value),
            f'{metric_name}_t_statistic': float(t_stat),
            f'{metric_name}_significant_at_05': str(p_value < 0.05),
            f'{metric_name}_effect_size_cohens_d': float(cohens_d),
            f'{metric_name}_confidence_interval_95': [float(ci_95[0]), float(ci_95[1])],
            f'{metric_name}_std_improvement': float(std_diff),
            f'{metric_name}_better_equal_worse': f"{better}/{equal}/{worse}"
        }

    # Extract all metrics for 3 models
    text_only_acc = [exp['text_only_baseline']['test_metrics']['accuracy'] for exp in results]
    text_only_f1 = [exp['text_only_baseline']['test_metrics']['f1_score'] for exp in results]
    text_only_mcc = [exp['text_only_baseline']['test_metrics']['mcc'] for exp in results]

    single_acc = [exp['single_modal_baseline']['test_metrics']['accuracy'] for exp in results]
    single_f1 = [exp['single_modal_baseline']['test_metrics']['f1_score'] for exp in results]
    single_mcc = [exp['single_modal_baseline']['test_metrics']['mcc'] for exp in results]

    multi_acc = [exp['multimodal_custom_captions']['test_metrics']['accuracy'] for exp in results]
    multi_f1 = [exp['multimodal_custom_captions']['test_metrics']['f1_score'] for exp in results]
    multi_mcc = [exp['multimodal_custom_captions']['test_metrics']['mcc'] for exp in results]

    return {
        'num_experiments': len(results),
        'single_modal_vs_text_only': {
            'accuracy': analyze_paired_metric(text_only_acc, single_acc, 'accuracy'),
            'f1_score': analyze_paired_metric(text_only_f1, single_f1, 'f1_score'),
            'mcc': analyze_paired_metric(text_only_mcc, single_mcc, 'mcc')
        },
        'multimodal_vs_single_modal': {
            'accuracy': analyze_paired_metric(single_acc, multi_acc, 'accuracy'),
            'f1_score': analyze_paired_metric(single_f1, multi_f1, 'f1_score'),
            'mcc': analyze_paired_metric(single_mcc, multi_mcc, 'mcc')
        },
        'multimodal_vs_text_only_total': {
            'accuracy': analyze_paired_metric(text_only_acc, multi_acc, 'accuracy'),
            'f1_score': analyze_paired_metric(text_only_f1, multi_f1, 'f1_score'),
            'mcc': analyze_paired_metric(text_only_mcc, multi_mcc, 'mcc')
        }
    }


def run_ood_generalization_experiment(dataset_path: str, balance_strategy: str,
                                      output_base_dir: str,
                                      use_all_combinations: bool = True) -> Dict[str, Any]:
    

    experiment_name = f"ood_frozen_3way_{balance_strategy.replace(':', '_')}"
    output_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    from utils import setup_logging
    log_file = os.path.join(output_dir, 'ood_experiment_log.txt')
    logger = setup_logging("INFO", log_file)

    logger.info(f"🧬 OOD GENERALIZATION EXPERIMENT - 3-WAY COMPARISON")
    logger.info(f"Mode: ALL 36 SPECIES COMBINATIONS")
    logger.info(f"Comparing:")
    logger.info(f"  1. Text-Only (captions only)")
    logger.info(f"  2. Single Modal (vision only)")
    logger.info(f"  3. Multimodal (custom captions)")
    
    logger.info(f"Output: {output_dir}")

    # Get all 36 combinations
    from dataset import prepare_dataset
    from random_species_splitter import get_all_species_combinations

    # Load dataset to get species list
    image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)
    all_test_combinations = get_all_species_combinations(species_list, n_test_species=2)

    logger.info(f"\n📋 Testing ALL {len(all_test_combinations)} species combinations:")
    for i, combo in enumerate(all_test_combinations, 1):
        logger.info(f"  {i:2d}. {combo[0]} + {combo[1]}")

    num_experiments = len(all_test_combinations)

    # Generate captions once
    logger.info(f"\n🔍 Generating captions...")
    captions_file = os.path.join(output_dir, 'ood_experiment_captions.json')

    if not os.path.exists(captions_file):
        caption_cmd = [
            sys.executable, 'caption_generator.py',
            '--dataset_path', dataset_path,
            '--output_file', captions_file,
            '--balance_strategy', balance_strategy,
            '--device', 'auto'
        ]

        try:
            subprocess.run(caption_cmd, check=True)
            logger.info(f"✅ Captions generated")
        except subprocess.CalledProcessError:
            logger.error(f"❌ Caption generation failed")
            return {}
    else:
        logger.info(f"✅ Using existing captions")

    # Run experiments
    experiment_results = {
        'experiment_type': 'ood_all_combinations_3way',
        'experiment_name': experiment_name,
        'balance_strategy': balance_strategy,
        'num_runs': num_experiments,
        'start_time': datetime.now().isoformat(),
        'successful_experiments': [],
        'failed_experiments': []
    }

    logger.info(f"\n🔬 Running {num_experiments} OOD experiments (3 models each)...")

    for run_idx in range(num_experiments):
        run_id = run_idx + 1
        seed = 42 + run_idx * 100
        test_species_pair = all_test_combinations[run_idx]

        result = run_single_ood_experiment(
            dataset_path, balance_strategy, captions_file,
            output_dir, run_id, seed, logger,
            test_species_pair=test_species_pair
        )

        if result is not None:
            experiment_results['successful_experiments'].append(result)
            logger.info(f"✅ Completed {run_id}/{num_experiments}: {test_species_pair}")
        else:
            experiment_results['failed_experiments'].append({
                'run_id': run_id,
                'test_species': list(test_species_pair),
                'seed': seed
            })
            logger.error(f"❌ Failed {run_id}/{num_experiments}")

    # Statistical analysis
    successful_runs = len(experiment_results['successful_experiments'])
    if successful_runs >= 3:
        logger.info(f"\n📊 Computing statistical analysis...")
        stats_results = compute_ood_statistics(experiment_results['successful_experiments'])
        experiment_results['statistical_analysis'] = stats_results

    # Save results
    experiment_results['end_time'] = datetime.now().isoformat()
    experiment_results['successful_runs'] = successful_runs
    experiment_results['failed_runs'] = len(experiment_results['failed_experiments'])

    results_file = os.path.join(output_dir, 'ood_generalization_3way_results.json')
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 100)
    logger.info("OOD 3-WAY COMPARISON RESULTS - ALL 36 COMBINATIONS")
    logger.info("=" * 100)
    logger.info(f"Successful: {successful_runs}/{num_experiments}")

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
    logger.info(f"\n🎉 ALL {num_experiments} COMBINATIONS COMPLETE!")
    logger.info(f"   Successful: {successful_runs}/{num_experiments}")
    logger.info(f"   Failed: {len(experiment_results['failed_experiments'])}/{num_experiments}")
    logger.info("=" * 100)

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run OOD experiments with 3-way comparison (no adversarial)")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--balance_strategy', type=str, default='1:10')
    parser.add_argument('--output_dir', type=str, default='./ood_3way_results')

    args = parser.parse_args()

    print("✅ Running ALL 36 OOD combinations with 3-way comparison")
    print("   This tests every possible 2-species OOD scenario")
    print("   Each combination runs 3 models: Text → Vision → Multimodal")

    run_ood_generalization_experiment(
        args.dataset_path,
        args.balance_strategy,
        args.output_dir,
        use_all_combinations=True
    )


if __name__ == "__main__":
    main()