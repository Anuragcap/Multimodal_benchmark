import os
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from scipy import stats


def run_single_dinov2_ood_experiment(dataset_path: str, balance_strategy: str,
                                     output_dir: str, run_id: int, seed: int,
                                     experiment_logger, test_species_pair: tuple = None) -> Dict[str, Any]:
    
    
    experiment_logger.info(f"\n🦖 Running DINOv2 OOD Experiment {run_id} (seed={seed})")
    if test_species_pair:
        experiment_logger.info(f"   Test Species: {test_species_pair}")
    experiment_logger.info("-" * 80)
    
    try:
        from dataset import prepare_dataset, create_data_transforms
        from random_species_splitter import create_random_species_splits
        from config import create_config
        from utils import setup_logging, set_seed
        from dinov2_baseline import DINOv2Trainer
        from dataset import create_dataloaders as create_single_dataloaders
        
        set_seed(seed)
        
        # Prepare dataset
        image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, experiment_logger)
        
        # Create OOD splits with specific test species
        splits = create_random_species_splits(
            image_paths, labels, species_list,
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
            random_seed=seed, logger=experiment_logger,
            test_species_override=test_species_pair
        )
        
        # Verify no species overlap
        train_species = set(splits['train'][2])
        val_species = set(splits['val'][2])
        test_species = set(splits['test'][2])
        
        train_test_overlap = train_species & test_species
        if train_test_overlap:
            raise ValueError(f"Species overlap detected: {train_test_overlap}")
        
        experiment_logger.info(f"📊 SPECIES SPLITS FOR RUN {run_id}:")
        experiment_logger.info(f"  Train Species ({len(train_species)}): {sorted(train_species)}")
        experiment_logger.info(f"  Val Species   ({len(val_species)}): {sorted(val_species)}")
        experiment_logger.info(f"  Test Species  ({len(test_species)}): {sorted(test_species)}")
        experiment_logger.info(f"  ✅ No overlap confirmed")
        
        # Create config
        config = create_config(
            dataset_path=dataset_path,
            output_dir=output_dir,
            balance_strategy=balance_strategy,
            batch_size=32,
            num_epochs=15,
            learning_rate=1e-4,
            device="auto"
        )
        
        # Create transforms and dataloaders
        transforms_dict = create_data_transforms()
        dataloaders = create_single_dataloaders(
            splits, transforms_dict, config.data.batch_size, config.data.num_workers
        )
        
        # Train DINOv2
        experiment_logger.info(f"\n🦖 Training DINOv2...")
        trainer = DINOv2Trainer(config, experiment_logger)
        trainer.setup_training(splits['train'][1])
        training_results = trainer.train(dataloaders['train'], dataloaders['val'])
        test_results = trainer.evaluate(dataloaders['test'])
        
        metrics = test_results['overall_metrics']
        experiment_logger.info(f"  ✓ DINOv2 Test: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, MCC={metrics['mcc']:.4f}")
        
        # Compile results
        experiment_result = {
            'run_id': run_id,
            'test_species_pair': list(test_species_pair) if test_species_pair else None,
            'seed': seed,
            'split_type': 'ood',
            'species_splits': {
                'train_species': sorted(train_species),
                'val_species': sorted(val_species),
                'test_species': sorted(test_species),
                'no_overlap': True
            },
            'test_metrics': metrics,
            'training_info': {
                'best_epoch': training_results['best_epoch'],
                'best_val_loss': training_results['best_val_loss']
            }
        }
        
        experiment_logger.info(f"✅ DINOv2 OOD experiment {run_id} completed successfully")
        return experiment_result
        
    except Exception as e:
        experiment_logger.error(f"❌ Experiment {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'run_id': run_id, 'seed': seed, 'test_species': list(test_species_pair) if test_species_pair else None, 'error': str(e)}


def compute_statistics(successful_experiments: List[Dict]) -> Dict[str, Any]:
    
    
    accuracies = [exp['test_metrics']['accuracy'] for exp in successful_experiments]
    f1_scores = [exp['test_metrics']['f1_score'] for exp in successful_experiments]
    mccs = [exp['test_metrics']['mcc'] for exp in successful_experiments]
    
    return {
        'num_experiments': len(successful_experiments),
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies)),
            'median': float(np.median(accuracies))
        },
        'f1_score': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores)),
            'min': float(np.min(f1_scores)),
            'max': float(np.max(f1_scores)),
            'median': float(np.median(f1_scores))
        },
        'mcc': {
            'mean': float(np.mean(mccs)),
            'std': float(np.std(mccs)),
            'min': float(np.min(mccs)),
            'max': float(np.max(mccs)),
            'median': float(np.median(mccs))
        }
    }


def run_dinov2_ood_all_combinations(dataset_path: str, balance_strategy: str, output_dir: str):
    """Run DINOv2 on all 36 OOD species combinations"""
    
    from utils import setup_logging
    from dataset import prepare_dataset
    from random_species_splitter import get_all_species_combinations
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'dinov2_ood_all_combos.log')
    logger = setup_logging("INFO", log_file)
    
    logger.info("="*100)
    logger.info("DINOv2 OOD EXPERIMENT - ALL 36 SPECIES COMBINATIONS")
    logger.info("="*100)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Balance Strategy: {balance_strategy}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*100)
    
    # Get all species combinations
    image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)
    all_test_combinations = get_all_species_combinations(species_list, n_test_species=2)
    
    logger.info(f"\n📋 Testing ALL {len(all_test_combinations)} species combinations:")
    for i, combo in enumerate(all_test_combinations, 1):
        logger.info(f"  {i:2d}. {combo[0]} + {combo[1]}")
    
    # Run experiments
    experiment_results = {
        'experiment_type': 'dinov2_ood_all_combinations',
        'model': 'DINOv2-base',
        'num_combinations': len(all_test_combinations),
        'timestamp': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'balance_strategy': balance_strategy,
        'successful_experiments': [],
        'failed_experiments': []
    }
    
    logger.info(f"\n🔬 Running {len(all_test_combinations)} experiments...")
    
    for run_idx, test_species_pair in enumerate(all_test_combinations):
        run_id = run_idx + 1
        seed = 42 + run_idx * 100
        
        result = run_single_dinov2_ood_experiment(
            dataset_path, balance_strategy, output_dir,
            run_id, seed, logger, test_species_pair=test_species_pair
        )
        
        if result and 'error' not in result:
            experiment_results['successful_experiments'].append(result)
            logger.info(f"✅ Completed {run_id}/{len(all_test_combinations)}: {test_species_pair}")
        else:
            experiment_results['failed_experiments'].append(result)
            logger.error(f"❌ Failed {run_id}/{len(all_test_combinations)}: {test_species_pair}")
    
    # Compute statistics
    successful_runs = len(experiment_results['successful_experiments'])
    if successful_runs >= 3:
        logger.info(f"\n📊 Computing statistics from {successful_runs} successful runs...")
        stats_results = compute_statistics(experiment_results['successful_experiments'])
        experiment_results['statistics'] = stats_results
        
        logger.info(f"\n{'='*100}")
        logger.info("DINOV2 OOD RESULTS SUMMARY")
        logger.info(f"{'='*100}")
        logger.info(f"Successful: {successful_runs}/{len(all_test_combinations)}")
        logger.info(f"\nAccuracy: {stats_results['accuracy']['mean']:.4f} ± {stats_results['accuracy']['std']:.4f}")
        logger.info(f"F1-Score: {stats_results['f1_score']['mean']:.4f} ± {stats_results['f1_score']['std']:.4f}")
        logger.info(f"MCC:      {stats_results['mcc']['mean']:.4f} ± {stats_results['mcc']['std']:.4f}")
    
    # Save results
    experiment_results['successful_runs'] = successful_runs
    experiment_results['failed_runs'] = len(experiment_results['failed_experiments'])
    experiment_results['end_time'] = datetime.now().isoformat()
    
    results_file = os.path.join(output_dir, 'dinov2_ood_all_combos_results.json')
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"💾 Results saved to: {results_file}")
    logger.info(f"📋 Log saved to: {log_file}")
    logger.info(f"{'='*100}")
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="DINOv2 OOD All Combinations Experiment")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--balance_strategy', type=str, default='1:10')
    parser.add_argument('--output_dir', type=str, default='./dinov2_ood_all_combos')
    
    args = parser.parse_args()
    
    run_dinov2_ood_all_combinations(
        args.dataset_path,
        args.balance_strategy,
        args.output_dir
    )


if __name__ == "__main__":
    main()