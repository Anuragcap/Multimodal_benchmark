import os
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from scipy import stats


def run_single_bioclip_stratified_experiment(dataset_path: str, balance_strategy: str,
                                             output_dir: str, run_id: int, seed: int,
                                             experiment_logger) -> Dict[str, Any]:
    """Run a single BioClip experiment with stratified splits"""
    
    experiment_logger.info(f"\n🧬 Running BioClip Stratified Experiment {run_id} (seed={seed})")
    experiment_logger.info("   Mode: STRATIFIED (same species in train/test - NOT OOD)")
    experiment_logger.info("-" * 80)
    
    try:
        from dataset import prepare_dataset, create_data_transforms
        from stratified_random_splitter import create_stratified_random_splits
        from config import create_config
        from utils import set_seed
        from bioclip_baseline import BioClipTrainer
        from dataset import create_dataloaders as create_single_dataloaders
        
        set_seed(seed)
        
        # Prepare dataset
        image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, experiment_logger)
        
        # Create STRATIFIED splits
        splits = create_stratified_random_splits(
            image_paths, labels, species_list,
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
            random_seed=seed, logger=experiment_logger
        )
        
        # Check overlap (should be HIGH for stratified)
        train_species = set(splits['train'][2])
        val_species = set(splits['val'][2])
        test_species = set(splits['test'][2])
        
        overlap_count = len(train_species & test_species)
        total_species = len(set(species_list))
        
        experiment_logger.info(f"📊 SPECIES DISTRIBUTION FOR RUN {run_id}:")
        experiment_logger.info(f"  Train Species: {len(train_species)}")
        experiment_logger.info(f"  Val Species:   {len(val_species)}")
        experiment_logger.info(f"  Test Species:  {len(test_species)}")
        experiment_logger.info(f"  Species overlap: {overlap_count}/{total_species} ({100*overlap_count/total_species:.1f}%) ✅ EXPECTED")
        
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
        
        # Train BioClip
        experiment_logger.info(f"\n🧬 Training BioClip...")
        trainer = BioClipTrainer(config, experiment_logger)
        trainer.setup_training(splits['train'][1])
        training_results = trainer.train(dataloaders['train'], dataloaders['val'])
        test_results = trainer.evaluate(dataloaders['test'])
        
        metrics = test_results['overall_metrics']
        experiment_logger.info(f"  ✓ BioClip Test: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, MCC={metrics['mcc']:.4f}")
        
        # Compile results
        experiment_result = {
            'run_id': run_id,
            'seed': seed,
            'split_type': 'stratified',
            'species_splits': {
                'total_species': total_species,
                'train_test_overlap_count': overlap_count,
                'train_test_overlap_percent': 100 * overlap_count / total_species,
                'split_verification': 'stratified_with_overlap'
            },
            'dataset_sizes': {
                'train_images': len(splits['train'][0]),
                'val_images': len(splits['val'][0]),
                'test_images': len(splits['test'][0])
            },
            'test_metrics': metrics,
            'training_info': {
                'best_epoch': training_results['best_epoch'],
                'best_val_loss': training_results['best_val_loss']
            }
        }
        
        experiment_logger.info(f"✅ BioClip stratified experiment {run_id} completed successfully")
        return experiment_result
        
    except Exception as e:
        experiment_logger.error(f"❌ Experiment {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'run_id': run_id, 'seed': seed, 'error': str(e)}


def compute_statistics(successful_experiments: List[Dict]) -> Dict[str, Any]:
    """Compute statistics across all experiments"""
    
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


def run_bioclip_stratified_experiment(dataset_path: str, balance_strategy: str, 
                                      output_dir: str, num_runs: int = 36):
    """Run BioClip with stratified splits - 36 different seeds"""
    
    from utils import setup_logging
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'bioclip_stratified_36runs.log')
    logger = setup_logging("INFO", log_file)
    
    logger.info("="*100)
    logger.info("BIOCLIP STRATIFIED EXPERIMENT - 36 RUNS")
    logger.info("="*100)
    logger.info(f"Model: BioClip (CLIP trained on iNaturalist 2021)")
    logger.info(f"Goal: Test BioClip WITHOUT distribution shift")
    logger.info(f"Method: {num_runs} runs with STRATIFIED random splits")
    logger.info(f"Note: Same species appear in train AND test (NOT OOD)")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Balance Strategy: {balance_strategy}")
    logger.info("="*100)
    
    # Run experiments
    experiment_results = {
        'experiment_type': 'bioclip_stratified_36runs',
        'model': 'BioClip (CLIP on iNaturalist)',
        'split_type': 'stratified',
        'num_runs': num_runs,
        'timestamp': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'balance_strategy': balance_strategy,
        'successful_experiments': [],
        'failed_experiments': []
    }
    
    logger.info(f"\n🔬 Running {num_runs} experiments...")
    
    for i in range(num_runs):
        run_id = i + 1
        seed = 42 + i * 100
        
        result = run_single_bioclip_stratified_experiment(
            dataset_path, balance_strategy, output_dir,
            run_id, seed, logger
        )
        
        if result and 'error' not in result:
            experiment_results['successful_experiments'].append(result)
            logger.info(f"✅ Completed {run_id}/{num_runs}")
        else:
            experiment_results['failed_experiments'].append(result)
            logger.error(f"❌ Failed {run_id}/{num_runs}")
    
    # Compute statistics
    successful_runs = len(experiment_results['successful_experiments'])
    if successful_runs >= 3:
        logger.info(f"\n📊 Computing statistics from {successful_runs} successful runs...")
        stats_results = compute_statistics(experiment_results['successful_experiments'])
        experiment_results['statistics'] = stats_results
        
        logger.info(f"\n{'='*100}")
        logger.info("BIOCLIP STRATIFIED RESULTS SUMMARY")
        logger.info(f"{'='*100}")
        logger.info(f"Successful: {successful_runs}/{num_runs}")
        logger.info(f"\nAccuracy: {stats_results['accuracy']['mean']:.4f} ± {stats_results['accuracy']['std']:.4f}")
        logger.info(f"F1-Score: {stats_results['f1_score']['mean']:.4f} ± {stats_results['f1_score']['std']:.4f}")
        logger.info(f"MCC:      {stats_results['mcc']['mean']:.4f} ± {stats_results['mcc']['std']:.4f}")
    
    # Save results
    experiment_results['successful_runs'] = successful_runs
    experiment_results['failed_runs'] = len(experiment_results['failed_experiments'])
    experiment_results['end_time'] = datetime.now().isoformat()
    
    results_file = os.path.join(output_dir, 'bioclip_stratified_36runs_results.json')
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"💾 Results saved to: {results_file}")
    logger.info(f"📋 Log saved to: {log_file}")
    logger.info(f"{'='*100}")
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="BioClip Stratified 36 Runs Experiment")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--balance_strategy', type=str, default='1:10')
    parser.add_argument('--output_dir', type=str, default='./bioclip_stratified_36runs')
    parser.add_argument('--num_runs', type=int, default=36)
    
    args = parser.parse_args()
    
    run_bioclip_stratified_experiment(
        args.dataset_path,
        args.balance_strategy,
        args.output_dir,
        args.num_runs
    )


if __name__ == "__main__":
    main()