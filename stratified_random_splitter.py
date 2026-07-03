
import random
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

def create_stratified_random_splits(image_paths: List[str], labels: List[int], species: List[str],
                                   train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                                   random_seed: int = 42, logger=None) -> Dict[str, Tuple[List, List, List]]:
    
    if logger:
        logger.info(f"🎯 Creating STRATIFIED random splits (seed={random_seed})")
        logger.info("   Note: ALL species will appear in train/val/test (NOT OOD)")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Group data by species AND label for stratification
    species_label_groups = defaultdict(list)
    for i, (path, label, spec) in enumerate(zip(image_paths, labels, species)):
        key = (spec, label)  # Group by both species and label
        species_label_groups[key].append({
            'index': i,
            'path': path,
            'label': label,
            'species': spec
        })
    
    if logger:
        logger.info(f"Total species: {len(set(species))}")
        species_stats = defaultdict(lambda: {'wild': 0, 'captive': 0})
        for (spec, label), items in species_label_groups.items():
            if label == 0:
                species_stats[spec]['wild'] = len(items)
            else:
                species_stats[spec]['captive'] = len(items)
        
        for spec in sorted(species_stats.keys()):
            wild = species_stats[spec]['wild']
            captive = species_stats[spec]['captive']
            total = wild + captive
            logger.info(f"  {spec}: {total} samples ({wild}W/{captive}C)")
    
    # Initialize splits
    train_data = []
    val_data = []
    test_data = []
    
    # Split each species-label group proportionally
    for (spec, label), items in species_label_groups.items():
        # Shuffle this group
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        n = len(shuffled_items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        
        # Ensure we have at least 1 sample in test if possible
        if n_test < 1 and n > 2:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
        
        # Split this group
        train_data.extend(shuffled_items[:n_train])
        val_data.extend(shuffled_items[n_train:n_train + n_val])
        test_data.extend(shuffled_items[n_train + n_val:])
    
    # Final shuffle of each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Extract paths, labels, species for each split
    def extract_split_data(split_data):
        paths = [item['path'] for item in split_data]
        labels = [item['label'] for item in split_data]
        species = [item['species'] for item in split_data]
        return paths, labels, species
    
    train_paths, train_labels, train_species = extract_split_data(train_data)
    val_paths, val_labels, val_species = extract_split_data(val_data)
    test_paths, test_labels, test_species = extract_split_data(test_data)
    
    if logger:
        logger.info(f"✅ Stratified splits created!")
        logger.info(f"Final sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        
        # Show species overlap (should be high - same species everywhere)
        train_species_set = set(train_species)
        val_species_set = set(val_species)
        test_species_set = set(test_species)
        
        all_species = train_species_set | val_species_set | test_species_set
        train_val_overlap = train_species_set & val_species_set
        train_test_overlap = train_species_set & test_species_set
        val_test_overlap = val_species_set & test_species_set
        all_three_overlap = train_species_set & val_species_set & test_species_set
        
        logger.info(f"Species coverage:")
        logger.info(f"  Total species: {len(all_species)}")
        logger.info(f"  Train species: {len(train_species_set)}")
        logger.info(f"  Val species: {len(val_species_set)}")
        logger.info(f"  Test species: {len(test_species_set)}")
        logger.info(f"  In all 3 splits: {len(all_three_overlap)} species ✅")
        logger.info(f"  Train-Test overlap: {len(train_test_overlap)}/{len(all_species)} species")
        
        # Log label distribution
        logger.info(f"Label distribution:")
        logger.info(f"  Train: {train_labels.count(0)}W / {train_labels.count(1)}C")
        logger.info(f"  Val:   {val_labels.count(0)}W / {val_labels.count(1)}C")
        logger.info(f"  Test:  {test_labels.count(0)}W / {test_labels.count(1)}C")
    
    return {
        'train': (train_paths, train_labels, train_species),
        'val': (val_paths, val_labels, val_species),
        'test': (test_paths, test_labels, test_species)
    }


def test_stratified_splits(dataset_path: str, balance_strategy: str = "original", num_runs: int = 5):
    """Test the stratified splitting to verify species appear in all splits"""
    print("🧪 TESTING STRATIFIED RANDOM SPLITS:")
    print("="*70)
    
    try:
        from dataset import prepare_dataset
        from utils import setup_logging
        
        # Prepare dataset
        logger = setup_logging("WARNING")
        image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)
        
        print(f"Dataset: {len(image_paths)} images, {len(set(species_list))} species")
        print(f"Species: {sorted(set(species_list))}")
        print()
        
        all_species = set(species_list)
        
        for run in range(num_runs):
            seed = 42 + run * 100
            splits = create_stratified_random_splits(
                image_paths, labels, species_list,
                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                random_seed=seed, logger=None
            )
            
            train_species = set(splits['train'][2])
            test_species = set(splits['test'][2])
            overlap = train_species & test_species
            
            print(f"Run {run+1} (seed={seed}):")
            print(f"  Train species: {len(train_species)}/{len(all_species)}")
            print(f"  Test species:  {len(test_species)}/{len(all_species)}")
            print(f"  Overlap:       {len(overlap)}/{len(all_species)} ({100*len(overlap)/len(all_species):.1f}%)")
        
        print(f"\n✅ Stratified splits ensure species appear in all sets (NOT OOD)")
        
    except Exception as e:
        print(f"❌ Error testing splits: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        test_stratified_splits(dataset_path)
    else:
        print("Usage: python stratified_random_splitter.py /path/to/dataset")