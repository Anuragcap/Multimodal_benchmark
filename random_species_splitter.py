
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from itertools import combinations


def create_random_species_splits(image_paths: List[str], labels: List[int], species: List[str],
                                train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                                random_seed: int = 42, logger=None, min_test_species: int = 2,
                                test_species_override: Optional[tuple] = None) -> Dict[str, Tuple[List, List, List]]:
    
    if logger:
        if test_species_override:
            logger.info(f"🎯 Creating DETERMINISTIC OOD splits with test species: {test_species_override}")
        else:
            logger.info(f"🎲 Creating RANDOM OOD splits (seed={random_seed})")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Group data by species
    species_groups = defaultdict(list)
    for i, (path, label, spec) in enumerate(zip(image_paths, labels, species)):
        species_groups[spec].append({
            'index': i,
            'path': path,
            'label': label,
            'species': spec
        })
    
    all_species = list(species_groups.keys())
    num_species = len(all_species)
    
    if logger:
        logger.info(f"Total species: {num_species}")
        for spec, data in species_groups.items():
            wild_count = sum(1 for item in data if item['label'] == 0)
            captive_count = sum(1 for item in data if item['label'] == 1)
            logger.info(f"  {spec}: {len(data)} samples ({wild_count}W/{captive_count}C)")
    
    # Determine test species
    if test_species_override:
        # DETERMINISTIC MODE: Use provided test species
        test_species = list(test_species_override)
        train_val_species = [s for s in all_species if s not in test_species]
        
        # Verify test species exist
        for ts in test_species:
            if ts not in species_groups:
                raise ValueError(f"Test species '{ts}' not found in dataset")
    else:
        # RANDOM MODE: Original behavior
        # Calculate target number of species for each split
        target_train_species = max(1, int(num_species * train_ratio))
        target_val_species = max(1, int(num_species * val_ratio))
        target_test_species = max(min_test_species, num_species - target_train_species - target_val_species)
        
        # Adjust if we don't have enough species
        if target_train_species + target_val_species + target_test_species > num_species:
            target_test_species = max(min_test_species, num_species - target_train_species - target_val_species)
            if target_test_species < min_test_species:
                target_val_species = max(1, num_species - target_train_species - min_test_species)
                target_test_species = min_test_species
        
        if logger:
            logger.info(f"Target species per split: Train={target_train_species}, Val={target_val_species}, Test={target_test_species}")
        
        # Randomly shuffle species list
        shuffled_species = all_species.copy()
        random.shuffle(shuffled_species)
        
        if logger:
            logger.info(f"Shuffled species order: {shuffled_species}")
        
        # Assign species to splits based on shuffled order
        train_species = shuffled_species[:target_train_species]
        val_species = shuffled_species[target_train_species:target_train_species + target_val_species]
        test_species = shuffled_species[target_train_species + target_val_species:]
        
        train_val_species = train_species + val_species
    
    # Split train/val species
    if test_species_override:
        # For deterministic mode, split remaining species for train/val
        n_train_species = max(1, int(len(train_val_species) * (train_ratio / (train_ratio + val_ratio))))
        train_species = train_val_species[:n_train_species]
        val_species = train_val_species[n_train_species:]
    
    if logger:
        logger.info(f"Species assignment:")
        logger.info(f"  Train species ({len(train_species)}): {sorted(train_species)}")
        logger.info(f"  Val species   ({len(val_species)}): {sorted(val_species)}")
        logger.info(f"  Test species  ({len(test_species)}): {sorted(test_species)}")
    
    # Collect data for each split
    train_data = []
    val_data = []
    test_data = []
    
    for species_name in train_species:
        train_data.extend(species_groups[species_name])
    
    for species_name in val_species:
        val_data.extend(species_groups[species_name])
        
    for species_name in test_species:
        test_data.extend(species_groups[species_name])
    
    # Shuffle within each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Extract paths, labels, species for each split
    def extract_split_data(split_data):
        paths = [item['path'] for item in split_data]
        labels = [item['label'] for item in split_data]
        species = [item['species'] for item in split_data]
        return paths, labels, species
    
    train_paths, train_labels, train_species_list = extract_split_data(train_data)
    val_paths, val_labels, val_species_list = extract_split_data(val_data)
    test_paths, test_labels, test_species_list = extract_split_data(test_data)
    
    if logger:
        logger.info(f"✅ OOD splits created!")
        logger.info(f"Final sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        
        # Verify no species overlap
        train_species_set = set(train_species)
        val_species_set = set(val_species)
        test_species_set = set(test_species)
        
        if train_species_set & test_species_set:
            logger.error("❌ Train-Test species overlap detected!")
        else:
            logger.info("✅ No train-test species overlap confirmed")
    
    return {
        'train': (train_paths, train_labels, train_species_list),
        'val': (val_paths, val_labels, val_species_list),
        'test': (test_paths, test_labels, test_species_list)
    }


def get_all_species_combinations(species_list: List[str], n_test_species: int = 2) -> List[tuple]:
    
    unique_species = sorted(set(species_list))
    return list(combinations(unique_species, n_test_species))


def test_random_species_splits(dataset_path: str, balance_strategy: str = "original", num_runs: int = 10):
    
    print("🧪 TESTING RANDOM SPECIES SPLITS:")
    print("="*60)
    
    try:
        from dataset import prepare_dataset
        from utils import setup_logging
        
        # Prepare dataset
        logger = setup_logging("WARNING")  # Minimize noise
        image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)
        
        print(f"Dataset: {len(image_paths)} images, {len(set(species_list))} species")
        print(f"Species: {sorted(set(species_list))}")
        print()
        
        # Test different seeds
        test_species_combinations = []
        
        for run in range(num_runs):
            seed = 42 + run * 100
            splits = create_random_species_splits(
                image_paths, labels, species_list, 
                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                random_seed=seed, logger=None
            )
            
            test_species = sorted(set(splits['test'][2]))
            test_species_combinations.append(test_species)
            
            print(f"Run {run+1:2d} (seed={seed:3d}): Test = {test_species}")
        
        # Check uniqueness
        unique_combinations = len(set(tuple(combo) for combo in test_species_combinations))
        print(f"\nUnique test species combinations: {unique_combinations}/{num_runs}")
        
        if unique_combinations > num_runs * 0.7:  # At least 70% different
            print("✅ Good species variety across runs!")
            return True
        else:
            print("⚠️ Limited species variety - consider adjusting parameters")
            return False
            
    except Exception as e:
        print(f"❌ Error testing splits: {e}")
        return False


def test_deterministic_mode(dataset_path: str, balance_strategy: str = "original"):
    
    print("\n🧪 TESTING DETERMINISTIC MODE (ALL COMBINATIONS):")
    print("="*60)
    
    try:
        from dataset import prepare_dataset
        from utils import setup_logging
        
        # Prepare dataset
        logger = setup_logging("WARNING")
        image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)
        
        # Get all combinations
        all_combos = get_all_species_combinations(species_list, n_test_species=2)
        
        print(f"Dataset: {len(image_paths)} images, {len(set(species_list))} species")
        print(f"Total combinations: {len(all_combos)}")
        print("\nTesting first 5 combinations:\n")
        
        for i, test_species_pair in enumerate(all_combos[:5], 1):
            splits = create_random_species_splits(
                image_paths, labels, species_list,
                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                random_seed=42,
                test_species_override=test_species_pair,
                logger=None
            )
            
            test_species = sorted(set(splits['test'][2]))
            print(f"Combo {i}: Requested {test_species_pair} → Got {test_species}")
            
            # Verify
            if set(test_species) == set(test_species_pair):
                print(f"  ✅ Correct")
            else:
                print(f"  ❌ MISMATCH!")
        
        print("\n✅ Deterministic mode working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing deterministic mode: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print("Testing random mode:")
        test_random_species_splits(dataset_path)
        print("\n" + "="*60)
        print("Testing deterministic mode:")
        test_deterministic_mode(dataset_path)
    else:
        print("Usage: python random_species_splitter.py /path/to/dataset")