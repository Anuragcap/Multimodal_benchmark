import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter, defaultdict
import random

def prepare_dataset(dataset_path: str, balance_strategy: str = "original", 
                   logger=None) -> Tuple[List[str], List[int], List[str]]:
    """Prepare dataset from directory structure"""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        
    logger.info("Preparing dataset...")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    image_paths = []
    labels = []
    species_list = []
    
    
    for species_name in os.listdir(dataset_path):
        species_path = os.path.join(dataset_path, species_name)
        if not os.path.isdir(species_path):
            continue
            
        for category in ['wild', 'captive']:
            category_path = os.path.join(species_path, category)
            if not os.path.exists(category_path):
                continue
                
            for img_file in os.listdir(category_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue
                    
                img_path = os.path.join(category_path, img_file)
                
                # Validate image
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    image_paths.append(img_path)
                    labels.append(0 if category == 'wild' else 1)  # Wild=0, Captive=1
                    species_list.append(species_name)
                    
                except Exception as e:
                    logger.warning(f"Skipping corrupted image {img_path}: {e}")
                    continue
    
    logger.info(f"Collected {len(image_paths)} valid images")
    
    # Apply balancing if requested
    if balance_strategy != "original":
        from utils import balance_dataset
        image_paths, labels, species_list = balance_dataset(image_paths, labels, species_list, balance_strategy, logger)
    
    return image_paths, labels, species_list

def create_data_splits(image_paths: List[str], labels: List[int], species: List[str],
                      train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                      random_seed: int = 42, logger=None) -> Dict[str, Tuple[List, List, List]]:
    """Create train/validation/test splits with SPECIES-AWARE splitting to prevent data leakage"""
    if logger:
        logger.info("🛠️ Creating species-stratified splits to prevent data leakage...")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Create species groups with their data
    species_groups = defaultdict(list)
    for i, (path, label, spec) in enumerate(zip(image_paths, labels, species)):
        species_groups[spec].append({
            'index': i,
            'path': path,
            'label': label,
            'species': spec
        })
    
    # Calculate target sizes
    total_images = len(image_paths)
    target_train = int(total_images * train_ratio)
    target_val = int(total_images * val_ratio)
    target_test = total_images - target_train - target_val
    
    if logger:
        logger.info(f"Target split sizes: Train={target_train}, Val={target_val}, Test={target_test}")
    
    # Sort species by size and assign to splits
    species_by_size = sorted(species_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Initialize splits
    train_data, val_data, test_data = [], [], []
    train_size, val_size, test_size = 0, 0, 0
    
    # Greedy assignment of species to splits
    for species_name, species_data in species_by_size:
        species_size = len(species_data)
        
        # Calculate current split sizes
        current_sizes = [train_size, val_size, test_size]
        target_sizes = [target_train, target_val, target_test]
        
        # Calculate how far each split is from its target (as proportion)
        split_needs = []
        for i, (current, target) in enumerate(zip(current_sizes, target_sizes)):
            if target > 0:
                need = (target - current) / target
            else:
                need = 0
            split_needs.append(need)
        
        # Assign to the split that needs data most
        best_split = np.argmax(split_needs)
        
        # But ensure we don't exceed targets too much (within 20%)
        if current_sizes[best_split] + species_size > target_sizes[best_split] * 1.2:
            # Try next best option
            sorted_indices = np.argsort(split_needs)[::-1]
            for idx in sorted_indices:
                if current_sizes[idx] + species_size <= target_sizes[idx] * 1.2:
                    best_split = idx
                    break
        
        # Assign species to chosen split
        if best_split == 0:  # Train
            train_data.extend(species_data)
            train_size += species_size
        elif best_split == 1:  # Val
            val_data.extend(species_data)
            val_size += species_size
        else:  # Test
            test_data.extend(species_data)
            test_size += species_size
        
        if logger:
            split_name = ['Train', 'Val', 'Test'][best_split]
            # Count wild/captive for this species
            species_wild = sum(1 for item in species_data if item['label'] == 0)
            species_captive = sum(1 for item in species_data if item['label'] == 1)
            logger.info(f"  {species_name}: {species_size} samples ({species_wild}W/{species_captive}C) → {split_name}")
    
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
    
    train_paths, train_labels, train_species = extract_split_data(train_data)
    val_paths, val_labels, val_species = extract_split_data(val_data)
    test_paths, test_labels, test_species = extract_split_data(test_data)
    
    # Verify no species overlap
    train_species_set = set(train_species)
    val_species_set = set(val_species)
    test_species_set = set(test_species)
    
    overlap_train_val = train_species_set & val_species_set
    overlap_train_test = train_species_set & test_species_set
    overlap_val_test = val_species_set & test_species_set
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        error_msg = "CRITICAL: Species overlap detected in splits!"
        if overlap_train_val:
            error_msg += f" Train-Val: {overlap_train_val}"
        if overlap_train_test:
            error_msg += f" Train-Test: {overlap_train_test}"
        if overlap_val_test:
            error_msg += f" Val-Test: {overlap_val_test}"
        raise ValueError(error_msg)
    
    if logger:
        logger.info("✅ Species-stratified splits created successfully!")
        logger.info(f"Final sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        logger.info(f"Species count: Train={len(train_species_set)}, Val={len(val_species_set)}, Test={len(test_species_set)}")
        
        # Log detailed species assignments
        logger.info("\n📊 DETAILED SPECIES ASSIGNMENTS:")
        logger.info("-" * 60)
        
        # Group species by split for summary
        species_by_split = {'Train': [], 'Val': [], 'Test': []}
        for species_name, species_data in species_by_size:
            # Find which split this species went to
            if species_name in train_species_set:
                species_by_split['Train'].append(species_name)
            elif species_name in val_species_set:
                species_by_split['Val'].append(species_name)
            elif species_name in test_species_set:
                species_by_split['Test'].append(species_name)
        
        for split_name, species_in_split in species_by_split.items():
            if species_in_split:
                logger.info(f"{split_name} Split ({len(species_in_split)} species):")
                for species in species_in_split:
                    # Count samples for this species in this split
                    if split_name == 'Train':
                        species_indices = [i for i, s in enumerate(train_species) if s == species]
                        species_labels = [train_labels[i] for i in species_indices]
                    elif split_name == 'Val':
                        species_indices = [i for i, s in enumerate(val_species) if s == species]
                        species_labels = [val_labels[i] for i in species_indices]
                    else:  # Test
                        species_indices = [i for i, s in enumerate(test_species) if s == species]
                        species_labels = [test_labels[i] for i in species_indices]
                    
                    wild_count = species_labels.count(0)
                    captive_count = species_labels.count(1)
                    total = len(species_labels)
                    logger.info(f"  • {species}: {total} samples ({wild_count}W/{captive_count}C)")
        
        logger.info("-" * 60)
        
        # Log label distributions
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        test_dist = Counter(test_labels)
        logger.info(f"Label distribution:")
        logger.info(f"  Train: Wild={train_dist[0]}, Captive={train_dist[1]} (ratio={train_dist[1]/train_dist[0]:.2f})")
        logger.info(f"  Val: Wild={val_dist[0]}, Captive={val_dist[1]} (ratio={val_dist[1]/val_dist[0]:.2f})")
        logger.info(f"  Test: Wild={test_dist[0]}, Captive={test_dist[1]} (ratio={test_dist[1]/test_dist[0]:.2f})")
    
    return {
        'train': (train_paths, train_labels, train_species),
        'val': (val_paths, val_labels, val_species),
        'test': (test_paths, test_labels, test_species)
    }

def validate_splits_integrity(splits: Dict[str, Tuple[List, List, List]], logger=None) -> bool:
    
    if logger:
        logger.info("🔍 Validating split integrity...")
    
    all_paths = []
    all_species_by_split = {}
    
    for split_name, (paths, labels, species) in splits.items():
        all_paths.extend(paths)
        all_species_by_split[split_name] = set(species)
    
    # Check for duplicate paths
    if len(all_paths) != len(set(all_paths)):
        if logger:
            logger.error("❌ Duplicate image paths found across splits!")
        return False
    
    # Check for species overlap
    train_species = all_species_by_split['train']
    val_species = all_species_by_split['val']
    test_species = all_species_by_split['test']
    
    overlaps = []
    if train_species & val_species:
        overlaps.append(f"Train-Val: {train_species & val_species}")
    if train_species & test_species:
        overlaps.append(f"Train-Test: {train_species & test_species}")
    if val_species & test_species:
        overlaps.append(f"Val-Test: {val_species & test_species}")
    
    if overlaps:
        if logger:
            logger.error(f"❌ Species overlap detected: {'; '.join(overlaps)}")
        return False
    
    if logger:
        logger.info("✅ All integrity checks passed - no data leakage detected!")
        
        # Show detailed split summary
        logger.info("\n📋 SPLIT VALIDATION SUMMARY:")
        logger.info("=" * 70)
        
        total_species = len(train_species | val_species | test_species)
        total_images = sum(len(paths) for paths, _, _ in splits.values())
        
        logger.info(f"Total species: {total_species}")
        logger.info(f"Total images: {total_images}")
        
        for split_name, (paths, labels, species_list) in splits.items():
            unique_species = set(species_list)
            wild_count = labels.count(0)
            captive_count = labels.count(1)
            
            logger.info(f"\n{split_name.upper()} SPLIT:")
            logger.info(f"  Images: {len(paths)}")
            logger.info(f"  Species: {len(unique_species)}")
            logger.info(f"  Wild: {wild_count}, Captive: {captive_count}")
            logger.info(f"  Species list: {sorted(unique_species)}")
        
        logger.info("=" * 70)
    
    return True

def create_data_transforms() -> Dict[str, transforms.Compose]:
    
    # CLIP normalization for consistency
    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    transforms_dict = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            clip_normalize
        ]),
        
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            clip_normalize
        ]),
        
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            clip_normalize
        ])
    }
    
    return transforms_dict

class WildlifeDataset(Dataset):
    """Base dataset class for wildlife images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 species: List[str], transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.species = species
        self.transform = transform
        
        assert len(image_paths) == len(labels) == len(species), \
            "All input lists must have the same length"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'label': torch.tensor(self.labels[idx], dtype=torch.long),
                'species': self.species[idx],
                'image_path': image_path
            }
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return dummy sample
            dummy_image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            return {
                'image': dummy_image,
                'label': torch.tensor(0, dtype=torch.long),
                'species': 'unknown',
                'image_path': self.image_paths[idx]
            }

class MultiModalDataset(WildlifeDataset):
    """Dataset for multimodal training with captions"""
    
    def __init__(self, image_paths: List[str], labels: List[int], species: List[str],
                 captions_data: Dict[str, Dict], clip_tokenizer,
                 transform: Optional[transforms.Compose] = None):
        super().__init__(image_paths, labels, species, transform)
        self.captions_data = captions_data
        self.clip_tokenizer = clip_tokenizer
    
    def format_caption(self, captions: Dict[str, str]) -> str:
        """Format caption aspects into a single caption"""
        parts = []
        
        if 'animal_behavior' in captions:
            parts.append(f"the animal is {captions['animal_behavior']}")
        
        if 'surroundings' in captions:
            parts.append(f"surrounded by {captions['surroundings']}")
        
        if 'background' in captions:
            parts.append(f"with {captions['background']} in the background")
        
        if 'lighting' in captions:
            parts.append(f"under {captions['lighting']}")
        
        if 'vegetation' in captions:
            parts.append(f"near {captions['vegetation']}")
        
        caption = " and ".join(parts) if parts else "an animal in an environment"
        
        # Add contextual prefix
        caption = "An animal where " + caption
            
        return caption
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        
        # Get caption data
        image_path = self.image_paths[idx]
        if image_path in self.captions_data:
            caption_info = self.captions_data[image_path]
            formatted_caption = self.format_caption(caption_info.get('captions', {}))
        else:
            # Default caption if not available
            formatted_caption = f"A wild animal" if self.labels[idx] == 0 else "A captive animal"
        
        # Tokenize caption
        try:
            text_tokens = self.clip_tokenizer([formatted_caption]).squeeze(0)
        except Exception as e:
            print(f"Error tokenizing caption for {image_path}: {str(e)}")
            # Use empty token sequence as fallback
            text_tokens = torch.zeros(77, dtype=torch.long)  # CLIP's max length
        
        item['text'] = text_tokens
        item['caption'] = formatted_caption
        
        return item

def collate_fn(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Custom collate function for multimodal data"""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # Check if batch contains text tokens (multimodal)
    has_text = 'text' in batch[0]
    
    # Prepare batch dict
    batch_dict = {
        'image': torch.stack([item['image'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }
    
    if has_text:
        # Handle variable-length text tokens
        text_tokens = [item['text'] for item in batch]
        max_len = max(t.size(0) for t in text_tokens)
        
        # Pad sequences
        padded_text = torch.zeros(len(text_tokens), max_len, dtype=text_tokens[0].dtype)
        for i, tokens in enumerate(text_tokens):
            padded_text[i, :tokens.size(0)] = tokens
        
        batch_dict['text'] = padded_text
        batch_dict['caption'] = [item['caption'] for item in batch]
    
    # Add non-tensor items
    batch_dict['species'] = [item['species'] for item in batch]
    batch_dict['image_path'] = [item['image_path'] for item in batch]
    
    return batch_dict

def create_dataloaders(splits: Dict[str, Tuple[List, List, List]], 
                      transforms_dict: Dict[str, transforms.Compose],
                      batch_size: int = 32,
                      num_workers: int = 4,
                      captions_data: Optional[Dict] = None,
                      clip_tokenizer = None) -> Dict[str, DataLoader]:
    """Create data loaders for train/val/test splits"""
    dataloaders = {}
    
    for split_name, (paths, labels, species_list) in splits.items():
        transform = transforms_dict[split_name]
        
        # Choose dataset class based on whether we have captions
        if captions_data is not None and clip_tokenizer is not None:
            # Multimodal dataset
            dataset = MultiModalDataset(
                image_paths=paths,
                labels=labels,
                species=species_list,
                captions_data=captions_data,
                clip_tokenizer=clip_tokenizer,
                transform=transform
            )
            collate_func = collate_fn
        else:
            # Single modal dataset
            dataset = WildlifeDataset(
                image_paths=paths,
                labels=labels,
                species=species_list,
                transform=transform
            )
            collate_func = None
        
        # Create dataloader
        shuffle = (split_name == 'train')
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_func,
            pin_memory=torch.cuda.is_available()
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders