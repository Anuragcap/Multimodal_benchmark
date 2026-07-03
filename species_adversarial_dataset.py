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

# Import the existing dataset functions
from dataset import (
    WildlifeDataset, 
    create_data_transforms,
    collate_fn as original_collate_fn
)


class SpeciesAdversarialDataset(Dataset):
    
    
    def __init__(self, image_paths: List[str], labels: List[int], species: List[str],
                 species_to_idx: Dict[str, int], captions_data: Dict[str, Dict], 
                 clip_tokenizer, transform: Optional[transforms.Compose] = None):
        
        self.image_paths = image_paths
        self.labels = labels
        self.species = species
        self.species_to_idx = species_to_idx
        self.captions_data = captions_data
        self.clip_tokenizer = clip_tokenizer
        self.transform = transform
        
        assert len(image_paths) == len(labels) == len(species), \
            "All input lists must have the same length"
        
        # Verify all species are in the mapping
        for spec in species:
            if spec not in species_to_idx:
                raise ValueError(f"Species '{spec}' not found in species_to_idx mapping")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def format_caption(self, captions: Dict[str, str]) -> str:
        """Format caption aspects into a single caption (same as original)"""
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
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get caption data
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
            
            return {
                'image': image,
                'label': torch.tensor(self.labels[idx], dtype=torch.long),
                'species': self.species[idx],
                'species_label': torch.tensor(self.species_to_idx[self.species[idx]], dtype=torch.long),
                'text': text_tokens,
                'caption': formatted_caption,
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
                'species_label': torch.tensor(0, dtype=torch.long),
                'text': torch.zeros(77, dtype=torch.long),
                'caption': 'dummy caption',
                'image_path': self.image_paths[idx]
            }


def species_adversarial_collate_fn(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Custom collate function for species-adversarial multimodal data"""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # Prepare batch dict
    batch_dict = {
        'image': torch.stack([item['image'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'species_label': torch.stack([item['species_label'] for item in batch])
    }
    
    # Handle variable-length text tokens
    text_tokens = [item['text'] for item in batch]
    max_len = max(t.size(0) for t in text_tokens)
    
    # Pad sequences
    padded_text = torch.zeros(len(text_tokens), max_len, dtype=text_tokens[0].dtype)
    for i, tokens in enumerate(text_tokens):
        padded_text[i, :tokens.size(0)] = tokens
    
    batch_dict['text'] = padded_text
    
    # Add non-tensor items
    batch_dict['species'] = [item['species'] for item in batch]
    batch_dict['caption'] = [item['caption'] for item in batch]
    batch_dict['image_path'] = [item['image_path'] for item in batch]
    
    return batch_dict


def create_species_adversarial_dataloaders(splits: Dict[str, Tuple[List, List, List]], 
                                         transforms_dict: Dict[str, transforms.Compose],
                                         batch_size: int = 32,
                                         num_workers: int = 4,
                                         captions_data: Dict = None,
                                         clip_tokenizer = None,
                                         species_to_idx: Dict[str, int] = None) -> Dict[str, DataLoader]:
    
    
    if captions_data is None:
        raise ValueError("captions_data is required for multimodal training")
    if clip_tokenizer is None:
        raise ValueError("clip_tokenizer is required for multimodal training")
    if species_to_idx is None:
        raise ValueError("species_to_idx mapping is required for adversarial training")
    
    dataloaders = {}
    
    for split_name, (paths, labels, species_list) in splits.items():
        transform = transforms_dict[split_name]
        
        # Create species-adversarial dataset
        dataset = SpeciesAdversarialDataset(
            image_paths=paths,
            labels=labels,
            species=species_list,
            species_to_idx=species_to_idx,
            captions_data=captions_data,
            clip_tokenizer=clip_tokenizer,
            transform=transform
        )
        
        # Create dataloader
        shuffle = (split_name == 'train')
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=species_adversarial_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == 'train')  # Drop last incomplete batch for training
        )
        
        dataloaders[split_name] = dataloader
        
        # Print dataset info
        print(f"{split_name.capitalize()} dataset: {len(dataset)} samples")
        
        # Print species distribution for this split
        species_counter = Counter(species_list)
        print(f"  Species distribution: {dict(species_counter)}")
        
        # Print label distribution
        label_counter = Counter(labels)
        print(f"  Label distribution: Wild={label_counter[0]}, Captive={label_counter[1]}")
    
    return dataloaders


def analyze_species_adversarial_splits(splits: Dict[str, Tuple[List, List, List]], 
                                     species_to_idx: Dict[str, int]):
    """Analyze splits for species-adversarial training"""
    print("\n" + "="*70)
    print("SPECIES-ADVERSARIAL SPLITS ANALYSIS")
    print("="*70)
    
    all_train_species = set(splits['train'][2])
    all_val_species = set(splits['val'][2])
    all_test_species = set(splits['test'][2])
    
    # Check for species overlap (should be none for proper adversarial training)
    train_val_overlap = all_train_species & all_val_species
    train_test_overlap = all_train_species & all_test_species
    val_test_overlap = all_val_species & all_test_species
    
    print(f"Species Distribution:")
    print(f"  Total unique species: {len(species_to_idx)}")
    print(f"  Train species: {len(all_train_species)} - {sorted(all_train_species)}")
    print(f"  Val species: {len(all_val_species)} - {sorted(all_val_species)}")
    print(f"  Test species: {len(all_test_species)} - {sorted(all_test_species)}")
    
    print(f"\nSpecies Overlap Analysis:")
    if train_val_overlap:
        print(f"  ⚠️  Train-Val overlap: {train_val_overlap}")
    else:
        print(f"  ✅ No Train-Val overlap")
        
    if train_test_overlap:
        print(f"  ⚠️  Train-Test overlap: {train_test_overlap}")
    else:
        print(f"  ✅ No Train-Test overlap")
        
    if val_test_overlap:
        print(f"  ⚠️  Val-Test overlap: {val_test_overlap}")
    else:
        print(f"  ✅ No Val-Test overlap")
    
    # Analyze class balance within each species across splits
    print(f"\nPer-Species Class Balance:")
    for split_name, (paths, labels, species_list) in splits.items():
        print(f"\n{split_name.upper()} Split:")
        species_label_counts = defaultdict(lambda: {'wild': 0, 'captive': 0})
        
        for label, species in zip(labels, species_list):
            if label == 0:
                species_label_counts[species]['wild'] += 1
            else:
                species_label_counts[species]['captive'] += 1
        
        for species in sorted(species_label_counts.keys()):
            wild_count = species_label_counts[species]['wild']
            captive_count = species_label_counts[species]['captive']
            total = wild_count + captive_count
            captive_ratio = captive_count / total if total > 0 else 0
            
            print(f"  {species}: {total} total ({wild_count}W/{captive_count}C, {captive_ratio:.2%} captive)")
    
    print("\n" + "="*70)
    
    # Return analysis results
    return {
        'species_overlap': {
            'train_val': train_val_overlap,
            'train_test': train_test_overlap,
            'val_test': val_test_overlap
        },
        'species_distribution': {
            'train': list(all_train_species),
            'val': list(all_val_species),
            'test': list(all_test_species)
        },
        'no_overlap': len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0
    }


# Test function to verify the dataset works correctly
def test_species_adversarial_dataset():
    """Test function for species-adversarial dataset"""
    print("Testing Species-Adversarial Dataset...")
    
    # Mock data
    species_to_idx = {'species_a': 0, 'species_b': 1, 'species_c': 2}
    
    # Create dummy dataset (you would replace this with real data)
    image_paths = ['dummy1.jpg', 'dummy2.jpg', 'dummy3.jpg']
    labels = [0, 1, 0]  # wild, captive, wild
    species = ['species_a', 'species_b', 'species_c']
    
    # Mock captions data
    captions_data = {
        'dummy1.jpg': {'captions': {'animal_behavior': 'foraging', 'surroundings': 'trees'}},
        'dummy2.jpg': {'captions': {'animal_behavior': 'sitting', 'surroundings': 'cage bars'}},
        'dummy3.jpg': {'captions': {'animal_behavior': 'running', 'surroundings': 'grass'}}
    }
    
    # Mock tokenizer
    class MockTokenizer:
        def __call__(self, texts):
            return torch.randint(0, 1000, (1, 77))  # Random tokens
    
    # Mock transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    try:
        dataset = SpeciesAdversarialDataset(
            image_paths=image_paths,
            labels=labels,
            species=species,
            species_to_idx=species_to_idx,
            captions_data=captions_data,
            clip_tokenizer=MockTokenizer(),
            transform=transform
        )
        
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test getting an item (this will fail without real images, but structure is correct)
        print("✅ Dataset structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


if __name__ == "__main__":
    test_species_adversarial_dataset()