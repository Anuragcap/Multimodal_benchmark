import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Import our modules
from config import create_config, LABEL_NAMES
from utils import (setup_logging, set_seed, compute_class_weights, 
                  plot_training_history, evaluate_model_performance, 
                  save_results, load_captions_data)
from dataset import (prepare_dataset, create_data_splits, create_data_transforms, 
                   create_dataloaders)

class MultiModalCLIP(nn.Module):
    """Multimodal Wildlife Captivity Detection using CLIP architecture"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        
        # Load CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device)
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get feature dimensions
        self._determine_feature_dimensions()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Wild=0, Captive=1
        )
        
        print(f"Initialized multimodal model with combined dimension: {self.combined_dim}")
    
    def _determine_feature_dimensions(self):
        """Determine CLIP feature dimensions"""
        with torch.no_grad():
            # Create dummy inputs
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_text = torch.randint(0, 49408, (1, 77)).to(self.device)
            
            # Get feature dimensions
            image_features = self.clip_model.encode_image(dummy_image)
            text_features = self.clip_model.encode_text(dummy_text)
            
            self.image_dim = image_features.shape[1]
            self.text_dim = text_features.shape[1]
            self.combined_dim = self.image_dim + self.text_dim
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass"""
        # Ensure inputs are on correct device
        images = batch['image'].to(self.device)
        text = batch['text'].to(self.device)
        
        with torch.no_grad():
            # Get CLIP features
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Concatenate features
        combined_features = torch.cat([image_features, text_features], dim=-1)
        
        # Classification
        outputs = self.classifier(combined_features)
        return outputs

class MultiModalTrainer:
    """Trainer for multimodal model"""
    
    def __init__(self, config, logger, model_name="multimodal"):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.model_name = model_name

        # Initialize model
        self.model = MultiModalCLIP(config.device).to(self.device)
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'val_f1': [],
            'val_auc': []
        }
    
    def setup_training(self, train_labels: List[int]):
        """Setup training components"""
        # Compute class weights
        if self.config.training.use_class_weights:
            class_weights = compute_class_weights(
                train_labels, 
                self.config.training.captive_weight_multiplier
            ).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.logger.info("Using unweighted CrossEntropyLoss")
        
        # Setup optimizer - only for classifier parameters
        trainable_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            # Move batch to device
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            running_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * running_correct / total_samples
            progress_bar.set_postfix({
                'Loss': f'{running_loss / total_samples:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * running_correct / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch is None:
                    continue
                
                labels = batch['label'].to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                
                # Get predictions and probabilities
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of captive class
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100.0 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Calculate additional metrics
        metrics, _ = evaluate_model_performance(
            all_labels, all_predictions, all_probs, LABEL_NAMES
        )
        
        return val_loss, val_acc, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Full training loop"""
        self.logger.info("Starting multimodal training...")
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            self.logger.info("-" * 60)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Val F1: {val_metrics['f1_score']:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}")
            self.logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['val_auc'].append(val_metrics['auc_roc'])
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch
                self.no_improve_count = 0
                
                # Save best model
                model_path = os.path.join(
                    self.config.data.output_dir, 
                    'models', 
                    f'best_{self.model_name}.pth' 
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.best_model_state, model_path)
                self.logger.info(f"New best model saved: {model_path} (Val Loss: {val_loss:.4f})")
            else:
                self.no_improve_count += 1
                self.logger.info(f"No improvement for {self.no_improve_count} epochs")
            
            # Early stopping
            if self.no_improve_count >= self.config.training.patience:
                self.logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Loaded best model from epoch {self.best_epoch+1}")
        
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set"""
        self.logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        all_species = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if batch is None:
                    continue
                
                labels = batch['label'].to(self.device)
                species = batch.get('species', ['unknown'] * len(labels))
                
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_species.extend(species)
        
        # Calculate comprehensive metrics
        metrics, report = evaluate_model_performance(
            all_labels, all_predictions, all_probs, LABEL_NAMES,
            save_dir=os.path.join(self.config.data.output_dir, 'plots')
        )
        
        # Per-species analysis
        species_metrics = {}
        for species in set(all_species):
            if species == 'unknown':
                continue
            
            species_indices = [i for i, s in enumerate(all_species) if s == species]
            if len(species_indices) > 0:
                species_labels = [all_labels[i] for i in species_indices]
                species_preds = [all_predictions[i] for i in species_indices]
                
                species_acc = sum(p == l for p, l in zip(species_preds, species_labels)) / len(species_labels)
                species_metrics[species] = {
                    'accuracy': species_acc,
                    'samples': len(species_indices)
                }
        
        return {
            'overall_metrics': metrics,
            'classification_report': report,
            'species_metrics': species_metrics,
            'predictions': {
                'y_true': all_labels,
                'y_pred': all_predictions,
                'y_probs': all_probs,
                'species': all_species
            }
        }

def main():
    """Main training function for baseline multimodal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Multimodal Wildlife Captivity Detection")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--captions_file", type=str, required=True, help="Path to captions JSON file")
    parser.add_argument("--output_dir", type=str, default="multimodal_baseline", help="Output directory")
    parser.add_argument("--balance_strategy", type=str, default="original", 
                       choices=["original", "1:1", "1:10"], help="Data balancing strategy")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        balance_strategy=args.balance_strategy,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Setup logging
    log_file = os.path.join(config.data.output_dir, 'logs', 'multimodal_baseline.log')
    logger = setup_logging(config.log_level, log_file)
    
    # Set random seed
    set_seed(config.seed)
    
    logger.info("="*80)
    logger.info("BASELINE MULTIMODAL WILDLIFE CAPTIVITY DETECTION")
    logger.info("="*80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Dataset path: {config.data.dataset_path}")
    logger.info(f"Captions file: {args.captions_file}")
    logger.info(f"Balance strategy: {config.data.balance_strategy}")
    
    try:
        # Load captions data
        captions_data = load_captions_data(args.captions_file, logger)
        if captions_data is None:
            raise ValueError(f"Could not load captions from {args.captions_file}")
        
        # Prepare dataset
        image_paths, labels, species_list = prepare_dataset(
            config.data.dataset_path, config.data.balance_strategy, logger
        )
        
        # Filter to only images with captions
        filtered_paths, filtered_labels, filtered_species = [], [], []
        for path, label, species in zip(image_paths, labels, species_list):
            if path in captions_data:
                filtered_paths.append(path)
                filtered_labels.append(label)
                filtered_species.append(species)
        
        logger.info(f"Filtered to {len(filtered_paths)} images with captions")
        
        # Create data splits with species-aware splitting
        splits = create_data_splits(
            filtered_paths, filtered_labels, filtered_species,
            config.data.train_ratio, config.data.val_ratio, config.data.test_ratio,
            config.seed, logger
        )
        
        # Validate splits and show detailed information
        from dataset import validate_splits_integrity
        if not validate_splits_integrity(splits, logger):
            raise ValueError("Data leakage detected in splits!")
        
        # Load CLIP tokenizer
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
        
        # Create transforms and dataloaders
        transforms_dict = create_data_transforms()
        dataloaders = create_dataloaders(
            splits, transforms_dict, config.data.batch_size, config.data.num_workers,
            captions_data, clip_tokenizer
        )
        
        # Initialize trainer
        trainer = MultiModalTrainer(config, logger, model_name='multimodal_custom')
        trainer.setup_training(splits['train'][1])
        
        # Train model
        logger.info("Starting training...")
        training_results = trainer.train(dataloaders['train'], dataloaders['val'])
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_results = trainer.evaluate(dataloaders['test'])
        
        # Plot training history
        plot_path = os.path.join(config.data.output_dir, 'plots', 'training_history.png')
        plot_training_history(training_results['history'], plot_path)
        
        # Save results
        final_results = {
            'experiment_type': 'baseline_multimodal',
            'config': {
                'dataset_path': config.data.dataset_path,
                'captions_file': args.captions_file,
                'balance_strategy': config.data.balance_strategy,
                'batch_size': config.data.batch_size,
                'num_epochs': config.training.num_epochs,
                'learning_rate': config.training.learning_rate
            },
            'training_results': training_results,
            'test_results': test_results,
            'dataset_info': {
                'total_images': len(image_paths),
                'images_with_captions': len(filtered_paths),
                'train_images': len(splits['train'][0]),
                'val_images': len(splits['val'][0]),
                'test_images': len(splits['test'][0])
            },
            'caption_info': {
                'total_captions': len(captions_data),
                'captions_file': args.captions_file
            }
        }
        
        results_path = os.path.join(config.data.output_dir, 'baseline_multimodal_results.json')
        save_results(final_results, results_path)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("BASELINE MULTIMODAL RESULTS SUMMARY")
        logger.info("="*80)
        
        test_metrics = test_results['overall_metrics']
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info(f"Test MCC: {test_metrics['mcc']:.4f}")
        
        logger.info(f"\nCaptions used: {len(captions_data)}")
        logger.info(f"Images with captions: {len(filtered_paths)}")
        logger.info(f"Results saved to: {results_path}")
        logger.info("Baseline multimodal training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()