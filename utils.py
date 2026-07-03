import os
import random
import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
import json

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    
    logger = logging.getLogger("wildlife_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
    
    return logger

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate_image(image_path: str) -> bool:
    """Validate if image file exists and is readable"""
    if not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to check it loads properly
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False

def balance_dataset(image_paths: List[str], labels: List[int], species_list: List[str],
                   strategy: str, logger: logging.Logger) -> Tuple[List[str], List[int], List[str]]:
    """Balance dataset according to strategy"""
    if strategy == "original":
        logger.info("Using original dataset distribution")
        return image_paths, labels, species_list
    
    # Separate by class (Wild=0, Captive=1)
    wild_indices = [i for i, label in enumerate(labels) if label == 0]
    captive_indices = [i for i, label in enumerate(labels) if label == 1]
    
    logger.info(f"Original distribution - Wild: {len(wild_indices)}, Captive: {len(captive_indices)}")
    
    if strategy == "1:1":
        # Balance to have equal numbers
        min_count = min(len(wild_indices), len(captive_indices))
        if len(wild_indices) > min_count:
            wild_indices = random.sample(wild_indices, min_count)
        if len(captive_indices) > min_count:
            captive_indices = random.sample(captive_indices, min_count)
        
    elif strategy == "1:10":
        # 1 captive : 10 wild ratio
        target_wild = len(captive_indices) * 10
        if len(wild_indices) > target_wild:
            wild_indices = random.sample(wild_indices, target_wild)
    
    # Combine indices
    selected_indices = wild_indices + captive_indices
    random.shuffle(selected_indices)
    
    # Apply balancing to all three lists
    balanced_paths = [image_paths[i] for i in selected_indices]
    balanced_labels = [labels[i] for i in selected_indices]
    balanced_species = [species_list[i] for i in selected_indices]
    
    new_distribution = Counter(balanced_labels)
    logger.info(f"Balanced distribution - Wild: {new_distribution[0]}, Captive: {new_distribution[1]}")
    
    return balanced_paths, balanced_labels, balanced_species

def compute_class_weights(labels: List[int], multiplier: float = 1.5) -> torch.Tensor:
    """Compute class weights for imbalanced dataset"""
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    weights = torch.zeros(2)
    for label in [0, 1]:  # Wild=0, Captive=1
        count = label_counts.get(label, 1)  # Avoid division by zero
        weights[label] = total_samples / (2 * count)
    
    # Apply multiplier to captive class (typically minority)
    if label_counts.get(1, 0) < label_counts.get(0, 0):
        weights[1] *= multiplier
    
    return weights

def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot training history"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[0, 0].plot(history['train_acc'], label='Train', marker='o')
            axes[0, 0].plot(history['val_acc'], label='Validation', marker='s')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Training and validation loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 1].plot(history['train_loss'], label='Train', marker='o')
            axes[0, 1].plot(history['val_loss'], label='Validation', marker='s')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in history:
            axes[1, 0].plot(history['learning_rates'], marker='o')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Validation metrics
        if 'val_f1' in history:
            axes[1, 1].plot(history['val_f1'], label='F1-Score', marker='o')
            if 'val_auc' in history:
                axes[1, 1].plot(history['val_auc'], label='AUC-ROC', marker='s')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training history: {e}")


def evaluate_model_performance(y_true: List[int], y_pred: List[int], 
                             y_probs: List[float], class_names: List[str],
                             save_dir: Optional[str] = None) -> Tuple[Dict[str, float], Dict]:
    
    # Overall accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if len(y_true) > 0 else 0.0
    
    # Macro metrics (average across both classes - required for imbalanced data)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics (both wild=0 and captive=1)
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1]
    )
    
    # MCC and AUC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
    
    try:
        auc_roc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_roc = 0.0
    
    metrics = {
        # Overall metrics
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'mcc': float(mcc),
        'auc_roc': float(auc_roc),
        
        # Wild class (index 0)
        'wild_precision': float(per_class_precision[0]),
        'wild_recall': float(per_class_recall[0]),
        'wild_f1': float(per_class_f1[0]),
        'wild_support': int(support[0]),
        
        # Captive class (index 1)
        'captive_precision': float(per_class_precision[1]),
        'captive_recall': float(per_class_recall[1]),
        'captive_f1': float(per_class_f1[1]),
        'captive_support': int(support[1]),
        
        # Legacy names for backward compatibility
        'precision': float(per_class_precision[1]),
        'recall': float(per_class_recall[1]),
        'f1_score': float(per_class_f1[1]),
    }
    
    # Generate classification report
    try:
        report = classification_report(
            y_true, y_pred, target_names=class_names, 
            output_dict=True, zero_division=0
        )
    except:
        report = {'accuracy': accuracy}
    
    # Plot confusion matrix
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not save confusion matrix: {e}")
    
    return metrics, report


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file with proper numpy array handling"""
    
    def convert_to_serializable(obj):
        """Convert numpy arrays and tensors to JSON-compatible format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        elif torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        return obj
    
    try:
        # Convert to serializable format
        serializable_results = convert_to_serializable(results)
        
        # Create directory and save
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Verify predictions were saved
        if 'test_results' in serializable_results:
            if 'predictions' in serializable_results['test_results']:
                preds = serializable_results['test_results']['predictions']
                if 'y_true' in preds:
                    print(f"✓ Results saved: {filepath}")
                    print(f"  ✓ Predictions: {len(preds['y_true'])} samples")
                    return
        
        print(f"✓ Results saved: {filepath}")
        
    except Exception as e:
        print(f"✗ Error saving results to {filepath}: {e}")
        import traceback
        traceback.print_exc()


def load_captions_data(captions_file: str, logger: logging.Logger) -> Optional[Dict]:
    """Load captions data from JSON file"""
    if not os.path.exists(captions_file):
        logger.warning(f"Captions file not found: {captions_file}")
        return None
    
    try:
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        logger.info(f"Loaded {len(captions_data)} captions from {captions_file}")
        return captions_data
    except Exception as e:
        logger.error(f"Error loading captions file: {str(e)}")
        return None


def plot_adversarial_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot training history for adversarial model"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Training and validation accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[0, 0].plot(history['train_acc'], label='Train', marker='o')
            axes[0, 0].plot(history['val_acc'], label='Validation', marker='s')
            axes[0, 0].set_title('Captivity Detection Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Training losses breakdown
        if 'train_captivity_loss' in history and 'train_species_loss' in history:
            axes[0, 1].plot(history['train_captivity_loss'], label='Captivity Loss', marker='o')
            axes[0, 1].plot(history['train_species_loss'], label='Species Loss', marker='s')
            axes[0, 1].plot(history['train_loss'], label='Total Loss', marker='^')
            axes[0, 1].set_title('Training Loss Components')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Lambda progression
        if 'lambda_values' in history:
            axes[0, 2].plot(history['lambda_values'], marker='o', color='red')
            axes[0, 2].set_title('Adversarial Lambda Progression')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Lambda')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Validation loss
        if 'val_loss' in history:
            axes[1, 0].plot(history['val_loss'], marker='o', color='orange')
            axes[1, 0].set_title('Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in history:
            axes[1, 1].plot(history['learning_rates'], marker='o')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Validation metrics
        if 'val_f1' in history:
            axes[1, 2].plot(history['val_f1'], label='F1-Score', marker='o')
            if 'val_auc' in history:
                axes[1, 2].plot(history['val_auc'], label='AUC-ROC', marker='s')
            axes[1, 2].set_title('Validation Metrics')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting adversarial training history: {e}")