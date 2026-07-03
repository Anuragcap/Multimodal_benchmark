import os
import torch
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)

# Label mapping - Wild=0, Captive=1
LABEL_MAPPING = {
    'wild': 0,
    'captive': 1
}

LABEL_NAMES = ['Wild', 'Captive']

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str
    output_dir: str = "outputs"
    batch_size: int = 32
    num_workers: int = 4
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Data balancing
    balance_strategy: str = "original"  # "original", "1:1", "1:10"
    
    def __post_init__(self):
        self._validate_and_create_dirs()
    
    def _validate_and_create_dirs(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) >= 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['models', 'logs', 'plots', 'results']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

@dataclass
class ModelConfig:
    """Model configuration"""
    clip_model: str = "ViT-B-16"
    dropout: float = 0.2
    hidden_dim: int = 512
    freeze_clip: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 10
    use_class_weights: bool = True
    captive_weight_multiplier: float = 1.5

@dataclass
class Config:
    """Main configuration"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    device: str = "auto"
    seed: int = 42
    log_level: str = "INFO"
    
    def __post_init__(self):
        self._setup_device()
    
    def _setup_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Auto-selected device: cuda")
            else:
                self.device = "cpu"
                logger.info("Auto-selected device: cpu")

def create_config(dataset_path: str, 
                 output_dir: str = "outputs",
                 balance_strategy: str = "original",
                 batch_size: int = 32,
                 num_epochs: int = 15,
                 learning_rate: float = 1e-4,
                 device: str = "auto") -> Config:
    """Create configuration"""
    return Config(
        data=DataConfig(
            dataset_path=dataset_path,
            output_dir=output_dir,
            batch_size=batch_size,
            balance_strategy=balance_strategy
        ),
        model=ModelConfig(),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=learning_rate
        ),
        device=device
    )