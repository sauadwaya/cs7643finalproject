"""
Experiment Configuration Management
Centralized configuration for all model variants and training strategies
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class ModelConfig:
    """Base model configuration"""
    encoder_id: str = "google/vit-base-patch16-224-in21k"
    decoder_id: str = "gpt2" 
    tokenizer_name: str = "gpt2"
    max_length: int = 48
    image_size: Tuple[int, int] = (224, 224)
    
    # LoRA configuration
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Default LoRA targets for ViT
            self.lora_target_modules = ["query", "value", "dense"]

@dataclass 
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    
    # Training strategies
    use_label_smoothing: bool = False
    label_smoothing_factor: float = 0.1
    use_scheduled_sampling: bool = False
    scheduled_sampling_prob: float = 0.5
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    clip_grad_norm: float = 1.0

@dataclass
class DataConfig:
    """Data configuration"""
    datasets: List[str] = None  # ["flickr8k", "flickr30k", "coco"]
    train_split: str = "train"
    val_split: str = "dev" 
    test_split: str = "test"
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_prob: float = 0.5
    corruption_types: List[str] = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["flickr8k"]
        if self.corruption_types is None:
            self.corruption_types = ["gaussian_noise", "blur", "brightness"]

@dataclass
class GenerationConfig:
    """Text generation configuration"""
    max_new_tokens: int = 20
    min_length: int = 5
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 3.0
    length_penalty: float = 1.0
    num_beams: int = 1  # 1 for sampling, >1 for beam search
    early_stopping: bool = True

@dataclass
class VQAConfig:
    """VQA-specific configuration"""
    question_max_length: int = 32
    answer_vocab_size: int = 3000
    use_question_encoder: bool = True
    question_encoder_id: str = "bert-base-uncased"
    
    # Multi-task learning
    captioning_weight: float = 1.0
    vqa_weight: float = 1.0

# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "baseline": {
        "model": ModelConfig(),
        "training": TrainingConfig(learning_rate=1e-4, epochs=5),
        "data": DataConfig(),
        "generation": GenerationConfig()
    },
    
    "lora_encoder": {
        "model": ModelConfig(
            use_lora=True,
            lora_rank=8,
            lora_target_modules=["query", "value"]
        ),
        "training": TrainingConfig(learning_rate=1e-3, epochs=8),
        "data": DataConfig(), 
        "generation": GenerationConfig()
    },
    
    "multi_dataset": {
        "model": ModelConfig(),
        "training": TrainingConfig(
            learning_rate=5e-5,
            epochs=10,
            use_label_smoothing=True
        ),
        "data": DataConfig(datasets=["flickr8k", "flickr30k"]),
        "generation": GenerationConfig()
    },
    
    "robust_training": {
        "model": ModelConfig(),
        "training": TrainingConfig(
            learning_rate=1e-4,
            epochs=12,
            use_scheduled_sampling=True
        ),
        "data": DataConfig(
            use_augmentation=True,
            corruption_types=["gaussian_noise", "blur", "brightness", "contrast"]
        ),
        "generation": GenerationConfig()
    },
    
    "vqa_joint": {
        "model": ModelConfig(),
        "training": TrainingConfig(learning_rate=2e-5, epochs=15),
        "data": DataConfig(datasets=["flickr8k", "vqav2"]),
        "generation": GenerationConfig(),
        "vqa": VQAConfig()
    }
}

def get_config(experiment_name: str) -> Dict:
    """Get configuration for a specific experiment"""
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[experiment_name]

def list_experiments() -> List[str]:
    """List all available experiment configurations"""
    return list(EXPERIMENT_CONFIGS.keys())