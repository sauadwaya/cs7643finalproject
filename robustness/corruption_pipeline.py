"""
Image Corruption and Augmentation Pipeline
Main pipeline for applying corruptions and augmentations to images.
"""

from PIL import Image
from typing import List, Dict, Optional, Union, Callable
import random
from .image_corruptions import CORRUPTION_FUNCTIONS, apply_corruption
from .image_augmentations import AUGMENTATION_FUNCTIONS, apply_augmentation


class CorruptionPipeline:
    """
    Pipeline for applying image corruptions for robustness testing.
    """
    
    def __init__(self, corruption_types: Optional[List[str]] = None, 
                 severity: Union[int, List[int]] = 1,
                 random_severity: bool = False,
                 apply_probability: float = 1.0):
        """
        Initialize corruption pipeline.
        
        Args:
            corruption_types: List of corruption types to apply. If None, uses all.
            severity: Severity level(s) 1-5. Can be int or list for multiple levels.
            random_severity: If True, randomly selects severity from range
            apply_probability: Probability of applying corruption (0-1)
        """
        if corruption_types is None:
            self.corruption_types = list(CORRUPTION_FUNCTIONS.keys())
        else:
            invalid = [c for c in corruption_types if c not in CORRUPTION_FUNCTIONS]
            if invalid:
                raise ValueError(f"Invalid corruption types: {invalid}")
            self.corruption_types = corruption_types
        
        if isinstance(severity, int):
            self.severity_levels = [severity]
        else:
            self.severity_levels = severity
        
        self.random_severity = random_severity
        self.apply_probability = apply_probability
    
    def __call__(self, image: Image.Image, corruption_type: Optional[str] = None) -> Image.Image:
        """
        Apply corruption to image.
        
        Args:
            image: PIL Image
            corruption_type: Specific corruption to apply. If None, randomly selects.
        
        Returns:
            Corrupted PIL Image
        """
        if random.random() > self.apply_probability:
            return image
        
        if corruption_type is None:
            corruption_type = random.choice(self.corruption_types)
        elif corruption_type not in CORRUPTION_FUNCTIONS:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        
        if self.random_severity:
            severity = random.choice(self.severity_levels)
        else:
            severity = self.severity_levels[0]
        
        return apply_corruption(image, corruption_type, severity)
    
    def apply_all(self, image: Image.Image) -> Dict[str, Image.Image]:
        """
        Apply all corruption types to image and return dictionary.
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary mapping corruption type to corrupted image
        """
        results = {}
        for corruption_type in self.corruption_types:
            for severity in self.severity_levels:
                key = f"{corruption_type}_severity_{severity}"
                results[key] = apply_corruption(image, corruption_type, severity)
        return results


class AugmentationPipeline:
    """
    Pipeline for applying image augmentations for training.
    """
    
    def __init__(self, augmentation_types: Optional[List[str]] = None,
                 augmentation_configs: Optional[Dict[str, Dict]] = None,
                 apply_probability: float = 1.0,
                 sequential: bool = True):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentation_types: List of augmentation types to apply. If None, uses all.
            augmentation_configs: Dictionary mapping augmentation type to its config
            apply_probability: Probability of applying augmentation (0-1)
            sequential: If True, applies augmentations sequentially. If False, applies one randomly.
        """
        if augmentation_types is None:
            self.augmentation_types = list(AUGMENTATION_FUNCTIONS.keys())
        else:
            invalid = [a for a in augmentation_types if a not in AUGMENTATION_FUNCTIONS]
            if invalid:
                raise ValueError(f"Invalid augmentation types: {invalid}")
            self.augmentation_types = augmentation_types
        
        self.augmentation_configs = augmentation_configs or {}
        self.apply_probability = apply_probability
        self.sequential = sequential
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentations to image.
        
        Args:
            image: PIL Image
        
        Returns:
            Augmented PIL Image
        """
        if random.random() > self.apply_probability:
            return image
        
        if self.sequential:
            for aug_type in self.augmentation_types:
                if random.random() < self.apply_probability:
                    config = self.augmentation_configs.get(aug_type, {})
                    image = apply_augmentation(image, aug_type, **config)
        else:
            aug_type = random.choice(self.augmentation_types)
            config = self.augmentation_configs.get(aug_type, {})
            image = apply_augmentation(image, aug_type, **config)
        
        return image


class CombinedPipeline:
    """
    Combined pipeline for both augmentations and corruptions.
    Useful for training with augmentations and testing with corruptions.
    """
    
    def __init__(self, augmentation_pipeline: Optional[AugmentationPipeline] = None,
                 corruption_pipeline: Optional[CorruptionPipeline] = None,
                 mode: str = 'augment'):
        """
        Initialize combined pipeline.
        
        Args:
            augmentation_pipeline: Augmentation pipeline instance
            corruption_pipeline: Corruption pipeline instance
            mode: 'augment' (training), 'corrupt' (testing), or 'both'
        """
        self.augmentation_pipeline = augmentation_pipeline
        self.corruption_pipeline = corruption_pipeline
        self.mode = mode
    
    def __call__(self, image: Image.Image, **kwargs) -> Image.Image:
        """
        Apply pipeline to image based on mode.
        
        Args:
            image: PIL Image
            **kwargs: Additional arguments (e.g., corruption_type for corruption mode)
        
        Returns:
            Processed PIL Image
        """
        if self.mode == 'augment' and self.augmentation_pipeline:
            return self.augmentation_pipeline(image)
        elif self.mode == 'corrupt' and self.corruption_pipeline:
            corruption_type = kwargs.get('corruption_type', None)
            return self.corruption_pipeline(image, corruption_type)
        elif self.mode == 'both':
            if self.augmentation_pipeline:
                image = self.augmentation_pipeline(image)
            if self.corruption_pipeline:
                corruption_type = kwargs.get('corruption_type', None)
                image = self.corruption_pipeline(image, corruption_type)
            return image
        else:
            return image
    
    def set_mode(self, mode: str):
        """Change pipeline mode."""
        if mode not in ['augment', 'corrupt', 'both']:
            raise ValueError(f"Mode must be 'augment', 'corrupt', or 'both', got {mode}")
        self.mode = mode


# Convenience functions
def create_training_augmentation_pipeline():
    """Create a standard augmentation pipeline for training."""
    configs = {
        'random_rotation': {'max_angle': 10},
        'random_flip': {'p': 0.5},
        'color_jitter': {
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.05
        }
    }
    return AugmentationPipeline(
        augmentation_types=['random_rotation', 'random_flip', 'color_jitter'],
        augmentation_configs=configs,
        apply_probability=0.8,
        sequential=True
    )


def create_robustness_corruption_pipeline(severity_levels: List[int] = [1, 2, 3, 4, 5]):
    """Create a standard corruption pipeline for robustness testing."""
    return CorruptionPipeline(
        corruption_types=None,
        severity=severity_levels,
        random_severity=False,
        apply_probability=1.0
    )

