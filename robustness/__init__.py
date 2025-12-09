"""
Robustness Testing Module
Provides image corruption and augmentation pipelines for robustness testing.
"""

from .image_corruptions import (
    CORRUPTION_FUNCTIONS,
    apply_corruption,
    gaussian_noise,
    shot_noise,
    impulse_noise,
    defocus_blur,
    motion_blur,
    zoom_blur,
    brightness,
    contrast,
    jpeg_compression,
    pixelate,
    elastic_transform
)

from .image_augmentations import (
    AUGMENTATION_FUNCTIONS,
    apply_augmentation,
    random_rotation,
    random_flip,
    random_crop,
    color_jitter,
    random_erasing,
    gaussian_blur,
    add_gaussian_noise
)

from .corruption_pipeline import (
    CorruptionPipeline,
    AugmentationPipeline,
    CombinedPipeline,
    create_training_augmentation_pipeline,
    create_robustness_corruption_pipeline
)

from .corruption_tests import (
    evaluate_with_corruptions,
    evaluate_batch,
    calculate_caption_metrics,
    print_corruption_results,
    save_results,
    load_results,
    plot_corruption_results,
    calculate_robustness_summary,
    print_robustness_summary,
    create_corrupted_dataset
)

__all__ = [
    # Corruption functions
    'CORRUPTION_FUNCTIONS',
    'apply_corruption',
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'motion_blur',
    'zoom_blur',
    'brightness',
    'contrast',
    'jpeg_compression',
    'pixelate',
    'elastic_transform',
    # Augmentation functions
    'AUGMENTATION_FUNCTIONS',
    'apply_augmentation',
    'random_rotation',
    'random_flip',
    'random_crop',
    'color_jitter',
    'random_erasing',
    'gaussian_blur',
    'add_gaussian_noise',
    # Pipelines
    'CorruptionPipeline',
    'AugmentationPipeline',
    'CombinedPipeline',
    'create_training_augmentation_pipeline',
    'create_robustness_corruption_pipeline',
    # Testing utilities
    'evaluate_with_corruptions',
    'evaluate_batch',
    'calculate_caption_metrics',
    'print_corruption_results',
    'save_results',
    'load_results',
    'plot_corruption_results',
    'calculate_robustness_summary',
    'print_robustness_summary',
    'create_corrupted_dataset',
]

