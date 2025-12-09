"""
Image Augmentation Functions
Implements various image augmentations for training and robustness testing.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import cv2


def random_rotation(image, max_angle=15):
    """
    Randomly rotate image.
    
    Args:
        image: PIL Image
        max_angle: Maximum rotation angle in degrees
    
    Returns:
        Augmented PIL Image
    """
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))


def random_flip(image, p=0.5):
    """
    Randomly flip image horizontally.
    
    Args:
        image: PIL Image
        p: Probability of flipping
    
    Returns:
        Augmented PIL Image
    """
    if random.random() < p:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_crop(image, crop_ratio=0.9):
    """
    Random crop and resize to original size.
    
    Args:
        image: PIL Image
        crop_ratio: Ratio of image to keep (0-1)
    
    Returns:
        Augmented PIL Image
    """
    w, h = image.size
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Apply random color jitter.
    
    Args:
        image: PIL Image
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range
        hue: Hue jitter range
    
    Returns:
        Augmented PIL Image
    """
    if brightness > 0:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(1 - brightness, 1 + brightness)
        image = enhancer.enhance(factor)
    
    if contrast > 0:
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(1 - contrast, 1 + contrast)
        image = enhancer.enhance(factor)
    
    if saturation > 0:
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(1 - saturation, 1 + saturation)
        image = enhancer.enhance(factor)
    
    if hue > 0:
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue * 180, hue * 180)) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image = Image.fromarray(img_array)
    
    return image


def random_erasing(image, p=0.5, area_ratio=(0.02, 0.33), aspect_ratio=(0.3, 3.3)):
    """
    Randomly erase a rectangular region of the image.
    
    Args:
        image: PIL Image
        p: Probability of applying erasing
        area_ratio: Range of area ratio to erase
        aspect_ratio: Range of aspect ratio for erased region
    
    Returns:
        Augmented PIL Image
    """
    if random.random() > p:
        return image
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Calculate erase area
    img_area = h * w
    erase_area = img_area * random.uniform(area_ratio[0], area_ratio[1])
    aspect = random.uniform(aspect_ratio[0], aspect_ratio[1])
    
    erase_h = int(np.sqrt(erase_area * aspect))
    erase_w = int(np.sqrt(erase_area / aspect))
    
    if erase_h < h and erase_w < w:
        top = random.randint(0, h - erase_h)
        left = random.randint(0, w - erase_w)
        
        img_array[top:top+erase_h, left:left+erase_w] = np.random.randint(
            0, 255, (erase_h, erase_w, img_array.shape[2])
        )
    
    return Image.fromarray(img_array)


def gaussian_blur(image, sigma_range=(0.1, 2.0)):
    """
    Apply random Gaussian blur.
    
    Args:
        image: PIL Image
        sigma_range: Range of blur sigma values
    
    Returns:
        Augmented PIL Image
    """
    import cv2
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    img_array = np.array(image)
    blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)
    return Image.fromarray(blurred)


def add_gaussian_noise(image, std_range=(0.01, 0.1)):
    """
    Add random Gaussian noise.
    
    Args:
        image: PIL Image
        std_range: Range of noise standard deviation
    
    Returns:
        Augmented PIL Image
    """
    std = random.uniform(std_range[0], std_range[1])
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, img_array.shape)
    corrupted = np.clip(img_array + noise, 0, 1)
    return Image.fromarray((corrupted * 255).astype(np.uint8))


# Dictionary of all augmentation functions
AUGMENTATION_FUNCTIONS = {
    'random_rotation': random_rotation,
    'random_flip': random_flip,
    'random_crop': random_crop,
    'color_jitter': color_jitter,
    'random_erasing': random_erasing,
    'gaussian_blur': gaussian_blur,
    'add_gaussian_noise': add_gaussian_noise,
}


def apply_augmentation(image, augmentation_type, **kwargs):
    """
    Apply a specific augmentation to an image.
    
    Args:
        image: PIL Image
        augmentation_type: Name of augmentation function
        **kwargs: Additional arguments for the augmentation function
    
    Returns:
        Augmented PIL Image
    """
    if augmentation_type not in AUGMENTATION_FUNCTIONS:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}. "
                        f"Available: {list(AUGMENTATION_FUNCTIONS.keys())}")
    
    return AUGMENTATION_FUNCTIONS[augmentation_type](image, **kwargs)

