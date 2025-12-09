"""
Image Corruption Functions
Implements various image corruptions for robustness testing.
Based on common corruptions used in robustness benchmarks.
"""

import numpy as np
from PIL import Image
import cv2
import io


def gaussian_noise(image, severity=1):
    """
    Add Gaussian noise to image.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5 (higher = more noise)
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [0.08, 0.12, 0.18, 0.26, 0.38]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, c, img_array.shape)
    corrupted = np.clip(img_array + noise, 0, 1)
    
    return Image.fromarray((corrupted * 255).astype(np.uint8))


def shot_noise(image, severity=1):
    """
    Add shot (Poisson) noise to image.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [60, 25, 12, 5, 3]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image).astype(np.float32)
    corrupted = np.random.poisson(img_array * c) / c
    corrupted = np.clip(corrupted, 0, 255)
    
    return Image.fromarray(corrupted.astype(np.uint8))


def impulse_noise(image, severity=1):
    """
    Add impulse (salt and pepper) noise to image.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [0.03, 0.06, 0.09, 0.17, 0.27]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image)
    mask = np.random.random(img_array.shape[:2]) < c
    noise = np.random.random(img_array.shape[:2]) < 0.5
    
    corrupted = img_array.copy()
    corrupted[mask & noise] = 0
    corrupted[mask & ~noise] = 255
    
    return Image.fromarray(corrupted)


def defocus_blur(image, severity=1):
    """
    Apply defocus blur to image.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [3, 5, 7, 9, 11]  # odd numbers for OpenCV
    c = severity_levels[severity - 1]
    
    img_array = np.array(image)
    kernel = np.zeros((c, c))
    kernel[c//2, c//2] = 1
    # Ensure kernel size is odd for GaussianBlur
    ksize = c if c % 2 == 1 else c + 1
    kernel = cv2.GaussianBlur(kernel, (ksize, ksize), c/3)
    kernel = kernel / kernel.sum()
    
    corrupted = cv2.filter2D(img_array, -1, kernel)
    
    return Image.fromarray(corrupted)


def motion_blur(image, severity=1):
    """
    Apply motion blur to image.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [10, 15, 20, 25, 30]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image)
    kernel = np.zeros((c, c))
    kernel[int((c-1)/2), :] = np.ones(c)
    kernel = kernel / c
    
    # Random rotation
    angle = np.random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((c/2, c/2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (c, c))
    
    corrupted = cv2.filter2D(img_array, -1, kernel)
    
    return Image.fromarray(corrupted)


def zoom_blur(image, severity=1):
    """
    Apply zoom blur to image.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [0.15, 0.25, 0.50, 0.75, 1.00]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    num_samples = 8
    zoomed_images = []
    
    for i in range(num_samples):
        zoom_factor = 1 + c * (i / (num_samples - 1))
        center_x, center_y = w // 2, h // 2
        
        # Calculate crop size
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        
        # Crop and resize
        left = max(0, center_x - crop_w // 2)
        top = max(0, center_y - crop_h // 2)
        right = min(w, center_x + crop_w // 2)
        bottom = min(h, center_y + crop_h // 2)
        
        cropped = img_array[top:bottom, left:right]
        resized = cv2.resize(cropped, (w, h))
        zoomed_images.append(resized)
    
    corrupted = np.mean(zoomed_images, axis=0).astype(np.uint8)
    
    return Image.fromarray(corrupted)


def brightness(image, severity=1):
    """
    Adjust image brightness.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5 (1-2 darker, 4-5 brighter)
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [0.4, 0.3, 0.2, 0.1, 0.05]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image).astype(np.float32)
    
    # Darken or brighten based on severity
    if severity <= 2:
        corrupted = img_array * c
    else:
        corrupted = np.clip(img_array + (1 - c) * 255, 0, 255)
    
    return Image.fromarray(corrupted.astype(np.uint8))


def contrast(image, severity=1):
    """
    Adjust image contrast.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [0.4, 0.3, 0.2, 0.1, 0.05]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image).astype(np.float32)
    mean = img_array.mean()
    corrupted = (img_array - mean) * c + mean
    corrupted = np.clip(corrupted, 0, 255)
    
    return Image.fromarray(corrupted.astype(np.uint8))


def jpeg_compression(image, severity=1):
    """
    Apply JPEG compression artifacts.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5 (lower quality = higher severity)
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [25, 18, 15, 10, 7]
    quality = severity_levels[severity - 1]
    
    # Convert to JPEG and back
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    corrupted = Image.open(buffer)
    corrupted = corrupted.convert('RGB')
    
    return corrupted


def pixelate(image, severity=1):
    """
    Pixelate image by downsampling and upsampling.
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [0.6, 0.5, 0.4, 0.3, 0.25]
    c = severity_levels[severity - 1]
    
    w, h = image.size
    new_w, new_h = int(w * c), int(h * c)
    
    downsampled = image.resize((new_w, new_h), Image.BILINEAR)
    corrupted = downsampled.resize((w, h), Image.NEAREST)
    
    return corrupted


def elastic_transform(image, severity=1):
    """
    Apply elastic transformation (distortion).
    
    Args:
        image: PIL Image
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    severity_levels = [244, 488, 730, 972, 1093]
    c = severity_levels[severity - 1]
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Random displacement fields
    dx = np.random.uniform(-1, 1, (h, w)) * c
    dy = np.random.uniform(-1, 1, (h, w)) * c
    
    # Smooth the displacement fields
    dx = cv2.GaussianBlur(dx, (17, 17), 0)
    dy = cv2.GaussianBlur(dy, (17, 17), 0)
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # Apply transformation
    corrupted = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return Image.fromarray(corrupted)


CORRUPTION_FUNCTIONS = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'brightness': brightness,
    'contrast': contrast,
    'jpeg_compression': jpeg_compression,
    'pixelate': pixelate,
    'elastic_transform': elastic_transform,
}


def apply_corruption(image, corruption_type, severity=1):
    """
    Apply a specific corruption to an image.
    
    Args:
        image: PIL Image
        corruption_type: Name of corruption function
        severity: Severity level 1-5
    
    Returns:
        Corrupted PIL Image
    """
    if corruption_type not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption type: {corruption_type}. "
                        f"Available: {list(CORRUPTION_FUNCTIONS.keys())}")
    
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be between 1 and 5, got {severity}")
    
    return CORRUPTION_FUNCTIONS[corruption_type](image, severity)

