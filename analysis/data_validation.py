"""
Data Quality Validation Tools
Comprehensive validation utilities for checking data quality across datasets.
"""

import os
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from PIL import Image
import numpy as np


def load_dataset_data(dataset_name: str, split: str, captions_dir: str = './') -> Dict:
    """Load dataset caption data from pickle files."""
    if dataset_name.lower() == 'flickr8k':
        pickle_path = os.path.join(captions_dir, 'Flicker8k_captions', f'{split}_data.pickle')
    elif dataset_name.lower() == 'flickr30k':
        pickle_path = os.path.join(captions_dir, 'Flickr30k_captions', f'{split}_data.pickle')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(pickle_path):
        return {}
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def validate_image_exists(filename: str, image_dirs: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if image exists in any of the provided directories.
    
    Args:
        filename: Image filename
        image_dirs: List of directories to check
    
    Returns:
        Tuple of (exists, path_if_found)
    """
    for image_dir in image_dirs:
        if not image_dir or not os.path.exists(image_dir):
            continue
        
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            test_path = os.path.join(image_dir, filename)
            if not os.path.splitext(test_path)[1]:
                test_path = test_path + ext
            if os.path.exists(test_path):
                return True, test_path
    
    return False, None


def validate_image_integrity(image_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image file can be opened and is not corrupted.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (is_valid, error_message_if_invalid)
    """
    try:
        img = Image.open(image_path)
        img.verify()  # Verify the image is not corrupted
        return True, None
    except Exception as e:
        return False, str(e)


def validate_caption_format(caption: str) -> Tuple[bool, Optional[str]]:
    """
    Validate caption format and content.
    
    Args:
        caption: Caption string
    
    Returns:
        Tuple of (is_valid, error_message_if_invalid)
    """
    if not caption or not isinstance(caption, str):
        return False, "Caption is empty or not a string"
    
    if len(caption.strip()) == 0:
        return False, "Caption is empty after stripping"
    
    # Check for reasonable length (too short or too long might be errors)
    words = caption.split()
    if len(words) < 2:
        return False, f"Caption too short: {len(words)} words"
    
    if len(words) > 100:
        return False, f"Caption too long: {len(words)} words"
    
    # Check for excessive special characters (might indicate encoding issues)
    special_char_ratio = sum(1 for c in caption if not c.isalnum() and not c.isspace()) / len(caption)
    if special_char_ratio > 0.5:
        return False, f"Too many special characters: {special_char_ratio:.2%}"
    
    return True, None


def validate_dataset_split(data: Dict, split_name: str, image_dirs: List[str]) -> Dict:
    """
    Validate a single dataset split.
    
    Args:
        data: Dictionary mapping filenames to captions
        split_name: Name of the split
        image_dirs: List of image directories to check
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'split': split_name,
        'total_images': len(data),
        'issues': {
            'missing_images': [],
            'corrupted_images': [],
            'invalid_captions': [],
            'empty_captions': [],
            'duplicate_images': [],
            'images_without_captions': []
        },
        'statistics': {
            'valid_images': 0,
            'valid_captions': 0,
            'total_captions': 0,
            'avg_captions_per_image': 0
        }
    }
    
    seen_filenames = set()
    image_hashes = {}  # For duplicate detection
    
    for filename, captions in data.items():
        # Check for duplicate filenames
        if filename in seen_filenames:
            results['issues']['duplicate_images'].append(filename)
        seen_filenames.add(filename)
        
        # Validate image exists
        image_exists, image_path = validate_image_exists(filename, image_dirs)
        if not image_exists:
            results['issues']['missing_images'].append(filename)
        else:
            # Validate image integrity
            is_valid, error = validate_image_integrity(image_path)
            if not is_valid:
                results['issues']['corrupted_images'].append((filename, error))
            else:
                results['statistics']['valid_images'] += 1
                
                # Check for duplicate images (by content hash)
                try:
                    with open(image_path, 'rb') as f:
                        image_hash = hashlib.md5(f.read()).hexdigest()
                    if image_hash in image_hashes:
                        results['issues']['duplicate_images'].append(
                            f"{filename} (duplicate of {image_hashes[image_hash]})"
                        )
                    else:
                        image_hashes[image_hash] = filename
                except:
                    pass
        
        # Validate captions
        if not captions or len(captions) == 0:
            results['issues']['images_without_captions'].append(filename)
        else:
            results['statistics']['total_captions'] += len(captions)
            for caption in captions:
                is_valid, error = validate_caption_format(caption)
                if not is_valid:
                    results['issues']['invalid_captions'].append((filename, error))
                else:
                    results['statistics']['valid_captions'] += 1
    
    # Calculate statistics
    if results['statistics']['total_captions'] > 0:
        results['statistics']['avg_captions_per_image'] = (
            results['statistics']['total_captions'] / len(data)
        )
    
    return results


def validate_dataset(dataset_name: str, splits: List[str] = ['train', 'dev', 'test'],
                    captions_dir: str = './', image_dirs: Optional[Dict[str, List[str]]] = None) -> Dict:
    """
    Validate entire dataset across all splits.
    
    Args:
        dataset_name: Name of the dataset
        splits: List of splits to validate
        captions_dir: Directory containing caption files
        image_dirs: Dictionary mapping splits to lists of image directories
    
    Returns:
        Dictionary with validation results
    """
    # Default image directories
    if image_dirs is None:
        if dataset_name.lower() == 'flickr8k':
            base_dir = os.path.join(captions_dir, '..', 'Flickr8k_Data', 'Flicker8k_Dataset')
            resized_dir = os.path.join(captions_dir, '..', 'Flickr8k_Data', 'Flicker8k_Dataset_resized')
            image_dirs = {split: [base_dir, resized_dir] for split in splits}
        elif dataset_name.lower() == 'flickr30k':
            base_dir = os.path.join(captions_dir, '..', 'Flickr30k_Data', 'flickr30k_images')
            resized_dir = os.path.join(captions_dir, '..', 'Flickr30k_Data', 'flickr30k_images_resized')
            image_dirs = {split: [base_dir, resized_dir] for split in splits}
        else:
            image_dirs = {split: [] for split in splits}
    
    validation_results = {
        'dataset_name': dataset_name,
        'splits': {},
        'overall': {
            'total_images': 0,
            'valid_images': 0,
            'total_captions': 0,
            'valid_captions': 0,
            'total_issues': 0
        }
    }
    
    for split in splits:
        print(f"Validating {dataset_name} {split} split...")
        
        # Load data
        data = load_dataset_data(dataset_name, split, captions_dir)
        
        if not data:
            print(f"  Warning: No data found for {split} split")
            continue
        
        # Validate split
        split_dirs = image_dirs.get(split, [])
        split_results = validate_dataset_split(data, split, split_dirs)
        validation_results['splits'][split] = split_results
        
        # Update overall statistics
        validation_results['overall']['total_images'] += split_results['total_images']
        validation_results['overall']['valid_images'] += split_results['statistics']['valid_images']
        validation_results['overall']['total_captions'] += split_results['statistics']['total_captions']
        validation_results['overall']['valid_captions'] += split_results['statistics']['valid_captions']
        
        # Count total issues
        for issue_type, issues in split_results['issues'].items():
            validation_results['overall']['total_issues'] += len(issues)
    
    return validation_results


def validate_cross_dataset_consistency(dataset_names: List[str], splits: List[str] = ['train', 'dev', 'test'],
                                       captions_dir: str = './') -> Dict:
    """
    Validate consistency across multiple datasets.
    
    Args:
        dataset_names: List of dataset names to compare
        splits: List of splits to validate
        captions_dir: Directory containing caption files
    
    Returns:
        Dictionary with cross-dataset validation results
    """
    results = {
        'datasets': {},
        'consistency_issues': {
            'overlapping_images': [],
            'inconsistent_splits': {},
            'caption_overlap': {}
        }
    }
    
    all_image_sets = {}
    all_caption_sets = {}
    
    # Collect data from all datasets
    for dataset_name in dataset_names:
        dataset_images = {}
        dataset_captions = {}
        
        for split in splits:
            data = load_dataset_data(dataset_name, split, captions_dir)
            dataset_images[split] = set(data.keys())
            dataset_captions[split] = set()
            
            for filename, captions in data.items():
                for caption in captions:
                    dataset_captions[split].add(caption.lower().strip())
        
        all_image_sets[dataset_name] = dataset_images
        all_caption_sets[dataset_name] = dataset_captions
        
        # Validate individual dataset
        validation = validate_dataset(dataset_name, splits, captions_dir)
        results['datasets'][dataset_name] = validation
    
    # Check for overlapping images across datasets
    for i, dataset1 in enumerate(dataset_names):
        for dataset2 in dataset_names[i+1:]:
            for split in splits:
                images1 = all_image_sets[dataset1].get(split, set())
                images2 = all_image_sets[dataset2].get(split, set())
                overlap = images1.intersection(images2)
                if overlap:
                    results['consistency_issues']['overlapping_images'].append({
                        'datasets': (dataset1, dataset2),
                        'split': split,
                        'overlapping_images': list(overlap)[:10]  # First 10
                    })
    
    # Check split consistency
    for split in splits:
        split_sizes = {}
        for dataset_name in dataset_names:
            data = load_dataset_data(dataset_name, split, captions_dir)
            split_sizes[dataset_name] = len(data)
        
        # Check if splits are significantly different in size
        sizes = list(split_sizes.values())
        if sizes and max(sizes) / min(sizes) > 2:  # More than 2x difference
            results['consistency_issues']['inconsistent_splits'][split] = split_sizes
    
    return results


def print_validation_results(validation_results: Dict, detailed: bool = False):
    """
    Print validation results in a readable format.
    
    Args:
        validation_results: Validation results dictionary
        detailed: Whether to print detailed issue lists
    """
    dataset_name = validation_results.get('dataset_name', 'Unknown')
    print("\n" + "="*80)
    print(f"DATA VALIDATION RESULTS: {dataset_name.upper()}")
    print("="*80)
    
    overall = validation_results.get('overall', {})
    print(f"\nOverall Statistics:")
    print(f"  Total Images: {overall.get('total_images', 0)}")
    print(f"  Valid Images: {overall.get('valid_images', 0)} ({overall.get('valid_images', 0)/max(overall.get('total_images', 1), 1)*100:.1f}%)")
    print(f"  Total Captions: {overall.get('total_captions', 0)}")
    print(f"  Valid Captions: {overall.get('valid_captions', 0)} ({overall.get('valid_captions', 0)/max(overall.get('total_captions', 1), 1)*100:.1f}%)")
    print(f"  Total Issues: {overall.get('total_issues', 0)}")
    
    # Per-split results
    for split, split_results in validation_results.get('splits', {}).items():
        print(f"\n{split.upper()} Split:")
        stats = split_results.get('statistics', {})
        issues = split_results.get('issues', {})
        
        print(f"  Images: {split_results.get('total_images', 0)}")
        print(f"  Valid Images: {stats.get('valid_images', 0)}")
        print(f"  Valid Captions: {stats.get('valid_captions', 0)} / {stats.get('total_captions', 0)}")
        print(f"  Avg Captions per Image: {stats.get('avg_captions_per_image', 0):.2f}")
        
        # Issue summary
        print(f"\n  Issues:")
        for issue_type, issue_list in issues.items():
            count = len(issue_list)
            if count > 0:
                print(f"    {issue_type.replace('_', ' ').title()}: {count}")
                if detailed and count <= 10:
                    for issue in issue_list[:5]:  # Show first 5
                        if isinstance(issue, tuple):
                            print(f"      - {issue[0]}: {issue[1]}")
                        else:
                            print(f"      - {issue}")
                    if count > 5:
                        print(f"      ... and {count - 5} more")


def print_cross_dataset_validation(results: Dict):
    """
    Print cross-dataset validation results.
    
    Args:
        results: Cross-dataset validation results
    """
    print("\n" + "="*80)
    print("CROSS-DATASET VALIDATION RESULTS")
    print("="*80)
    
    # Individual dataset results
    for dataset_name, validation in results.get('datasets', {}).items():
        print(f"\n{dataset_name.upper()}:")
        overall = validation.get('overall', {})
        print(f"  Valid Images: {overall.get('valid_images', 0)}/{overall.get('total_images', 0)} "
              f"({overall.get('valid_images', 0)/max(overall.get('total_images', 1), 1)*100:.1f}%)")
        print(f"  Total Issues: {overall.get('total_issues', 0)}")
    
    # Consistency issues
    consistency = results.get('consistency_issues', {})
    
    if consistency.get('overlapping_images'):
        print(f"\n⚠️  Overlapping Images Found:")
        for overlap_info in consistency['overlapping_images']:
            datasets = overlap_info['datasets']
            split = overlap_info['split']
            count = len(overlap_info['overlapping_images'])
            print(f"  {datasets[0]} ↔ {datasets[1]} ({split}): {count} overlapping images")
    
    if consistency.get('inconsistent_splits'):
        print(f"\n⚠️  Inconsistent Split Sizes:")
        for split, sizes in consistency['inconsistent_splits'].items():
            print(f"  {split}: {sizes}")


def generate_validation_report(validation_results: Dict, output_path: str = './validation_report.txt'):
    """
    Generate a text report of validation results.
    
    Args:
        validation_results: Validation results dictionary
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"DATA VALIDATION REPORT: {validation_results.get('dataset_name', 'Unknown').upper()}\n")
        f.write("="*80 + "\n\n")
        
        overall = validation_results.get('overall', {})
        f.write("Overall Statistics:\n")
        f.write(f"  Total Images: {overall.get('total_images', 0)}\n")
        f.write(f"  Valid Images: {overall.get('valid_images', 0)}\n")
        f.write(f"  Total Captions: {overall.get('total_captions', 0)}\n")
        f.write(f"  Valid Captions: {overall.get('valid_captions', 0)}\n")
        f.write(f"  Total Issues: {overall.get('total_issues', 0)}\n\n")
        
        for split, split_results in validation_results.get('splits', {}).items():
            f.write(f"{split.upper()} Split:\n")
            issues = split_results.get('issues', {})
            for issue_type, issue_list in issues.items():
                if issue_list:
                    f.write(f"  {issue_type.replace('_', ' ').title()}: {len(issue_list)}\n")
                    for issue in issue_list[:20]:  # First 20 issues
                        if isinstance(issue, tuple):
                            f.write(f"    - {issue[0]}: {issue[1]}\n")
                        else:
                            f.write(f"    - {issue}\n")
            f.write("\n")
    
    print(f"Validation report saved to {output_path}")

