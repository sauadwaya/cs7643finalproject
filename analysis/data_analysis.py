import os
import pickle
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import re
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset_data(dataset_name: str, split: str, captions_dir: str = './') -> Dict:
    """
    Load dataset caption data from pickle files.
    
    Args:
        dataset_name: Name of the dataset ('flickr8k' or 'flickr30k')
        split: Data split ('train', 'dev', 'test')
        captions_dir: Directory containing caption pickle files
    
    Returns:
        Dictionary mapping image filenames to lists of captions
    """
    if dataset_name.lower() == 'flickr8k':
        pickle_path = os.path.join(captions_dir, 'Flicker8k_captions', f'{split}_data.pickle')
    elif dataset_name.lower() == 'flickr30k':
        pickle_path = os.path.join(captions_dir, 'Flickr30k_captions', f'{split}_data.pickle')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'flickr8k', 'flickr30k'")
    
    if not os.path.exists(pickle_path):
        print(f"Warning: {pickle_path} not found. Returning empty dict.")
        return {}
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def analyze_captions(data: Dict) -> Dict:
    """
    Analyze caption statistics.
    
    Args:
        data: Dictionary mapping image filenames to lists of captions
    
    Returns:
        Dictionary with caption statistics
    """
    all_captions = []
    caption_lengths = []
    word_counts = Counter()
    unique_words = set()
    
    for filename, captions in data.items():
        for caption in captions:
            all_captions.append(caption)
            
            # Tokenize caption (simple word splitting)
            words = caption.lower().split()
            caption_lengths.append(len(words))
            
            # Count words
            for word in words:
                # Remove punctuation
                word = re.sub(r'[^\w\s]', '', word)
                if word:
                    word_counts[word] += 1
                    unique_words.add(word)
    
    stats = {
        'total_images': len(data),
        'total_captions': len(all_captions),
        'captions_per_image': len(all_captions) / len(data) if data else 0,
        'avg_caption_length': np.mean(caption_lengths) if caption_lengths else 0,
        'std_caption_length': np.std(caption_lengths) if caption_lengths else 0,
        'min_caption_length': np.min(caption_lengths) if caption_lengths else 0,
        'max_caption_length': np.max(caption_lengths) if caption_lengths else 0,
        'vocabulary_size': len(unique_words),
        'total_words': sum(word_counts.values()),
        'unique_words': len(unique_words),
        'word_counts': dict(word_counts.most_common(100)),  # Top 100 words
        'caption_lengths': caption_lengths
    }
    
    return stats


def analyze_images(data: Dict, image_dir: str) -> Dict:
    """
    Analyze image statistics.
    
    Args:
        data: Dictionary mapping image filenames to lists of captions
        image_dir: Directory containing images
    
    Returns:
        Dictionary with image statistics
    """
    image_sizes = []
    image_formats = Counter()
    total_size_bytes = 0
    missing_images = []
    
    for filename in data.keys():
        # Try different extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            test_path = os.path.join(image_dir, filename)
            if not os.path.splitext(test_path)[1]:
                test_path = test_path + ext
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                width, height = img.size
                image_sizes.append((width, height))
                image_formats[img.format] += 1
                total_size_bytes += os.path.getsize(image_path)
            except Exception as e:
                missing_images.append(filename)
        else:
            missing_images.append(filename)
    
    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        
        stats = {
            'total_images': len(data),
            'found_images': len(image_sizes),
            'missing_images': len(missing_images),
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'min_width': np.min(widths),
            'max_width': np.max(widths),
            'min_height': np.min(heights),
            'max_height': np.max(heights),
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'avg_size_kb': (total_size_bytes / len(image_sizes)) / 1024 if image_sizes else 0,
            'image_formats': dict(image_formats),
            'image_sizes': image_sizes
        }
    else:
        stats = {
            'total_images': len(data),
            'found_images': 0,
            'missing_images': len(missing_images),
            'error': 'No images found'
        }
    
    return stats


def analyze_dataset(dataset_name: str, splits: List[str] = ['train', 'dev', 'test'],
                   captions_dir: str = './', image_dirs: Optional[Dict[str, str]] = None) -> Dict:
    """
    Comprehensive dataset analysis.
    
    Args:
        dataset_name: Name of the dataset
        splits: List of splits to analyze
        captions_dir: Directory containing caption files
        image_dirs: Dictionary mapping splits to image directories
    
    Returns:
        Dictionary with complete analysis
    """
    analysis = {
        'dataset_name': dataset_name,
        'splits': {}
    }
    
    # Default image directories
    if image_dirs is None:
        if dataset_name.lower() == 'flickr8k':
            base_dir = os.path.join(captions_dir, '..', 'Flickr8k_Data', 'Flicker8k_Dataset')
            image_dirs = {split: base_dir for split in splits}
        elif dataset_name.lower() == 'flickr30k':
            base_dir = os.path.join(captions_dir, '..', 'Flickr30k_Data', 'flickr30k_images')
            image_dirs = {split: base_dir for split in splits}
        else:
            image_dirs = {split: '' for split in splits}
    
    for split in splits:
        print(f"Analyzing {dataset_name} {split} split...")
        
        # Load data
        data = load_dataset_data(dataset_name, split, captions_dir)
        
        if not data:
            print(f"  Warning: No data found for {split} split")
            continue
        
        # Analyze captions
        caption_stats = analyze_captions(data)
        
        # Analyze images
        image_dir = image_dirs.get(split, '')
        image_stats = analyze_images(data, image_dir) if image_dir else {}
        
        analysis['splits'][split] = {
            'caption_stats': caption_stats,
            'image_stats': image_stats
        }
    
    # Overall statistics
    all_captions = []
    all_lengths = []
    total_images = 0
    
    for split_data in analysis['splits'].values():
        caption_stats = split_data.get('caption_stats', {})
        all_captions.append(caption_stats.get('total_captions', 0))
        all_lengths.extend(caption_stats.get('caption_lengths', []))
        total_images += caption_stats.get('total_images', 0)
    
    analysis['overall'] = {
        'total_images': total_images,
        'total_captions': sum(all_captions),
        'avg_caption_length': np.mean(all_lengths) if all_lengths else 0,
        'vocabulary_size': len(set([w for split_data in analysis['splits'].values() 
                                   for w in split_data.get('caption_stats', {}).get('word_counts', {}).keys()]))
    }
    
    return analysis


def compare_datasets(dataset_names: List[str], splits: List[str] = ['train', 'dev', 'test'],
                    captions_dir: str = './') -> Dict:
    """
    Compare multiple datasets.
    
    Args:
        dataset_names: List of dataset names to compare
        splits: List of splits to compare
        captions_dir: Directory containing caption files
    
    Returns:
        Dictionary with comparison statistics
    """
    comparisons = {}
    
    for dataset_name in dataset_names:
        analysis = analyze_dataset(dataset_name, splits, captions_dir)
        comparisons[dataset_name] = analysis
    
    # Create comparison summary
    summary = {}
    for split in splits:
        summary[split] = {}
        for dataset_name in dataset_names:
            if split in comparisons[dataset_name]['splits']:
                caption_stats = comparisons[dataset_name]['splits'][split]['caption_stats']
                summary[split][dataset_name] = {
                    'images': caption_stats.get('total_images', 0),
                    'captions': caption_stats.get('total_captions', 0),
                    'avg_length': caption_stats.get('avg_caption_length', 0),
                    'vocab_size': caption_stats.get('vocabulary_size', 0)
                }
    
    return {
        'datasets': comparisons,
        'summary': summary
    }


def print_analysis(analysis: Dict, detailed: bool = False):
    """
    Print analysis results in a readable format.
    
    Args:
        analysis: Analysis dictionary from analyze_dataset
        detailed: Whether to print detailed statistics
    """
    dataset_name = analysis.get('dataset_name', 'Unknown')
    print("\n" + "="*80)
    print(f"DATASET ANALYSIS: {dataset_name.upper()}")
    print("="*80)
    
    # Overall statistics
    overall = analysis.get('overall', {})
    print(f"\nOverall Statistics:")
    print(f"  Total Images: {overall.get('total_images', 0)}")
    print(f"  Total Captions: {overall.get('total_captions', 0)}")
    print(f"  Average Caption Length: {overall.get('avg_caption_length', 0):.2f} words")
    print(f"  Vocabulary Size: {overall.get('vocabulary_size', 0)}")
    
    # Per-split statistics
    for split, split_data in analysis.get('splits', {}).items():
        print(f"\n{split.upper()} Split:")
        caption_stats = split_data.get('caption_stats', {})
        image_stats = split_data.get('image_stats', {})
        
        print(f"  Images: {caption_stats.get('total_images', 0)}")
        print(f"  Captions: {caption_stats.get('total_captions', 0)}")
        print(f"  Captions per Image: {caption_stats.get('captions_per_image', 0):.2f}")
        print(f"  Avg Caption Length: {caption_stats.get('avg_caption_length', 0):.2f} ± {caption_stats.get('std_caption_length', 0):.2f} words")
        print(f"  Caption Length Range: [{caption_stats.get('min_caption_length', 0)}, {caption_stats.get('max_caption_length', 0)}]")
        print(f"  Vocabulary Size: {caption_stats.get('vocabulary_size', 0)}")
        
        if image_stats:
            print(f"  Found Images: {image_stats.get('found_images', 0)}")
            if image_stats.get('found_images', 0) > 0:
                print(f"  Avg Image Size: {image_stats.get('avg_width', 0):.0f}x{image_stats.get('avg_height', 0):.0f}")
                print(f"  Total Size: {image_stats.get('total_size_mb', 0):.2f} MB")
        
        if detailed and caption_stats.get('word_counts'):
            print(f"\n  Top 10 Words:")
            top_words = sorted(caption_stats['word_counts'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
            for word, count in top_words:
                print(f"    {word}: {count}")


def print_comparison(comparison: Dict):
    """
    Print dataset comparison results.
    
    Args:
        comparison: Comparison dictionary from compare_datasets
    """
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    summary = comparison.get('summary', {})
    
    for split in summary.keys():
        print(f"\n{split.upper()} Split:")
        print(f"{'Dataset':<15} {'Images':<10} {'Captions':<10} {'Avg Length':<12} {'Vocab':<10}")
        print("-" * 60)
        
        for dataset_name, stats in summary[split].items():
            print(f"{dataset_name:<15} {stats['images']:<10} {stats['captions']:<10} "
                  f"{stats['avg_length']:<12.2f} {stats['vocab_size']:<10}")

