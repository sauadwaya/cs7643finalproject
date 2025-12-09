import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import Counter

from .data_analysis import analyze_dataset, compare_datasets, load_dataset_data


def plot_caption_length_distribution(analysis: Dict, save_path: Optional[str] = None):
    """
    Plot caption length distribution.
    
    Args:
        analysis: Analysis dictionary from analyze_dataset
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, len(analysis.get('splits', {})), 
                            figsize=(5 * len(analysis.get('splits', {})), 4))
    
    if len(analysis.get('splits', {})) == 1:
        axes = [axes]
    
    for idx, (split, split_data) in enumerate(analysis.get('splits', {}).items()):
        caption_stats = split_data.get('caption_stats', {})
        lengths = caption_stats.get('caption_lengths', [])
        
        if lengths:
            axes[idx].hist(lengths, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].axvline(np.mean(lengths), color='r', linestyle='--', 
                            label=f'Mean: {np.mean(lengths):.1f}')
            axes[idx].set_xlabel('Caption Length (words)')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{split.upper()} Split')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_word_frequency(analysis: Dict, top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot most frequent words.
    
    Args:
        analysis: Analysis dictionary
        top_n: Number of top words to show
        save_path: Optional path to save plot
    """
    # Combine word counts from all splits
    all_word_counts = Counter()
    for split_data in analysis.get('splits', {}).values():
        caption_stats = split_data.get('caption_stats', {})
        word_counts = caption_stats.get('word_counts', {})
        all_word_counts.update(word_counts)
    
    if not all_word_counts:
        print("No word frequency data available")
        return
    
    top_words = all_word_counts.most_common(top_n)
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(words)), counts, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_dataset_comparison(comparison: Dict, metric: str = 'total_images', 
                           save_path: Optional[str] = None):
    """
    Plot comparison between datasets.
    
    Args:
        comparison: Comparison dictionary from compare_datasets
        metric: Metric to compare ('total_images', 'total_captions', 'avg_length', 'vocab_size')
        save_path: Optional path to save plot
    """
    summary = comparison.get('summary', {})
    
    if not summary:
        print("No comparison data available")
        return
    
    splits = list(summary.keys())
    if not splits:
        print("No splits found in comparison data")
        return
    
    datasets = list(summary[splits[0]].keys())
    if not datasets:
        print("No datasets found in comparison data")
        return
    
    # Map metric names to actual keys in summary
    metric_map = {
        'total_images': 'images',
        'total_captions': 'captions',
        'avg_length': 'avg_length',
        'vocab_size': 'vocab_size'
    }
    actual_metric = metric_map.get(metric, metric)
    
    # Prepare data
    data = {split: [summary[split][ds].get(actual_metric, 0) for ds in datasets] 
            for split in splits}
    
    # Check if we have any data
    all_values = [v for split_data in data.values() for v in split_data]
    if not all_values or all(v == 0 for v in all_values):
        print(f"No data available for metric '{metric}'")
        return
    
    x = np.arange(len(datasets))
    width = 0.8 / len(splits)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, split in enumerate(splits):
        offset = (idx - len(splits)/2 + 0.5) * width
        ax.bar(x + offset, data[split], width, label=split.upper(), alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Dataset Comparison: {metric.replace("_", " ").title()}')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_split_distribution(analysis: Dict, save_path: Optional[str] = None):
    """
    Plot distribution of data across splits.
    
    Args:
        analysis: Analysis dictionary
        save_path: Optional path to save plot
    """
    splits = list(analysis.get('splits', {}).keys())
    images = []
    captions = []
    
    for split in splits:
        split_data = analysis['splits'][split]
        caption_stats = split_data.get('caption_stats', {})
        images.append(caption_stats.get('total_images', 0))
        captions.append(caption_stats.get('total_captions', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Images
    ax1.pie(images, labels=splits, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Image Distribution')
    
    # Captions
    ax2.pie(captions, labels=splits, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Caption Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_caption_length_by_split(comparison: Dict, save_path: Optional[str] = None):
    """
    Plot caption length distribution comparison across datasets and splits.
    
    Args:
        comparison: Comparison dictionary
        save_path: Optional path to save plot
    """
    datasets = list(comparison.get('datasets', {}).keys())
    splits = ['train', 'dev', 'test']
    
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 5))
    if len(splits) == 1:
        axes = [axes]
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        
        for dataset_name in datasets:
            dataset_analysis = comparison['datasets'][dataset_name]
            if split in dataset_analysis.get('splits', {}):
                lengths = dataset_analysis['splits'][split]['caption_stats'].get('caption_lengths', [])
                if lengths:
                    ax.hist(lengths, bins=20, alpha=0.5, label=dataset_name, density=True)
        
        ax.set_xlabel('Caption Length (words)')
        ax.set_ylabel('Density')
        ax.set_title(f'{split.upper()} Split')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def create_analysis_report(analysis: Dict, output_dir: str = './analysis_reports'):
    """
    Create a comprehensive analysis report with all visualizations.
    
    Args:
        analysis: Analysis dictionary
        output_dir: Directory to save report files
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = analysis.get('dataset_name', 'unknown')
    
    print(f"\nCreating analysis report for {dataset_name}...")
    
    # Caption length distribution
    plot_caption_length_distribution(
        analysis, 
        save_path=os.path.join(output_dir, f'{dataset_name}_caption_lengths.png')
    )
    
    # Word frequency
    plot_word_frequency(
        analysis,
        save_path=os.path.join(output_dir, f'{dataset_name}_word_frequency.png')
    )
    
    # Split distribution
    plot_split_distribution(
        analysis,
        save_path=os.path.join(output_dir, f'{dataset_name}_split_distribution.png')
    )
    
    print(f"\nAnalysis report saved to {output_dir}/")


def create_comparison_report(comparison: Dict, output_dir: str = './analysis_reports'):
    """
    Create a comprehensive comparison report.
    
    Args:
        comparison: Comparison dictionary
        output_dir: Directory to save report files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating comparison report...")
    
    # Dataset comparison - images
    plot_dataset_comparison(
        comparison, 
        metric='total_images',
        save_path=os.path.join(output_dir, 'comparison_images.png')
    )
    
    # Dataset comparison - captions
    plot_dataset_comparison(
        comparison,
        metric='total_captions',
        save_path=os.path.join(output_dir, 'comparison_captions.png')
    )
    
    # Caption length comparison
    plot_caption_length_by_split(
        comparison,
        save_path=os.path.join(output_dir, 'comparison_caption_lengths.png')
    )
    
    print(f"\nComparison report saved to {output_dir}/")

