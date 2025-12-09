"""
Robustness Testing Utilities
Comprehensive utilities for evaluating model robustness with corruptions.
Includes metrics calculation, result saving/loading, visualization, and analysis.
"""

import os
import json
import pickle
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from .corruption_pipeline import CorruptionPipeline, create_robustness_corruption_pipeline
from .image_corruptions import CORRUPTION_FUNCTIONS

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU scores will not be calculated.")


def evaluate_with_corruptions(model, dataloader, tokenizer, device, 
                             corruption_pipeline: Optional[CorruptionPipeline] = None,
                             corruption_types: Optional[List[str]] = None,
                             severity_levels: List[int] = [1, 2, 3, 4, 5],
                             img_processor=None,
                             checkpoint_file: Optional[str] = None,
                             dataset_name: str = 'flickr8k',
                             split: str = 'test',
                             base_dir: str = './',
                             max_len: int = 48,
                             batch_size: int = 32):
    """
    Evaluate model performance with various image corruptions.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        tokenizer: Tokenizer for decoding
        device: Device to run on
        corruption_pipeline: Pre-configured corruption pipeline
        corruption_types: List of corruption types to test
        severity_levels: List of severity levels to test
        img_processor: Image processor (if needed for custom processing)
        checkpoint_file: Optional path to checkpoint file for saving/loading progress
    
    Returns:
        Dictionary with results for each corruption type and severity
    """
    model.eval()
    
    if corruption_pipeline is None:
        if corruption_types is None:
            corruption_types = list(CORRUPTION_FUNCTIONS.keys())
        corruption_pipeline = CorruptionPipeline(
            corruption_types=corruption_types,
            severity=severity_levels,
            random_severity=False
        )
    
    results = {}
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            results = load_results(checkpoint_file, format='json')
            print(f"Loaded existing checkpoint from {checkpoint_file}")
            print(f"Found {len([k for k in results.keys() if k != 'baseline'])} completed corruption types")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")
            results = {}
    
    if 'baseline' not in results:
        print("Evaluating baseline (no corruption)...")
        baseline_loss, baseline_metrics = evaluate_batch(model, dataloader, tokenizer, device, 
                                                         img_processor=img_processor)
        results['baseline'] = {
            'loss': baseline_loss,
            'metrics': baseline_metrics
        }
        if checkpoint_file:
            save_results(results, checkpoint_file, format='json')
            print(f"Checkpoint saved: baseline complete")
    else:
        print("Baseline already evaluated, skipping...")
    
    for corruption_type in corruption_pipeline.corruption_types:
        if corruption_type not in results:
            results[corruption_type] = {}
        
        for severity in severity_levels:
            severity_key = f'severity_{severity}'
            
            if severity_key in results[corruption_type]:
                print(f"Skipping {corruption_type} severity {severity} (already completed)...")
                continue
            
            print(f"Evaluating {corruption_type} severity {severity}...")
            
            test_pipeline = CorruptionPipeline(
                corruption_types=[corruption_type],
                severity=[severity],
                apply_probability=1.0
            )
            
            try:
                from data.multi_dataset_loader import create_multi_dataset
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from data.multi_dataset_loader import create_multi_dataset
            
            corrupted_dataset = create_multi_dataset(
                datasets=[dataset_name],
                split=split,
                tokenizer=tokenizer,
                img_processor=img_processor,
                max_len=max_len,
                base_dir=base_dir,
                use_resized=True,
                corruption_pipeline=test_pipeline
            )
            corrupted_loader = DataLoader(corrupted_dataset, batch_size=batch_size, shuffle=False)
            
            loss, metrics = evaluate_batch(model, corrupted_loader, tokenizer, device,
                                         img_processor=img_processor)
            
            results[corruption_type][severity_key] = {
                'loss': loss,
                'metrics': metrics
            }
            
            if checkpoint_file:
                save_results(results, checkpoint_file, format='json')
                print(f"Checkpoint saved: {corruption_type} severity {severity} complete")
        
        if checkpoint_file:
            save_results(results, checkpoint_file, format='json')
            print(f"Checkpoint saved: {corruption_type} complete")
    
    return results


def evaluate_batch(model, dataloader, tokenizer, device, 
                  img_processor=None):
    """
    Evaluate a single batch.
    
    Args:
        model: Trained model
        dataloader: DataLoader (should already have corruption applied if needed)
        tokenizer: Tokenizer
        device: Device
        img_processor: Image processor (unused, kept for compatibility)
    
    Returns:
        Average loss and metrics dictionary
    """
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask')
            
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
                decoder_attention_mask=attention_mask.to(device) if attention_mask is not None else None
            )
            
            total_loss += outputs.loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=50,
                decoder_start_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=3,
                repetition_penalty=2.0  # to prevent "stuttering"
            )
            
            generated_ids = generated_ids.cpu().numpy()
            vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
            eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
            
            preds = []
            for gen_ids in generated_ids:
                gen_ids_list = gen_ids.tolist()
                
                while gen_ids_list and gen_ids_list[0] in [pad_token_id, eos_token_id]:
                    gen_ids_list = gen_ids_list[1:]
                
                while gen_ids_list and gen_ids_list[-1] == pad_token_id:
                    gen_ids_list = gen_ids_list[:-1]
                
                valid_ids = [int(tid) for tid in gen_ids_list if 0 <= tid < vocab_size]
                
                if valid_ids:
                    try:
                        decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
                        preds.append(decoded if decoded.strip() else "")
                    except (OverflowError, ValueError):
                        preds.append("")
                else:
                    preds.append("")
            
            refs = []
            labels_np = labels.cpu().numpy()
            for label_seq in labels_np:
                # Filter out -100 (label padding) and invalid token IDs
                valid_ids = [int(tid) for tid in label_seq if tid != -100 and 0 <= tid < vocab_size]
                if valid_ids:
                    try:
                        decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
                        refs.append(decoded if decoded.strip() else "")
                    except (OverflowError, ValueError):
                        refs.append("")
                else:
                    refs.append("")
            
            all_predictions.extend(preds)
            all_references.extend(refs)
    
    avg_loss = total_loss / len(dataloader)
    
    if len(all_predictions) > 0:
        print(f"  Sample prediction: {all_predictions[0][:100] if len(all_predictions[0]) > 100 else all_predictions[0]}")
        print(f"  Sample reference: {all_references[0][:100] if len(all_references[0]) > 100 else all_references[0]}")
        print(f"  Total predictions: {len(all_predictions)}, Non-empty: {sum(1 for p in all_predictions if p.strip())}")
        print(f"  Total references: {len(all_references)}, Non-empty: {sum(1 for r in all_references if r.strip())}")
    
    metrics = calculate_caption_metrics(all_predictions, all_references)
    metrics['loss'] = avg_loss
    metrics['num_samples'] = len(all_predictions)
    
    metrics['_debug_predictions'] = all_predictions[:3] if len(all_predictions) >= 3 else all_predictions
    metrics['_debug_references'] = all_references[:3] if len(all_references) >= 3 else all_references
    
    return avg_loss, metrics


def calculate_caption_metrics(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate captioning metrics (BLEU, etc.).
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    valid_preds = [p for p in predictions if p and p.strip()]
    valid_refs = [r for r in references if r and r.strip()]
    
    if not valid_preds or not valid_refs:
        print(f"  Warning: No valid predictions ({len(valid_preds)}) or references ({len(valid_refs)}) for BLEU calculation")
        metrics['bleu_1'] = 0.0
        metrics['bleu_2'] = 0.0
        metrics['bleu_3'] = 0.0
        metrics['bleu_4'] = 0.0
        return metrics
    
    if not NLTK_AVAILABLE:
        print("  Warning: NLTK not available. BLEU scores cannot be calculated.")
        metrics['bleu_1'] = 0.0
        metrics['bleu_2'] = 0.0
        metrics['bleu_3'] = 0.0
        metrics['bleu_4'] = 0.0
        return metrics
    
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and p.strip() and r and r.strip()]
    
    if not valid_pairs:
        print(f"  Warning: No valid prediction-reference pairs for BLEU calculation")
        metrics['bleu_1'] = 0.0
        metrics['bleu_2'] = 0.0
        metrics['bleu_3'] = 0.0
        metrics['bleu_4'] = 0.0
        return metrics
    
    error_count = 0
    for pred, ref in valid_pairs:
        try:
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            
            if not pred_tokens or not ref_tokens:
                continue
            
            bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu([ref_tokens], pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            bleu_scores.append({
                'bleu_1': bleu_1,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3,
                'bleu_4': bleu_4
            })
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # Print first 3 errors for debugging
                print(f"  Warning: BLEU calculation error: {e}")
            continue
    
    if error_count > 0:
        print(f"  Warning: {error_count} BLEU calculation errors out of {len(valid_pairs)} pairs")
    
    if bleu_scores:
        metrics['bleu_1'] = np.mean([s['bleu_1'] for s in bleu_scores])
        metrics['bleu_2'] = np.mean([s['bleu_2'] for s in bleu_scores])
        metrics['bleu_3'] = np.mean([s['bleu_3'] for s in bleu_scores])
        metrics['bleu_4'] = np.mean([s['bleu_4'] for s in bleu_scores])
    else:
        # If no BLEU scores calculated, set to 0
        metrics['bleu_1'] = 0.0
        metrics['bleu_2'] = 0.0
        metrics['bleu_3'] = 0.0
        metrics['bleu_4'] = 0.0
    
    avg_pred_len = np.mean([len(pred.split()) for pred in predictions])
    avg_ref_len = np.mean([len(ref.split()) for ref in references])
    metrics['avg_pred_length'] = avg_pred_len
    metrics['avg_ref_length'] = avg_ref_len
    
    return metrics


def print_corruption_results(results: Dict, metrics_to_show: List[str] = ['loss', 'bleu_4']):
    """
    Print corruption test results in a readable format.
    
    Args:
        results: Results dictionary from evaluate_with_corruptions
        metrics_to_show: List of metrics to display
    """
    print("\n" + "="*80)
    print("CORRUPTION ROBUSTNESS TEST RESULTS")
    print("="*80)
    
    # Baseline
    baseline = results.get('baseline', {})
    baseline_metrics = baseline.get('metrics', {})
    print(f"\nBaseline (No Corruption):")
    for metric in metrics_to_show:
        if metric == 'loss':
            value = baseline.get('loss', 'N/A')
        else:
            value = baseline_metrics.get(metric, 'N/A')
        if isinstance(value, float):
            print(f"  {metric.upper()}: {value:.4f}")
        else:
            print(f"  {metric.upper()}: {value}")
    
    # Corruptions
    for corruption_type in sorted(results.keys()):
        if corruption_type == 'baseline':
            continue
        
        print(f"\n{corruption_type.upper().replace('_', ' ')}:")
        corruption_results = results[corruption_type]
        
        for severity_key in sorted(corruption_results.keys()):
            severity_data = corruption_results[severity_key]
            metrics = severity_data.get('metrics', {})
            
            print(f"  {severity_key}:")
            for metric in metrics_to_show:
                if metric == 'loss':
                    value = severity_data.get('loss', 'N/A')
                else:
                    value = metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    print(f"    {metric.upper()}: {value:.4f}")
                else:
                    print(f"    {metric.upper()}: {value}")


def save_results(results: Dict, filepath: str, format: str = 'json'):
    """
    Save robustness test results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
        format: 'json' or 'pickle'
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    if format == 'json':
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'pickle'")
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = 'json') -> Dict:
    """
    Load robustness test results from file.
    
    Args:
        filepath: Path to results file
        format: 'json' or 'pickle'
    
    Returns:
        Results dictionary
    """
    if format == 'json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'pickle'")
    
    return results


def plot_corruption_results(results: Dict, metric: str = 'loss', save_path: Optional[str] = None):
    """
    Plot corruption test results.
    
    Args:
        results: Results dictionary
        metric: Metric to plot ('loss', 'bleu_1', 'bleu_2', etc.)
        save_path: Optional path to save plot
    """
    baseline = results.get('baseline', {})
    if metric == 'loss':
        baseline_value = baseline.get('loss', 0)
    else:
        baseline_value = baseline.get('metrics', {}).get(metric, 0)
    
    corruption_data = []
    for corruption_type in sorted(results.keys()):
        if corruption_type == 'baseline':
            continue
        
        corruption_results = results[corruption_type]
        for severity_key in sorted(corruption_results.keys()):
            severity_data = corruption_results[severity_key]
            severity_num = int(severity_key.split('_')[1])
            
            if metric == 'loss':
                value = severity_data.get('loss', 0)
            else:
                value = severity_data.get('metrics', {}).get(metric, 0)
            
            corruption_data.append({
                'corruption': corruption_type,
                'severity': severity_num,
                'value': value
            })
    
    if not corruption_data:
        print("No corruption data to plot")
        return
    
    corruptions = [d['corruption'] for d in corruption_data]
    severities = [d['severity'] for d in corruption_data]
    values = [d['value'] for d in corruption_data]
    
    plt.figure(figsize=(14, 8))
    
    unique_corruptions = sorted(set(corruptions))
    x_pos = np.arange(len(unique_corruptions))
    width = 0.15
    
    for severity in [1, 2, 3, 4, 5]:
        severity_values = []
        for corr in unique_corruptions:
            value = next((d['value'] for d in corruption_data 
                         if d['corruption'] == corr and d['severity'] == severity), None)
            severity_values.append(value if value is not None else 0)
        
        plt.bar(x_pos + (severity - 3) * width, severity_values, width, 
                label=f'Severity {severity}', alpha=0.8)
    
    plt.axhline(y=baseline_value, color='r', linestyle='--', linewidth=2, label='Baseline')
    
    plt.xlabel('Corruption Type', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f'Robustness Test Results: {metric.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [c.replace('_', ' ').title() for c in unique_corruptions], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def calculate_robustness_summary(results: Dict, metric: str = 'loss') -> Dict:
    """
    Calculate summary statistics for robustness.
    
    Args:
        results: Results dictionary
        metric: Metric to analyze
    
    Returns:
        Summary statistics dictionary
    """
    baseline = results.get('baseline', {})
    if metric == 'loss':
        baseline_value = baseline.get('loss', 0)
    else:
        baseline_value = baseline.get('metrics', {}).get(metric, 0)
    
    all_values = []
    corruption_summaries = {}
    
    for corruption_type in sorted(results.keys()):
        if corruption_type == 'baseline':
            continue
        
        corruption_values = []
        corruption_results = results[corruption_type]
        
        for severity_key in sorted(corruption_results.keys()):
            severity_data = corruption_results[severity_key]
            if metric == 'loss':
                value = severity_data.get('loss', 0)
            else:
                value = severity_data.get('metrics', {}).get(metric, 0)
            
            corruption_values.append(value)
            all_values.append(value)
        
        if corruption_values:
            corruption_summaries[corruption_type] = {
                'mean': np.mean(corruption_values),
                'std': np.std(corruption_values),
                'min': np.min(corruption_values),
                'max': np.max(corruption_values),
                'degradation': np.mean(corruption_values) - baseline_value
            }
    
    summary = {
        'baseline': baseline_value,
        'overall_mean': np.mean(all_values) if all_values else 0,
        'overall_std': np.std(all_values) if all_values else 0,
        'overall_degradation': np.mean(all_values) - baseline_value if all_values else 0,
        'worst_case': np.max(all_values) if all_values else 0,
        'best_case': np.min(all_values) if all_values else 0,
        'by_corruption': corruption_summaries
    }
    
    return summary


def print_robustness_summary(results: Dict, metric: str = 'loss'):
    """
    Print a summary of robustness test results.
    
    Args:
        results: Results dictionary
        metric: Metric to summarize
    """
    summary = calculate_robustness_summary(results, metric)
    
    print("\n" + "="*80)
    print(f"ROBUSTNESS SUMMARY: {metric.upper()}")
    print("="*80)
    print(f"\nBaseline: {summary['baseline']:.4f}")
    print(f"Overall Mean: {summary['overall_mean']:.4f} ± {summary['overall_std']:.4f}")
    print(f"Overall Degradation: {summary['overall_degradation']:.4f}")
    print(f"Worst Case: {summary['worst_case']:.4f}")
    print(f"Best Case: {summary['best_case']:.4f}")
    
    print("\nBy Corruption Type:")
    for corr_type, stats in summary['by_corruption'].items():
        print(f"\n  {corr_type.replace('_', ' ').title()}:")
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    Degradation: {stats['degradation']:.4f}")


def create_corrupted_dataset(dataset, corruption_pipeline: CorruptionPipeline):
    """
    Create a dataset wrapper that applies corruptions.
    
    Args:
        dataset: Original dataset
        corruption_pipeline: Corruption pipeline to apply
    
    Returns:
        Wrapped dataset with corruptions
    """
    class CorruptedDataset:
        def __init__(self, base_dataset, pipeline):
            self.base_dataset = base_dataset
            self.pipeline = pipeline
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            sample = self.base_dataset[idx]
            return sample
    
    return CorruptedDataset(dataset, corruption_pipeline)

