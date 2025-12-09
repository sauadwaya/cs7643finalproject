"""
Baseline Robustness Experiments
Run robustness tests on baseline models.
Can be run independently or after model evaluation is complete.
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from typing import Optional, Dict, List

from robustness import (
    evaluate_with_corruptions,
    save_results,
    print_corruption_results,
    print_robustness_summary,
    plot_corruption_results,
    create_robustness_corruption_pipeline
)
from data.multi_dataset_loader import create_multi_dataset


def run_baseline_robustness_experiments(
    model_path: str,
    dataset_name: str = 'flickr8k',
    split: str = 'test',
    output_dir: str = './robustness_results',
    corruption_types: Optional[List[str]] = None,
    severity_levels: List[int] = [1, 2, 3, 4, 5],
    batch_size: int = 32,
    max_len: int = 48,
    device: Optional[str] = None,
    base_dir: str = './'
):
    """
    Run baseline robustness experiments.
    
    Args:
        model_path: Path to trained model directory
        dataset_name: Dataset to test on ('flickr8k' or 'flickr30k')
        split: Data split to use ('test' or 'dev')
        output_dir: Directory to save results
        corruption_types: List of corruption types to test (None = all)
        severity_levels: List of severity levels to test
        batch_size: Batch size for evaluation
        max_len: Maximum sequence length
        device: Device to use ('cuda' or 'cpu', None = auto-detect)
        base_dir: Base directory for data
    
    Returns:
        Dictionary with results
    """
    print("="*80)
    print("BASELINE ROBUSTNESS EXPERIMENTS")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name} ({split} split)")
    print(f"Corruptions: {corruption_types if corruption_types else 'All'}")
    print(f"Severity Levels: {severity_levels}")
    print("="*80)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"\nLoading model from {model_path}...")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Fallback to default tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Using default GPT-2 tokenizer")
    
    # Setup image processor
    img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    # Create baseline dataset (no corruption)
    print(f"\nCreating baseline dataset ({dataset_name}, {split})...")
    baseline_dataset = create_multi_dataset(
        datasets=[dataset_name],
        split=split,
        tokenizer=tokenizer,
        img_processor=img_processor,
        max_len=max_len,
        base_dir=base_dir,
        use_resized=True
    )
    baseline_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False)
    print(f"Baseline dataset: {len(baseline_dataset)} samples")
    
    corruption_pipeline = create_robustness_corruption_pipeline(severity_levels=severity_levels)
    
    # Setup checkpoint file for resuming
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{dataset_name}_{split}.json')
    
    print("\n" + "="*80)
    print("Running robustness experiments...")
    print("="*80)
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found: {checkpoint_file}")
        print("Will resume from last saved progress...")
    else:
        print(f"Starting new experiment. Checkpoint will be saved to: {checkpoint_file}")
    print("="*80)
    
    results = evaluate_with_corruptions(
        model=model,
        dataloader=baseline_loader,
        tokenizer=tokenizer,
        device=device,
        corruption_pipeline=corruption_pipeline,
        corruption_types=corruption_types,
        severity_levels=severity_levels,
        img_processor=img_processor,
        checkpoint_file=checkpoint_file,
        dataset_name=dataset_name,
        split=split,
        base_dir=base_dir,
        max_len=max_len,
        batch_size=batch_size
    )
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print_corruption_results(results, metrics_to_show=['loss', 'bleu_1', 'bleu_2', 'bleu_4'])
    
    print_robustness_summary(results, metric='loss')
    print_robustness_summary(results, metric='bleu_1')
    print_robustness_summary(results, metric='bleu_4')
    
    results_file = os.path.join(output_dir, f'baseline_robustness_{dataset_name}_{split}.json')
    save_results(results, results_file, format='json')
    print(f"\nFinal results saved to {results_file}")
    print(f"Checkpoint file: {checkpoint_file}")
    
    print("\nCreating visualizations...")
    plot_corruption_results(
        results, 
        metric='loss',
        save_path=os.path.join(output_dir, f'robustness_loss_{dataset_name}_{split}.png')
    )
    plot_corruption_results(
        results,
        metric='bleu_4',
        save_path=os.path.join(output_dir, f'robustness_bleu4_{dataset_name}_{split}.png')
    )
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    
    return results


def run_multiple_baseline_experiments(
    model_paths: Dict[str, str],
    dataset_name: str = 'flickr8k',
    split: str = 'test',
    output_dir: str = './robustness_results',
    **kwargs
):
    """
    Run robustness experiments on multiple baseline models.
    
    Args:
        model_paths: Dictionary mapping model names to model paths
        dataset_name: Dataset to test on
        split: Data split to use
        output_dir: Directory to save results
        **kwargs: Additional arguments for run_baseline_robustness_experiments
    
    Returns:
        Dictionary mapping model names to results
    """
    all_results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n{'='*80}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*80}\n")
        
        model_output_dir = os.path.join(output_dir, model_name)
        results = run_baseline_robustness_experiments(
            model_path=model_path,
            dataset_name=dataset_name,
            split=split,
            output_dir=model_output_dir,
            **kwargs
        )
        
        all_results[model_name] = results
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model_name, results in all_results.items():
        baseline = results.get('baseline', {})
        baseline_loss = baseline.get('loss', 0)
        baseline_bleu = baseline.get('metrics', {}).get('bleu_4', 0)
        print(f"\n{model_name}:")
        print(f"  Baseline Loss: {baseline_loss:.4f}")
        print(f"  Baseline BLEU-4: {baseline_bleu:.4f}")
    
    return all_results


if __name__ == "__main__":
    results = run_baseline_robustness_experiments(
        model_path='./image-captioning-model/epoch_8',
        dataset_name='flickr8k',
        split='test',
        output_dir='./robustness_results/baseline',
        corruption_types=None,
        severity_levels=[1, 2, 3, 4, 5]
    )

