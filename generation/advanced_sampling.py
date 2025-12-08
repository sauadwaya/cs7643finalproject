"""
Advanced text generation strategies for image captioning
Team Member 1's Week 1 starter code (second part)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import numpy as np

class AdvancedGenerator:
    """Advanced text generation with multiple strategies"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        tokenizer: AutoTokenizer,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def beam_search_generate(
        self,
        pixel_values: torch.Tensor,
        num_beams: int = 5,
        max_new_tokens: int = 20,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 2
    ) -> List[str]:
        """Generate captions using beam search"""
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False  # Deterministic beam search
            )
        
        # Decode generated sequences
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
    
    def nucleus_sampling_generate(
        self,
        pixel_values: torch.Tensor,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_new_tokens: int = 20,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate captions using nucleus (top-p) sampling"""
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences
            )
        
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
    
    def top_k_sampling_generate(
        self,
        pixel_values: torch.Tensor,
        top_k: int = 50,
        temperature: float = 0.7,
        max_new_tokens: int = 20,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate captions using top-k sampling"""
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences
            )
        
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
    
    def diverse_beam_search_generate(
        self,
        pixel_values: torch.Tensor,
        num_beams: int = 5,
        num_beam_groups: int = 5,
        diversity_penalty: float = 1.0,
        max_new_tokens: int = 20,
        length_penalty: float = 1.0
    ) -> List[str]:
        """Generate diverse captions using diverse beam search"""
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                length_penalty=length_penalty,
                early_stopping=True,
                do_sample=False
            )
        
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
    
    def contrastive_search_generate(
        self,
        pixel_values: torch.Tensor,
        top_k: int = 4,
        penalty_alpha: float = 0.6,
        max_new_tokens: int = 20
    ) -> List[str]:
        """Generate captions using contrastive search"""
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                penalty_alpha=penalty_alpha,
                top_k=top_k,
                do_sample=False
            )
        
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions

class GenerationComparator:
    """Compare different generation strategies"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        tokenizer: AutoTokenizer,
        device: torch.device
    ):
        self.generator = AdvancedGenerator(model, tokenizer, device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def compare_strategies(
        self,
        pixel_values: torch.Tensor,
        strategies: List[str] = None
    ) -> Dict[str, List[str]]:
        """Compare multiple generation strategies on the same image"""
        
        if strategies is None:
            strategies = ['beam_search', 'nucleus_sampling', 'top_k_sampling']
        
        results = {}
        
        for strategy in strategies:
            if strategy == 'beam_search':
                captions = self.generator.beam_search_generate(pixel_values, num_beams=5)
                results['beam_search'] = captions
                
            elif strategy == 'nucleus_sampling':
                captions = self.generator.nucleus_sampling_generate(
                    pixel_values, top_p=0.9, temperature=0.7, num_return_sequences=3
                )
                results['nucleus_sampling'] = captions
                
            elif strategy == 'top_k_sampling':
                captions = self.generator.top_k_sampling_generate(
                    pixel_values, top_k=50, temperature=0.7, num_return_sequences=3
                )
                results['top_k_sampling'] = captions
                
            elif strategy == 'diverse_beam_search':
                captions = self.generator.diverse_beam_search_generate(
                    pixel_values, num_beams=5, num_beam_groups=5
                )
                results['diverse_beam_search'] = captions
                
            elif strategy == 'contrastive_search':
                captions = self.generator.contrastive_search_generate(pixel_values)
                results['contrastive_search'] = captions
        
        return results
    
    def parameter_sweep(
        self,
        pixel_values: torch.Tensor,
        strategy: str = 'beam_search',
        param_ranges: Dict = None
    ) -> Dict:
        """Perform parameter sweep for a given strategy"""
        
        if param_ranges is None:
            if strategy == 'beam_search':
                param_ranges = {
                    'num_beams': [3, 5, 7, 10],
                    'length_penalty': [0.8, 1.0, 1.2, 1.5]
                }
            elif strategy == 'nucleus_sampling':
                param_ranges = {
                    'top_p': [0.7, 0.8, 0.9, 0.95],
                    'temperature': [0.5, 0.7, 0.9, 1.1]
                }
        
        results = {}
        
        if strategy == 'beam_search':
            for num_beams in param_ranges.get('num_beams', [5]):
                for length_penalty in param_ranges.get('length_penalty', [1.0]):
                    key = f"beams_{num_beams}_penalty_{length_penalty}"
                    captions = self.generator.beam_search_generate(
                        pixel_values, num_beams=num_beams, length_penalty=length_penalty
                    )
                    results[key] = captions
        
        elif strategy == 'nucleus_sampling':
            for top_p in param_ranges.get('top_p', [0.9]):
                for temperature in param_ranges.get('temperature', [0.7]):
                    key = f"top_p_{top_p}_temp_{temperature}"
                    captions = self.generator.nucleus_sampling_generate(
                        pixel_values, top_p=top_p, temperature=temperature
                    )
                    results[key] = captions
        
        return results

def test_generation_strategies():
    """Test function for Team Member 1"""
    from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
    from PIL import Image
    import os
    
    # Load model and tokenizer
    model_path = "./image-captioning-model/epoch_decoder_only_baseline_3"
    if not os.path.exists(model_path):
        print("Model not found. Please train a model first.")
        return False
    
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '<PAD>', 'bos_token': '<BOS>'})
        img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Load test image
        test_image_path = "./Flickr8k_Data/Flicker8k_Dataset/667626_18933d713e.jpg"
        if not os.path.exists(test_image_path):
            print("Test image not found")
            return False
        
        image = Image.open(test_image_path).convert("RGB")
        pixel_values = img_processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Test generation strategies
        comparator = GenerationComparator(model, tokenizer, device)
        
        print("Testing generation strategies...")
        results = comparator.compare_strategies(pixel_values)
        
        for strategy, captions in results.items():
            print(f"\n{strategy.upper()}:")
            for i, caption in enumerate(captions):
                print(f"  {i+1}: {caption}")
        
        # Test parameter sweep
        print("\nTesting parameter sweep for beam search...")
        sweep_results = comparator.parameter_sweep(pixel_values, 'beam_search')
        
        for config, captions in list(sweep_results.items())[:3]:  # Show first 3
            print(f"\n{config}: {captions[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error in testing: {e}")
        return False

if __name__ == "__main__":
    success = test_generation_strategies()
    if success:
        print("\nGeneration strategies test passed!")
    else:
        print("\nGeneration strategies test failed!")