"""
LoRA (Low-Rank Adaptation) implementation for Vision Transformer
Enables parameter-efficient fine-tuning of ViT encoder
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import VisionEncoderDecoderModel, ViTModel
import math

class LoRALinear(nn.Module):
    """LoRA adaptation for linear layers"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA parameters
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        """Forward pass with LoRA adaptation"""
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        return original_output + lora_output * self.scaling

def apply_lora_to_vit(
    model: VisionEncoderDecoderModel,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: List[str] = None
) -> VisionEncoderDecoderModel:
    """
    Apply LoRA to Vision Transformer encoder
    
    Args:
        model: VisionEncoderDecoderModel to modify
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to
    """
    if target_modules is None:
        target_modules = ["query", "value", "dense"]
    
    encoder = model.encoder
    
    # Apply LoRA to attention layers
    for layer_idx, layer in enumerate(encoder.encoder.layer):
        attention = layer.attention.attention
        
        # Apply to specified modules
        if "query" in target_modules:
            attention.query = LoRALinear(attention.query, rank, alpha, dropout)
        if "key" in target_modules:
            attention.key = LoRALinear(attention.key, rank, alpha, dropout)
        if "value" in target_modules:
            attention.value = LoRALinear(attention.value, rank, alpha, dropout)
            
        # Apply to output dense layer if specified
        if "dense" in target_modules:
            layer.attention.output.dense = LoRALinear(
                layer.attention.output.dense, rank, alpha, dropout
            )
            
        # Apply to feed-forward layers if specified
        if "intermediate" in target_modules:
            layer.intermediate.dense = LoRALinear(
                layer.intermediate.dense, rank, alpha, dropout
            )
        if "output" in target_modules:
            layer.output.dense = LoRALinear(
                layer.output.dense, rank, alpha, dropout
            )
    
    return model

def get_lora_parameters(model: VisionEncoderDecoderModel) -> List[nn.Parameter]:
    """Get only LoRA parameters for optimization"""
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A.weight, module.lora_B.weight])
    return lora_params

def count_lora_parameters(model: VisionEncoderDecoderModel) -> int:
    """Count number of trainable LoRA parameters"""
    return sum(p.numel() for p in get_lora_parameters(model) if p.requires_grad)

def freeze_non_lora_parameters(model: VisionEncoderDecoderModel):
    """Freeze all parameters except LoRA parameters"""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

class LoRAVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    """Vision Encoder Decoder Model with LoRA support"""
    
    def __init__(self, config=None, encoder=None, decoder=None):
        super().__init__(config, encoder, decoder)
        self.lora_config = None
        
    @classmethod
    def from_encoder_decoder_pretrained_with_lora(
        cls,
        encoder_pretrained_model_name_or_path: str,
        decoder_pretrained_model_name_or_path: str,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        **kwargs
    ):
        """Create model with LoRA from pretrained encoder/decoder"""
        
        # Create base model
        model = cls.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path,
            decoder_pretrained_model_name_or_path,
            **kwargs
        )
        
        # Apply LoRA
        model = apply_lora_to_vit(
            model, lora_rank, lora_alpha, lora_dropout, lora_target_modules
        )
        
        # Store LoRA config
        model.lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": lora_target_modules or ["query", "value", "dense"]
        }
        
        # Freeze non-LoRA parameters
        freeze_non_lora_parameters(model)
        
        return model
        
    def get_lora_state_dict(self):
        """Get state dict containing only LoRA parameters"""
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_state_dict[name] = param
        return lora_state_dict
        
    def load_lora_state_dict(self, state_dict):
        """Load LoRA parameters from state dict"""
        for name, param in state_dict.items():
            if "lora_" in name:
                self.state_dict()[name].copy_(param)

# Example usage and testing functions
def test_lora_implementation():
    """Test LoRA implementation with a small model"""
    from transformers import VisionEncoderDecoderModel
    
    # Create small test model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k",
        "gpt2"
    )
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original trainable parameters: {original_params:,}")
    
    # Apply LoRA
    model = apply_lora_to_vit(model, rank=8, target_modules=["query", "value"])
    freeze_non_lora_parameters(model)
    
    # Count LoRA parameters
    lora_params = count_lora_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Parameter reduction: {(1 - trainable_params/original_params)*100:.2f}%")
    
    return model

if __name__ == "__main__":
    test_lora_implementation()