"""
Visualization tools for image captioning model analysis
Includes Grad-CAM, attention visualization, and interpretability tools
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class GradCAMVisualizer:
    """Grad-CAM visualization for ViT encoder in image captioning"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        tokenizer: AutoTokenizer,
        img_processor: ViTImageProcessor,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.setup_hooks()
    
    def setup_hooks(self):
        """Setup forward and backward hooks for gradient capture"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Hook to the last encoder layer
        target_layer = self.model.encoder.encoder.layer[-1]
        target_layer.register_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)
    
    def generate_gradcam(
        self,
        image: Image.Image,
        target_caption: Optional[str] = None,
        target_word: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for image captioning
        
        Args:
            image: Input PIL image
            target_caption: Specific caption to compute gradients for
            target_word: Specific word in caption to focus on
        """
        self.model.eval()
        
        # Process image
        pixel_values = self.img_processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        if target_caption is None:
            # Generate caption and use it as target
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=20,
                    do_sample=False,
                    decoder_start_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            target_caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Tokenize target caption
        caption_tokens = self.tokenizer(
            target_caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=48
        ).to(self.device)
        
        # Forward pass
        pixel_values.requires_grad_(True)
        outputs = self.model(
            pixel_values=pixel_values,
            labels=caption_tokens.input_ids,
            decoder_attention_mask=caption_tokens.attention_mask
        )
        
        # Compute loss (or focus on specific word)
        if target_word:
            # Find target word in caption and focus loss on it
            word_ids = self.tokenizer.encode(target_word)
            if word_ids:
                # Simplified: use overall loss (can be refined to focus on specific tokens)
                loss = outputs.loss
        else:
            loss = outputs.loss
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate Grad-CAM
        gradients = self.gradients[0]  # [seq_len, hidden_dim]
        activations = self.activations[0]  # [seq_len, hidden_dim]
        
        # Pool gradients across hidden dimension
        weights = torch.mean(gradients, dim=-1)  # [seq_len]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights.unsqueeze(-1) * activations, dim=0)  # [hidden_dim]
        
        # Convert to spatial map (ViT patches)
        # ViT has [CLS] + patch tokens, so remove CLS token
        spatial_cam = cam[1:]  # Remove CLS token
        
        # Reshape to spatial grid (14x14 for 224x224 input with 16x16 patches)
        grid_size = int(np.sqrt(len(spatial_cam)))
        spatial_cam = spatial_cam.reshape(grid_size, grid_size).cpu().numpy()
        
        # Apply ReLU and normalize
        spatial_cam = np.maximum(spatial_cam, 0)
        spatial_cam = spatial_cam / np.max(spatial_cam) if np.max(spatial_cam) > 0 else spatial_cam
        
        return spatial_cam, target_caption
    
    def visualize_gradcam(
        self,
        image: Image.Image,
        save_path: Optional[str] = None,
        target_word: Optional[str] = None
    ) -> plt.Figure:
        """Create Grad-CAM visualization"""
        
        # Generate Grad-CAM
        cam, caption = self.generate_gradcam(image, target_word=target_word)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        # Overlay
        img_array = np.array(image.resize((224, 224)))
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Create heatmap overlay
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.6 * img_array/255.0 + 0.4 * heatmap
        
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay")
        axes[2].axis('off')
        
        plt.suptitle(f"Generated Caption: '{caption}'", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class AttentionVisualizer:
    """Visualize cross-attention between image patches and text tokens"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        tokenizer: AutoTokenizer,
        img_processor: ViTImageProcessor,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.device = device
    
    def extract_cross_attention(
        self,
        image: Image.Image,
        caption: str
    ) -> Tuple[torch.Tensor, List[str], List[str]]:
        """Extract cross-attention weights between image patches and text tokens"""
        
        self.model.eval()
        
        # Process inputs
        pixel_values = self.img_processor(image, return_tensors="pt").pixel_values.to(self.device)
        caption_tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=48
        ).to(self.device)
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                decoder_input_ids=caption_tokens.input_ids,
                decoder_attention_mask=caption_tokens.attention_mask,
                output_attentions=True
            )
        
        # Extract cross-attention from decoder
        cross_attentions = outputs.cross_attentions  # List of attention tensors
        
        # Use attention from last layer, average across heads
        last_cross_attention = cross_attentions[-1]  # [batch, heads, seq_len, encoder_seq_len]
        avg_attention = torch.mean(last_cross_attention[0], dim=0)  # [seq_len, encoder_seq_len]
        
        # Get token names
        text_tokens = self.tokenizer.convert_ids_to_tokens(caption_tokens.input_ids[0])
        # Image patches (196 patches + 1 CLS token for ViT)
        patch_names = ['CLS'] + [f'Patch_{i}' for i in range(avg_attention.shape[1]-1)]
        
        return avg_attention.cpu(), text_tokens, patch_names
    
    def visualize_cross_attention(
        self,
        image: Image.Image,
        caption: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize cross-attention matrix"""
        
        attention_weights, text_tokens, patch_names = self.extract_cross_attention(image, caption)
        
        # Create heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Show original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Show attention heatmap
        sns.heatmap(
            attention_weights.numpy(),
            xticklabels=patch_names[::10],  # Show every 10th patch
            yticklabels=text_tokens,
            cmap='Blues',
            ax=axes[1],
            cbar=True
        )
        axes[1].set_title("Cross-Attention: Text Tokens → Image Patches")
        axes[1].set_xlabel("Image Patches")
        axes[1].set_ylabel("Text Tokens")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class AttentionRollout:
    """Attention rollout visualization for ViT"""
    
    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        img_processor: ViTImageProcessor,
        device: torch.device
    ):
        self.model = model
        self.img_processor = img_processor
        self.device = device
    
    def compute_rollout(
        self,
        image: Image.Image,
        discard_ratio: float = 0.9
    ) -> np.ndarray:
        """Compute attention rollout for ViT encoder"""
        
        self.model.eval()
        pixel_values = self.img_processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            # Get encoder outputs with attention
            encoder_outputs = self.model.encoder(
                pixel_values,
                output_attentions=True
            )
        
        attentions = encoder_outputs.attentions  # List of attention tensors
        
        # Attention rollout computation
        result = torch.eye(attentions[0].size(-1)).to(self.device)
        
        for attention in attentions:
            # Average across heads: [batch, heads, seq, seq] -> [seq, seq]
            attention_heads_fused = attention.mean(axis=1)[0]
            
            # Apply discard ratio to keep only top attention
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            flat[0, indices] = 0
            
            # Add identity matrix
            I = torch.eye(attention_heads_fused.size(-1)).to(self.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(a, result)
        
        # Look at the total attention between the class token and the patch tokens
        mask = result[0, 0, 1:]  # Remove CLS token
        
        # Reshape to spatial grid
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).cpu().numpy()
        
        # Normalize
        mask = mask / np.max(mask)
        
        return mask
    
    def visualize_rollout(
        self,
        image: Image.Image,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention rollout"""
        
        rollout = self.compute_rollout(image)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Attention rollout
        axes[1].imshow(rollout, cmap='jet')
        axes[1].set_title("Attention Rollout")
        axes[1].axis('off')
        
        # Overlay
        img_array = np.array(image.resize((224, 224)))
        rollout_resized = cv2.resize(rollout, (224, 224))
        heatmap = plt.cm.jet(rollout_resized)[:, :, :3]
        overlay = 0.6 * img_array/255.0 + 0.4 * heatmap
        
        axes[2].imshow(overlay)
        axes[2].set_title("Rollout Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Testing function
def test_visualizations():
    """Test visualization tools"""
    from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
    import os
    
    # Load model
    model_path = "./image-captioning-model/epoch_decoder_only_baseline_3"
    if not os.path.exists(model_path):
        print("Model not found")
        return False
    
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '<PAD>', 'bos_token': '<BOS>'})
        img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load test image
        test_image_path = "./Flickr8k_Data/Flicker8k_Dataset/667626_18933d713e.jpg"
        if not os.path.exists(test_image_path):
            print("Test image not found")
            return False
        
        image = Image.open(test_image_path).convert("RGB")
        
        print("Testing Grad-CAM...")
        gradcam_viz = GradCAMVisualizer(model, tokenizer, img_processor, device)
        fig1 = gradcam_viz.visualize_gradcam(image)
        print("Grad-CAM visualization created")
        
        print("Testing Attention Rollout...")
        rollout_viz = AttentionRollout(model, img_processor, device)
        fig2 = rollout_viz.visualize_rollout(image)
        print("Attention rollout visualization created")
        
        # Show plots
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_visualizations()
    if success:
        print("Visualization tests passed!")
    else:
        print("Visualization tests failed!")