"""
Multi-dataset loader for Flickr8k, Flickr30k, and COCO
Team Member 3's Week 1 starter code
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import numpy as np
from transformers import ViTImageProcessor, AutoTokenizer

class FlickrDataset(Dataset):
    """Enhanced dataset class supporting both Flickr8k and Flickr30k"""
    
    def __init__(
        self,
        data_dict: Dict[str, List[str]],
        img_dir: str,
        tokenizer: AutoTokenizer,
        img_processor: ViTImageProcessor,
        max_length: int = 48,
        dataset_name: str = "flickr8k"
    ):
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.img_dir = img_dir
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.data = []
        
        # Expand all image-caption pairs
        for filename, captions in data_dict.items():
            for caption in captions:
                self.data.append((filename, caption))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, caption = self.data[idx]
        
        # Load and process image
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.img_processor(image, return_tensors='pt').pixel_values.squeeze(0)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy image if loading fails
            pixel_values = torch.zeros((3, 224, 224))
        
        # Process caption
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        
        # Create labels (copy of input_ids with padding tokens set to -100)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'filename': filename,
            'caption': caption,
            'dataset': self.dataset_name
        }

class COCODataset(Dataset):
    """COCO dataset loader for captions"""
    
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        tokenizer: AutoTokenizer,
        img_processor: ViTImageProcessor,
        max_length: int = 48,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.img_dir = img_dir
        self.max_length = max_length
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        self.id_to_filename = {}
        for img_info in coco_data['images']:
            self.id_to_filename[img_info['id']] = img_info['file_name']
        
        # Extract image-caption pairs
        self.data = []
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.id_to_filename:
                filename = self.id_to_filename[img_id]
                caption = ann['caption']
                self.data.append((filename, caption))
        
        # Limit samples if specified
        if max_samples:
            self.data = self.data[:max_samples]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, caption = self.data[idx]
        
        # Load and process image
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.img_processor(image, return_tensors='pt').pixel_values.squeeze(0)
        except Exception as e:
            print(f"Error loading COCO image {img_path}: {e}")
            pixel_values = torch.zeros((3, 224, 224))
        
        # Process caption
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'filename': filename,
            'caption': caption,
            'dataset': 'coco'
        }

class MultiDatasetLoader:
    """Unified loader for multiple captioning datasets"""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        img_processor: ViTImageProcessor,
        max_length: int = 48
    ):
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.max_length = max_length
        self.datasets = {}
    
    def add_flickr8k(
        self,
        data_dir: str,
        img_dir: str,
        splits: List[str] = ['train', 'dev', 'test']
    ):
        """Add Flickr8k dataset"""
        for split in splits:
            pickle_file = os.path.join(data_dir, f'{split}_data.pickle')
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    data_dict = pickle.load(f)
                
                dataset = FlickrDataset(
                    data_dict, img_dir, self.tokenizer, 
                    self.img_processor, self.max_length, "flickr8k"
                )
                self.datasets[f'flickr8k_{split}'] = dataset
                print(f"Added Flickr8k {split}: {len(dataset)} samples")
    
    def add_flickr30k(
        self,
        captions_file: str,
        img_dir: str,
        train_file: str,
        val_file: str,
        test_file: str
    ):
        """Add Flickr30k dataset"""
        # Load captions
        captions_dict = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    img_caption, caption = line.strip().split('\t', 1)
                    img_name = img_caption.split('#')[0]
                    
                    if img_name not in captions_dict:
                        captions_dict[img_name] = []
                    captions_dict[img_name].append(caption)
        
        # Load splits
        splits_files = {
            'train': train_file,
            'dev': val_file,
            'test': test_file
        }
        
        for split, split_file in splits_files.items():
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    img_names = [line.strip() for line in f]
                
                split_data = {img: captions_dict.get(img, []) for img in img_names if img in captions_dict}
                
                dataset = FlickrDataset(
                    split_data, img_dir, self.tokenizer,
                    self.img_processor, self.max_length, "flickr30k"
                )
                self.datasets[f'flickr30k_{split}'] = dataset
                print(f"Added Flickr30k {split}: {len(dataset)} samples")
    
    def add_coco(
        self,
        train_annotations: str,
        val_annotations: str,
        train_img_dir: str,
        val_img_dir: str,
        max_samples_per_split: Optional[int] = 5000
    ):
        """Add COCO dataset"""
        if os.path.exists(train_annotations):
            train_dataset = COCODataset(
                train_annotations, train_img_dir, self.tokenizer,
                self.img_processor, self.max_length, max_samples_per_split
            )
            self.datasets['coco_train'] = train_dataset
            print(f"Added COCO train: {len(train_dataset)} samples")
        
        if os.path.exists(val_annotations):
            val_dataset = COCODataset(
                val_annotations, val_img_dir, self.tokenizer,
                self.img_processor, self.max_length, max_samples_per_split//2
            )
            self.datasets['coco_val'] = val_dataset
            print(f"Added COCO val: {len(val_dataset)} samples")
    
    def get_combined_dataset(self, dataset_names: List[str]) -> Dataset:
        """Combine multiple datasets"""
        datasets_to_combine = []
        for name in dataset_names:
            if name in self.datasets:
                datasets_to_combine.append(self.datasets[name])
            else:
                print(f"Warning: Dataset {name} not found")
        
        if not datasets_to_combine:
            raise ValueError("No valid datasets found")
        
        combined = ConcatDataset(datasets_to_combine)
        print(f"Combined dataset size: {len(combined)}")
        return combined
    
    def get_dataloader(
        self,
        dataset_names: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Get DataLoader for specified datasets"""
        combined_dataset = self.get_combined_dataset(dataset_names)
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def list_datasets(self):
        """List all available datasets"""
        print("Available datasets:")
        for name, dataset in self.datasets.items():
            print(f"  {name}: {len(dataset)} samples")

# Testing function for Team Member 3
def test_multi_dataset_loader():
    """Test the multi-dataset loader"""
    from transformers import AutoTokenizer, ViTImageProcessor
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<PAD>', 'bos_token': '<BOS>'})
    img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    # Create loader
    loader = MultiDatasetLoader(tokenizer, img_processor)
    
    # Test with existing Flickr8k data
    flickr8k_data_dir = "./Flicker8k_captions"
    flickr8k_img_dir = "./Flickr8k_Data/Flicker8k_Dataset_resized"
    
    if os.path.exists(flickr8k_data_dir):
        loader.add_flickr8k(flickr8k_data_dir, flickr8k_img_dir)
        loader.list_datasets()
        
        # Test combined dataset
        try:
            combined_dataset = loader.get_combined_dataset(['flickr8k_train'])
            sample = combined_dataset[0]
            print("\nSample data keys:", sample.keys())
            print("Sample caption:", sample['caption'])
            print("Pixel values shape:", sample['pixel_values'].shape)
            return True
        except Exception as e:
            print(f"Error testing dataset: {e}")
            return False
    else:
        print("Flickr8k data not found - please run preprocessing first")
        return False

if __name__ == "__main__":
    success = test_multi_dataset_loader()
    if success:
        print("Multi-dataset loader test passed!")
    else:
        print("Multi-dataset loader test failed!")