##  Experiments and Analysis

This repository contains the Jupyter notebooks used for **LoRA** (Low-Rank Adaptation) fine-tuning experiments and subsequent **GradCAM-inspired analysis** on a captioning task.

---

### 1. LoRA Experiments

This notebook contains the primary ablation study for LoRA fine-tuning on the captioning task.

| File Name | Description |
| :--- | :--- |
| **`LoRAAblations_Captioning_Final.ipynb`** | Final notebook for the LoRA ablation study on the captioning dataset. |

---

### 2. GradCAM-Inspired Analysis

These notebooks explore the model's focus using a **GradCAM-inspired technique** across various LoRA training setups. The files are organized by the training epochs and the LoRA rank ($r$) used during fine-tuning.

| File Name | LoRA Rank ($r$) | Epochs Trained | Purpose |
| :--- | :--- | :--- | :--- |
| **`GradCAM_2_epochs.ipynb`** | 8 | 2 | Analysis of a **short-trained** model (2 epochs). |
| **`GradCAM_8_epochs.ipynb`** | 8 | 8 | Analysis of a model trained for **8 epochs** with $r=8$. |
| **`GradCAM_8_16.ipynb`** | 16 | 8 | Analysis of a model trained for 8 epochs with a **higher rank** ($r=16$). |
| **`GradCAM_25.ipynb`** | 8 | 25 | Analysis of an **over-trained** model (25 epochs). |

### 3. VQA Analysis
The following files were used for VQA modeling, training, and graphing

| File Name | Purpose |
| :--- | :--- |
| **`vqa_preprocessing.py`** | Preprocessed the training and validation data from COCO into pkl files for ease of use in modeling and training |
| **`vqa_model.py`** | The basic VQA model with ViT and BERT |
| **`vqa_training.py`** | The training and evaluation of the VQA model |
| **`vqa_dev.ipynb`** | Runs the training and evaluation along with graphing and imaging |

---

## 4. Robustness Testing and Encoder Training

### 4.1 Robustness Module

| Folder | Purpose |
| :--- | :--- |
| **`robustness/`** | Python module for testing model robustness to different types of image corruptions. |

### 4.2 Encoder Training

| File Name | Description |
| :--- | :--- |
| **`train_encoder_v2.ipynb`** | Baseline encoder training (5 epochs) |
| **`train_encoder_v3.ipynb`** | Improved encoder training with learning rate scheduler and mixed precision |

### 4.3 Robustness Testing

| File Name | Description |
| :--- | :--- |
| **`test_robustness_baseline.ipynb`** | Tests the baseline model. |
| **`test_robustness_encoder_trained_v2.ipynb`** | Tests the encoder trained with v2 settings. |
| **`test_robustness_encoder_trained_v3.ipynb`** | Tests the encoder trained with v3 settings. |
