import torch
import torch.nn as nn
from transformers import ViTModel, BertTokenizer, BertModel
import pickle
import os

class VQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load vocab file
        self.answer_vocab = self.load_pickle("data/output/answer_vocab.pkl")
        self.answer_ids = self.answer_vocab["ans_to_idx"]
        self.answer_vocab_size = len(self.answer_ids)

        # Image Encoder (ViT)
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_dim = self.image_encoder.config.hidden_size

        # Question Encoder (BERT)
        self.q_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.question_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.question_dim = self.question_encoder.config.hidden_size

        # Fusion Layer
        fused_dim = self.image_dim + self.question_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Answer classifier
        self.classifier = nn.Linear(1024, self.answer_vocab_size)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_outputs = self.image_encoder(pixel_values=pixel_values)
        img_embeds = img_outputs.pooler_output

        q_outputs = self.question_encoder(input_ids=input_ids, attention_mask=attention_mask)
        q_embeds = q_outputs.pooler_output

        fused = torch.cat([img_embeds, q_embeds], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits

    def load_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
    def freeze_encoders(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.question_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = True
        for p in self.question_encoder.parameters():
            p.requires_grad = True

    def predict(self, pixel_values, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values, input_ids, attention_mask)
            pred_ids = logits.argmax(dim=1).cpu().tolist()
            idx_to_ans = {v: k for k, v in self.answer_ids.items()}
            return [idx_to_ans.get(i, "<UNK>") for i in pred_ids]

