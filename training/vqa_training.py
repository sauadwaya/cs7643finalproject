import os
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from models.vqa_model import VQAModel
import time

# COCO image path loader
def image_path_from_id(image_id, is_train):
    split = "train2014" if is_train else "val2014"
    filename = f"COCO_{split}_{int(image_id):012d}.jpg"
    return os.path.join("data/coco", split, filename)

# VQA Dataset
class VQADataset(Dataset):
    def __init__(self, pkl_file, tokenizer, max_length, is_train=True, limit=None):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        if limit:
            data = data[:limit]

        self.data = data
        self.questions = [d["question"] for d in data]
        self.image_ids = [d["image_id"] for d in data]
        self.answers = [d["answer"] for d in data]

        self.is_train = is_train
        self.max_length = max_length

        # Basic image preprocessing
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_id = item["image_id"]
        img_path = image_path_from_id(img_id, self.is_train)
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)

        question = item["question"]
        attention_mask = item["attention_mask"]
        answer = item["answer"]

        return {
            "image": image,
            "input_ids": question,
            "attention_mask": attention_mask,
            "answer": torch.tensor(answer)
        }


# Collate function
def collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch])
    ids = torch.stack([b["input_ids"] for b in batch])
    masks = torch.stack([b["attention_mask"] for b in batch])
    answers = torch.stack([b["answer"] for b in batch])
    return {
        "image": imgs,
        "input_ids": ids,
        "attention_mask": masks,
        "answer": answers
    }

def evaluate_vqa(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:  # iterate over the dict
            pixel_values = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            preds = logits.argmax(dim=1)

            total_correct += (preds == answers).sum().item()
            total_samples += pixel_values.size(0)

    return total_correct / total_samples if total_samples > 0 else 0

# Training Loop
def train_vqa(config, train_pkl, val_pkl, vocab_pkl,
              epochs=1, batch_size=32, lr=1e-4, limit=None):
    # Create model
    print("Building model...")
    model = VQAModel(config)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    model.freeze_encoders()
    print("Encoders frozen for warm-up")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Datasets
    print("Loading datasets...")
    train_dataset = VQADataset(train_pkl, tokenizer, config["vqa"].question_max_length, is_train=True, limit=limit)
    val_dataset = VQADataset(val_pkl, tokenizer, config["vqa"].question_max_length, is_train=False, limit=limit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    unfreeze_epoch = 6

    # Training
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        if epoch == unfreeze_epoch:
            model.unfreeze_encoders()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print("Encoders unfrozen — training full model")

        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            pixel_values = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            optimizer.zero_grad()

            logits = model(pixel_values, input_ids, attention_mask)
            loss = loss_fn(logits, answers)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pixel_values.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == answers).sum().item()
            total_samples += pixel_values.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_acc = evaluate_vqa(model, val_loader, device)
        print(f"Val Accuracy: {val_acc:.4f}")
        epoch_end = time.time()
        print(f"Time Taken: {epoch_end - epoch_start:.2f} sec ({(epoch_end - epoch_start)/60:.2f} min)")

    return model
