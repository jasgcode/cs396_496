#!/usr/bin/env python3
"""
Training module for Chord Classification.

This module contains the training logic for the chord classifier.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from classifier.chord_head import ChordLinearProbe
from serve.classsify import CHORD_CLASSES


class ChordEmbeddingDataset(Dataset):
    """Dataset for chord classification from pre-extracted embeddings."""
    
    def __init__(self, embeddings_paths: dict):
        """
        Args:
            embeddings_paths: dict mapping chord_name -> path to .pt file
                e.g., {"G": "embeddings/G.pt", "Am": "embeddings/Am.pt"}
        """
        self.samples = []  # List of (embeddings_tensor, label)
        
        for chord_name, emb_path in embeddings_paths.items():
            if chord_name not in CHORD_CLASSES:
                print(f"Warning: {chord_name} not in CHORD_CLASSES, skipping...")
                continue
            
            label = CHORD_CLASSES[chord_name]
            embeddings = torch.load(emb_path, map_location='cpu')
            
            # Each video becomes one sample (all frames = 1 sample)
            self.samples.append((embeddings, label))
            print(f"Loaded {chord_name}: {embeddings.shape[0]} frames")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        embeddings, label = self.samples[idx]
        return embeddings.unsqueeze(0), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Custom collate to handle variable-length sequences."""
    embeddings_list = []
    labels = []
    
    for emb, label in batch:
        embeddings_list.append(emb.squeeze(0))
        labels.append(label)
    
    # Pad sequences to the same length
    max_len = max(e.shape[0] for e in embeddings_list)
    padded = []
    for emb in embeddings_list:
        if emb.shape[0] < max_len:
            pad = torch.zeros(max_len - emb.shape[0], emb.shape[1])
            emb = torch.cat([emb, pad], dim=0)
        padded.append(emb)
    
    # Stack: (batch_size, max_frames, 384)
    embeddings = torch.stack(padded, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return embeddings, labels


def train_classifier(
    embeddings_paths: dict,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = 'cuda',
    model_save_path: str = "chord_classifier.pt"
):
    """
    Train a linear probe classifier on chord embeddings.
    
    Args:
        embeddings_paths: dict mapping chord_name -> .pt file path
        epochs: number of training epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
        model_save_path: path to save trained model
    
    Returns:
        trained model
    """
    print("\n" + "="*60)
    print("TRAINING CHORD CLASSIFIER")
    print("="*60)
    
    # Create dataset and dataloader
    dataset = ChordEmbeddingDataset(embeddings_paths)
    
    if len(dataset) == 0:
        print("Error: No valid samples in dataset")
        return None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset),  # Full batch (small dataset)
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    num_classes = len(CHORD_CLASSES)
    model = ChordLinearProbe(input_dim=384, num_classes=num_classes)
    model = model.to(device)
    
    print(f"\nModel: ChordLinearProbe")
    print(f"Input dim: 384, Output classes: {num_classes}")
    print(f"Training samples: {len(dataset)}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"Device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\n--- Training ---")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = 100 * correct / total
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {model_save_path}")
    print(f"Final accuracy: {100*correct/total:.1f}%")
    print(f"{'='*60}")
    
    return model


# === CLI for standalone training ===
if __name__ == "__main__":
    import argparse
    from glob import glob
    
    parser = argparse.ArgumentParser(description="Train Chord Classifier")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings",
                        help="Directory containing .pt embedding files")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--output", type=str, default="chord_classifier.pt",
                        help="Output model path")
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Auto-discover embeddings
    embeddings_dict = {}
    for pt_file in glob(os.path.join(args.embeddings_dir, "*.pt")):
        chord_name = os.path.splitext(os.path.basename(pt_file))[0]
        if chord_name in CHORD_CLASSES:
            embeddings_dict[chord_name] = pt_file
        else:
            print(f"Warning: {chord_name} not in CHORD_CLASSES, skipping...")
    
    if not embeddings_dict:
        print(f"No valid embeddings found in {args.embeddings_dir}/")
        print(f"Expected chord names: {list(CHORD_CLASSES.keys())}")
        sys.exit(1)
    
    print(f"Found {len(embeddings_dict)} chord embeddings: {list(embeddings_dict.keys())}")
    
    train_classifier(
        embeddings_paths=embeddings_dict,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_save_path=args.output
    )