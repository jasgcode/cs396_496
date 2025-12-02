import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from multiprocessing import Pool, cpu_count
import itertools
from tqdm import tqdm
import time

class ChordDataset(Dataset):
    """Dataset for guitar chord images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(dataset_path, test_size=0.2, val_size=0.125):
    """
    Load dataset and split into train/val/test sets.
    Final split: 70% train, 10% val, 20% test
    """
    dataset_path = Path(dataset_path)
    
    chord_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    
    class_to_idx = {folder.name: idx for idx, folder in enumerate(chord_folders)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    all_paths = []
    all_labels = []
    
    for chord_folder in chord_folders:
        chord_name = chord_folder.name
        label = class_to_idx[chord_name]
        
        images = list(chord_folder.glob("*.jpg")) + list(chord_folder.glob("*.png")) + list(chord_folder.glob("*.jpeg"))
        
        for img_path in images:
            all_paths.append(str(img_path))
            all_labels.append(label)
    
    # Split: 80% train+val, 20% test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, stratify=all_labels, random_state=42
    )
    
    # Split train+val: 87.5% train (70% total), 12.5% val (10% total)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size, 
        stratify=train_val_labels, random_state=42
    )
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_to_idx, idx_to_class

def get_transforms(input_size=224):
    """Get data augmentation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def create_model(num_classes, dropout_rate=0.3):
    """Create MobileNetV2 model with custom final layer."""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    
    return model

def train_with_hyperparams(args):
    """
    Train model with specific hyperparameters.
    Designed to be called by multiprocessing.
    """
    (hp_config, train_data, val_data, num_classes, run_id) = args
    
    lr = hp_config['lr']
    epochs = hp_config['epochs']
    batch_size = hp_config['batch_size']
    dropout = hp_config['dropout']
    
    print(f"\n[Run {run_id}] Starting: LR={lr}, Epochs={epochs}, Batch={batch_size}, Dropout={dropout}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set threads for this process
    torch.set_num_threads(max(1, cpu_count() // 4))
    
    # Unpack data
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = ChordDataset(train_paths, train_labels, train_transform)
    val_dataset = ChordDataset(val_paths, val_labels, test_transform)
    
    # Create dataloaders - num_workers=0 to avoid multiprocessing conflicts
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = create_model(num_classes, dropout_rate=dropout)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    epoch_times = []
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Calculate time and ETA
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        
        # Format output
        progress_pct = int(100 * (epoch + 1) / epochs)
        time_str = f"{int(epoch_time)}s"
        
        if epoch == 0:
            # First epoch - no ETA yet
            print(f"[Run {run_id}] Epoch {epoch+1}/{epochs} ({progress_pct}%): "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}% | {time_str}")
        else:
            # Show ETA
            eta_str = f"{int(eta_seconds/60)}m{int(eta_seconds%60)}s" if eta_seconds >= 60 else f"{int(eta_seconds)}s"
            print(f"[Run {run_id}] Epoch {epoch+1}/{epochs} ({progress_pct}%): "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}% | {time_str} | ETA {eta_str}")
    
    result = {
        'config': hp_config,
        'run_id': run_id,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_acc
    }
    
    total_time = sum(epoch_times)
    print(f"[Run {run_id}] ✓ Complete: Best Val Acc = {best_val_acc:.2f}% | Total time: {int(total_time/60)}m{int(total_time%60)}s")
    
    return result

def run_coarse_search(dataset_path):
    """Run coarse hyperparameter search."""
    
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Load dataset once
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_to_idx, idx_to_class = load_dataset(dataset_path)
    
    num_classes = len(class_to_idx)
    
    print(f"\nDataset loaded:")
    print(f"  Classes: {list(class_to_idx.keys())}")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images (held out)")
    
    print("\n" + "="*80)
    print("STAGE 1: COARSE HYPERPARAMETER SEARCH")
    print("="*80)
    
    # Coarse grid
    hp_grid = {
        'lr': [0.01,  0.001, 0.0001, 0.00001],
        'epochs': [10, 20, 30],
        'batch_size': [16, 32, 64],
        'dropout': [0.1, 0.2, 0.3, 0.4]
    }
    
    # Generate all combinations
    keys = hp_grid.keys()
    values = hp_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nHyperparameter grid:")
    for key, vals in hp_grid.items():
        print(f"  {key}: {vals}")
    print(f"\nTotal combinations: {len(combinations)}")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for i, config in enumerate(combinations):
        args_list.append((
            config,
            (train_paths, train_labels),
            (val_paths, val_labels),
            num_classes,
            i + 1
        ))
    
    # Run in parallel
    num_processes = min(4, len(combinations))
    
    print(f"\nRunning {len(combinations)} experiments with {num_processes} parallel processes...")
    print("="*80)
    
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(train_with_hyperparams, args_list)
    
    elapsed = time.time() - start_time
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("COARSE SEARCH RESULTS (sorted by validation accuracy)")
    print("="*80)
    
    for i, result in enumerate(results):
        config = result['config']
        print(f"\nRank {i+1}:")
        print(f"  LR: {config['lr']}, Epochs: {config['epochs']}, Batch: {config['batch_size']}, Dropout: {config['dropout']}")
        print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")
        print(f"  Final Train Acc: {result['final_train_acc']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Coarse search completed in {elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Save results
    with open('hyperparameter_search_coarse.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to hyperparameter_search_coarse.json")
    
    return results, (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_to_idx, idx_to_class

def run_fine_search(coarse_results, train_data, val_data, num_classes):
    """Run fine-tuning search around best coarse result."""
    
    best_coarse = coarse_results[0]['config']
    
    print("\n" + "="*80)
    print("STAGE 2: FINE-TUNING HYPERPARAMETER SEARCH")
    print("="*80)
    
    print(f"\nBest from coarse search:")
    print(f"  LR: {best_coarse['lr']}")
    print(f"  Epochs: {best_coarse['epochs']}")
    
    # Determine fine-tuning grid based on coarse results
    if best_coarse['lr'] == 0.001:
        lr_fine = [0.001, 0.0001]
    elif best_coarse['lr'] == 0.00001:
        lr_fine = [0.0001, 0.00001]
    else:
        lr_fine = [0.0001]
    
    if best_coarse['epochs'] == 10:
        epochs_fine = [10, 20]
    elif best_coarse['epochs'] == 30:
        epochs_fine = [20, 30]
    else:
        epochs_fine = [20]
    
    hp_grid = {
        'lr': lr_fine,
        'epochs': epochs_fine,
        'batch_size': [32],
        'dropout': [0.3]
    }
    
    print(f"\nFine-tuning grid:")
    for key, vals in hp_grid.items():
        print(f"  {key}: {vals}")
    
    # Generate combinations
    keys = hp_grid.keys()
    values = hp_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nTotal combinations: {len(combinations)}")
    
    # Prepare arguments
    args_list = []
    for i, config in enumerate(combinations):
        args_list.append((
            config,
            train_data,
            val_data,
            num_classes,
            i + 1
        ))
    
    # Run in parallel
    num_processes = min(4, len(combinations))
    
    print(f"\nRunning {len(combinations)} experiments with {num_processes} parallel processes...")
    print("="*80)
    
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(train_with_hyperparams, args_list)
    
    elapsed = time.time() - start_time
    
    # Sort results
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("FINE-TUNING RESULTS (sorted by validation accuracy)")
    print("="*80)
    
    for i, result in enumerate(results):
        config = result['config']
        print(f"\nRank {i+1}:")
        print(f"  LR: {config['lr']}, Epochs: {config['epochs']}, Batch: {config['batch_size']}, Dropout: {config['dropout']}")
        print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")
        print(f"  Final Train Acc: {result['final_train_acc']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Fine-tuning completed in {elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Save results
    with open('hyperparameter_search_fine.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to hyperparameter_search_fine.json")
    
    return results

def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")

def train_final_model(best_config, train_data, val_data, test_data, class_to_idx, idx_to_class):
    """Train final model with best hyperparameters."""
    
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    print(f"\nBest hyperparameters:")
    for key, val in best_config.items():
        print(f"  {key}: {val}")
    
    num_classes = len(class_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Save class mapping
    with open('class_mapping.json', 'w') as f:
        json.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f, indent=2)
    print("\n✓ Class mapping saved to class_mapping.json")
    
    # Unpack data
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = ChordDataset(train_paths, train_labels, train_transform)
    val_dataset = ChordDataset(val_paths, val_labels, test_transform)
    test_dataset = ChordDataset(test_paths, test_labels, test_transform)
    
    # Create dataloaders (use all cores for final training)
    num_workers = os.cpu_count()
    batch_size = best_config['batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    
    # Create model
    model = create_model(num_classes, dropout_rate=best_config['dropout'])
    model = model.to(device)
    
    print(f"\n✓ Model created (MobileNetV2)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(best_config['epochs']):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{best_config['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{best_config['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': best_config
            }, 'best_chord_model.pth')
            print(f"  ✓ NEW BEST! Saved model (val_acc={val_acc:.2f}%)")
    
    # Plot training history
    plot_training_history(history)
    
    # Test on held-out test set
    print(f"\n{'='*80}")
    print("TESTING ON HELD-OUT TEST SET")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load('best_chord_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy (held-out): {test_acc:.2f}%")
    print(f"\nFiles saved:")
    print(f"  • best_chord_model.pth - Trained model")
    print(f"  • class_mapping.json - Class to index mapping")
    print(f"  • training_history.png - Training curves")
    print(f"{'='*80}")

def main():
    DATASET_PATH = "dataset_sampled"
    
    # Stage 1: Coarse search
    print("\n" + "="*80)
    print("GUITAR CHORD CLASSIFICATION - HYPERPARAMETER SEARCH")
    print("="*80)
    
    coarse_results, train_data, val_data, test_data, class_to_idx, idx_to_class = run_coarse_search(DATASET_PATH)
    
    # Get best from coarse
    best_coarse = coarse_results[0]
    
    print(f"\n{'='*80}")
    print("BEST FROM COARSE SEARCH:")
    print(f"{'='*80}")
    print(f"LR: {best_coarse['config']['lr']}")
    print(f"Epochs: {best_coarse['config']['epochs']}")
    print(f"Val Acc: {best_coarse['best_val_acc']:.2f}%")
    
    # Ask user about fine-tuning
    print("\n" + "="*80)
    response = input("Proceed with fine-tuning search? (y/n): ").strip().lower()
    
    if response == 'y':
        # Run fine-tuning
        fine_results = run_fine_search(coarse_results, train_data, val_data, len(class_to_idx))
        
        # Get best overall
        best_overall = fine_results[0]
        
        print(f"\n{'='*80}")
        print("BEST FROM FINE-TUNING:")
        print(f"{'='*80}")
        print(f"LR: {best_overall['config']['lr']}")
        print(f"Epochs: {best_overall['config']['epochs']}")
        print(f"Val Acc: {best_overall['best_val_acc']:.2f}%")
        
        # Train final model
        train_final_model(best_overall['config'], train_data, val_data, test_data, class_to_idx, idx_to_class)
    else:
        # Train final model with best from coarse
        print("\nSkipping fine-tuning, training final model with best from coarse search...")
        train_final_model(best_coarse['config'], train_data, val_data, test_data, class_to_idx, idx_to_class)

if __name__ == "__main__":
    # This ensures the code only runs once in the main process
    main()