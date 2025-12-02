import torch
import torch.nn as nn

class ChordLinearProbe(nn.Module):
    def __init__(self, input_dim=384, num_classes=7):
        super().__init__()
        
        # Define the specific classes for clarity/debugging
        self.class_names = [
            "E major", "E minor", 
            "A major", "A minor", 
            "C major", "G major", "D major"
        ]
        
        # 1. Normalization (Optional but helps training stability)
        self.norm = nn.LayerNorm(input_dim)
        
        # 2. The Linear Classifier
        # Maps 384 dimensions -> 7 chord probabilities
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch_Size, Num_Frames, Feature_Dim)
        
        # --- MEAN POOLING ---
        # Squash the frame dimension (dim=1) by taking the average.
        # This creates one "summary" embedding for the whole video.
        # Shape becomes: (Batch_Size, Feature_Dim)
        video_summary = x.mean(dim=1)
        
        # Normalize
        video_summary = self.norm(video_summary)
        
        # Classify
        logits = self.classifier(video_summary)
        
        return logits