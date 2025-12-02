import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm # For progress bars

class DINOv3FeatureExtractor:
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", device='cuda'):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading DINOv3 model: {model_name} on {self.device}...")
        
        # Load Processor (handles resizing and normalization)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Load Model
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to evaluation mode
        print("DINOv3 loaded.")

    def extract_features(self, image_list, batch_size=16):
        """
        Extracts the [CLS] token embedding for a list of PIL images.
        Returns a torch tensor of shape (num_frames, embedding_dim).
        """
        all_embeddings = []
        
        # Process in batches to manage VRAM
        print(f"Extracting features from {len(image_list)} frames...")
        for i in tqdm(range(0, len(image_list), batch_size)):
            batch_images = image_list[i : i + batch_size]
            
            # 1. Preprocess inputs (Resizing, Normalization)
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 2. Forward pass (No Gradients needed for inference)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 3. Get the CLS token (Global representation of the frame)
            # DINOv3/v2 output usually has 'last_hidden_state'
            # Shape: (batch, num_patches + 1, dim). Index 0 is CLS.
            cls_token = outputs.last_hidden_state[:, 0, :]
            
            # Move to CPU to free GPU memory and append
            all_embeddings.append(cls_token.cpu())
            
        # Concatenate all batches
        final_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Feature extraction complete. Shape: {final_embeddings.shape}")
        return final_embeddings