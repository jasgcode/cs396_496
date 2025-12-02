#!/usr/bin/env python3
"""
Classification module for Chord Recognition.

This module contains functions for:
- Video classification (full pipeline: SAM → DINO → Classifier)
- Embedding-based classification
- Embedding extraction
- Video downscaling
- Batch processing

For training, see train/train.py
For main CLI, see main.py
"""

import os
import sys
import subprocess
import shutil
from glob import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import gc
from PIL import Image
import torch
from model.dino import DINOv3FeatureExtractor
from model.sam3 import ChordAnalysisPreprocessor
from classifier.chord_head import ChordLinearProbe


# === CHORD CLASSES (13 chords - what we have embeddings for) ===
CHORD_CLASSES = {
    "A": 0, "Am": 1,
    "B": 2, "Bm": 3,
    "C": 4, "Cm": 5,
    "D": 6, "Dm": 7,
    "Em": 8,  
    "F": 9, "Fm": 10,
    "G": 11, "Gm": 12
}
CLASS_NAMES = list(CHORD_CLASSES.keys())


# === VRAM DETECTION & VIDEO DOWNSCALING ===
def get_recommended_resolution():
    """
    Detect available VRAM and return recommended video resolution.
    
    Returns:
        tuple: (width, height, description)
    """
    if not torch.cuda.is_available():
        print("No CUDA available, using conservative 360p")
        return (640, 360, "360p (CPU mode)")
    
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected VRAM: {vram_gb:.1f} GB")
        
        if vram_gb <= 8:
            return (640, 360, "360p (≤8GB VRAM)")
        elif vram_gb <= 16:
            return (854, 480, "480p (≤16GB VRAM)")
        elif vram_gb <= 32:
            return (1280, 720, "720p (≤32GB VRAM)")
        else:
            return (1920, 1080, "1080p (>32GB VRAM)")
    except Exception as e:
        print(f"Error detecting VRAM: {e}, defaulting to 480p")
        return (854, 480, "480p (default)")


def parse_chord_from_filename(filename: str) -> str:
    """
    Extract chord name from video filename.
    
    Examples:
        'A.mov' -> 'A'
        'Am.mov' -> 'Am'
        'Am_0.mp4' -> 'Am' (skip _0 suffix duplicates)
    
    Returns:
        chord name or None if should skip
    """
    base = os.path.splitext(filename)[0]  # Remove extension
    
    # Skip _0, _1, etc. suffixes (duplicates)
    if '_' in base:
        return None
    
    return base


def downscale_video(input_path: str, output_path: str, width: int, height: int, fps: int = 15) -> bool:
    """
    Downscale a video using ffmpeg.
    
    Args:
        input_path: path to input video
        output_path: path to output video (.mp4)
        width: target width
        height: target height  
        fps: target frame rate (default 15 for faster processing)
    
    Returns:
        True if successful, False otherwise
    """
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        print("ERROR: ffmpeg not found. Please install ffmpeg.")
        return False
    
    # Build ffmpeg command
    # -y: overwrite output
    # -i: input file
    # -vf scale: resize video
    # -r: frame rate
    # -c:v libx264: H.264 codec
    # -preset fast: balance speed/quality
    # -crf 23: quality (lower = better, 23 is default)
    # -an: remove audio (not needed for chord classification)
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-r', str(fps),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an',  # No audio
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Timeout processing {input_path}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def downscale_videos(input_dir: str, output_dir: str, resolution: tuple = None, fps: int = 15):
    """
    Downscale all videos in a directory.
    
    Args:
        input_dir: directory containing source videos
        output_dir: directory to save downscaled videos
        resolution: (width, height, description) or None to auto-detect based on VRAM
        fps: target frame rate
    
    Returns:
        dict mapping chord_name -> output_path
    """
    print("\n" + "="*60)
    print("VIDEO DOWNSCALING PIPELINE")
    print("="*60)
    
    # Auto-detect resolution if not specified
    if resolution is None:
        width, height, desc = get_recommended_resolution()
    else:
        width, height, desc = resolution
    
    print(f"Target resolution: {width}x{height} ({desc})")
    print(f"Target FPS: {fps}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_extensions = ['*.mov', '*.mp4', '*.avi', '*.mkv', '*.MOV', '*.MP4']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return {}
    
    print(f"\nFound {len(video_files)} video files")
    
    # Process each video
    processed = {}
    skipped = []
    failed = []
    
    for i, video_path in enumerate(sorted(video_files)):
        filename = os.path.basename(video_path)
        chord_name = parse_chord_from_filename(filename)
        
        if chord_name is None:
            skipped.append(filename)
            continue
        
        output_filename = f"{chord_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n[{i+1}/{len(video_files)}] {filename} -> {output_filename}")
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"  Already exists, skipping...")
            processed[chord_name] = output_path
            continue
        
        # Downscale
        success = downscale_video(video_path, output_path, width, height, fps)
        
        if success:
            # Get file size comparison
            orig_size = os.path.getsize(video_path) / (1024*1024)
            new_size = os.path.getsize(output_path) / (1024*1024)
            print(f"  ✓ Done: {orig_size:.1f}MB -> {new_size:.1f}MB ({100*new_size/orig_size:.0f}%)")
            processed[chord_name] = output_path
        else:
            print(f"  ✗ Failed")
            failed.append(filename)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed: {len(processed)} videos")
    if skipped:
        print(f"Skipped (duplicates): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Chords available: {list(processed.keys())}")
    
    return processed


# === CLASSIFICATION FUNCTIONS ===

def classify_video_with_stages(video_path: str, model_path: str = "chord_classifier.pt", device='cuda'):
    """
    Classify a video and return intermediate stage visualizations.
    Full pipeline: SAM segmentation → DINO features → Classification
    
    Args:
        video_path: path to video file
        model_path: path to trained model weights
        device: 'cuda' or 'cpu'
    
    Returns:
        tuple: (predicted_chord, confidence, probs_dict, stage_data)
            - predicted_chord: string name of predicted chord
            - confidence: float confidence score
            - probs_dict: dict mapping chord_name -> probability
            - stage_data: dict with visualization frames for each stage
    """
    print(f"\n=== CLASSIFYING VIDEO WITH STAGES: {video_path} ===")
    
    # --- Step 1: Segmentation (SAM 3) with stage outputs ---
    print("\n--- Step 1: SAM Segmentation ---")
    sam_preprocessor = ChordAnalysisPreprocessor(device=device)
    
    masked_frames, stage_data = sam_preprocessor.segment_video_with_stages(
        video_path, 
        classes=["person", "guitar"]
    )
    
    # Cleanup SAM
    print("Cleaning up SAM model...")
    del sam_preprocessor
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Step 2: Feature Extraction (DINOv3) ---
    print("\n--- Step 2: DINO Feature Extraction ---")
    dino_extractor = DINOv3FeatureExtractor(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    
    features = dino_extractor.extract_features(masked_frames, batch_size=16)
    
    # Cleanup DINO
    del dino_extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Step 3: Classification ---
    print("\n--- Step 3: Chord Classification ---")
    model = ChordLinearProbe(input_dim=384, num_classes=len(CHORD_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Add batch dimension: (frames, 384) -> (1, frames, 384)
    features = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()
    
    predicted_chord = CLASS_NAMES[predicted_idx]
    
    # Create probability dictionary
    probs_dict = {CLASS_NAMES[i]: probs[0, i].item() for i in range(len(CLASS_NAMES))}
    
    print(f"\n{'='*50}")
    print(f"PREDICTED CHORD: {predicted_chord}")
    print(f"CONFIDENCE: {confidence*100:.1f}%")
    print(f"{'='*50}")
    
    return predicted_chord, confidence, probs_dict, stage_data


def classify_video(video_path: str, model_path: str = "chord_classifier.pt", device='cuda'):
    """
    Classify a video using the trained chord classifier.
    Full pipeline: SAM segmentation → DINO features → Classification
    
    Args:
        video_path: path to video file
        model_path: path to trained model weights
        device: 'cuda' or 'cpu'
    
    Returns:
        tuple: (predicted_chord, confidence, probabilities)
    """
    print(f"\n=== CLASSIFYING VIDEO: {video_path} ===")
    
    # --- Step 1: Segmentation (SAM 3) ---
    print("\n--- Step 1: SAM Segmentation ---")
    sam_preprocessor = ChordAnalysisPreprocessor(device=device)
    
    masked_frames = sam_preprocessor.segment_video(
        video_path, 
        classes=["person", "guitar"]
    )
    
    # Cleanup SAM
    print("Cleaning up SAM model...")
    del sam_preprocessor
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Step 2: Feature Extraction (DINOv3) ---
    print("\n--- Step 2: DINO Feature Extraction ---")
    dino_extractor = DINOv3FeatureExtractor(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    
    features = dino_extractor.extract_features(masked_frames, batch_size=16)
    
    # Cleanup DINO
    del dino_extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- Step 3: Classification ---
    print("\n--- Step 3: Chord Classification ---")
    model = ChordLinearProbe(input_dim=384, num_classes=len(CHORD_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Add batch dimension: (frames, 384) -> (1, frames, 384)
    features = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()
    
    predicted_chord = CLASS_NAMES[predicted_idx]
    
    print(f"\n{'='*50}")
    print(f"PREDICTED CHORD: {predicted_chord}")
    print(f"CONFIDENCE: {confidence*100:.1f}%")
    print(f"{'='*50}")
    
    # Show all probabilities
    print("\nAll class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[0, i].item()*100:.1f}%")
    
    return predicted_chord, confidence, probs


def classify_from_embeddings(embeddings_path: str, model_path: str = "chord_classifier.pt", device='cuda'):
    """
    Classify a chord from pre-extracted embeddings (skip SAM + DINO).
    
    Args:
        embeddings_path: path to .pt file with embeddings
        model_path: path to trained model weights
        device: 'cuda' or 'cpu'
    """
    print(f"\n=== CLASSIFYING FROM EMBEDDINGS: {embeddings_path} ===")
    
    # Load embeddings
    features = torch.load(embeddings_path, map_location=device)
    print(f"Loaded embeddings shape: {features.shape}")
    
    # Load model
    model = ChordLinearProbe(input_dim=384, num_classes=len(CHORD_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Add batch dimension: (frames, 384) -> (1, frames, 384)
    features = features.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()
    
    predicted_chord = CLASS_NAMES[predicted_idx]
    
    print(f"\n{'='*50}")
    print(f"PREDICTED CHORD: {predicted_chord}")
    print(f"CONFIDENCE: {confidence*100:.1f}%")
    print(f"{'='*50}")
    
    # Show all probabilities
    print("\nAll class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[0, i].item()*100:.1f}%")
    
    return predicted_chord, confidence, probs


def extract_and_save_embeddings(video_path: str, output_path: str, device='cuda', 
                                 sam_preprocessor=None, dino_extractor=None):
    """Extract embeddings from a video and save them.
    
    Args:
        video_path: path to video file
        output_path: path to save embeddings
        device: 'cuda' or 'cpu'
        sam_preprocessor: optional pre-loaded SAM model (for batch processing)
        dino_extractor: optional pre-loaded DINO model (for batch processing)
    """
    print(f"\n=== EXTRACTING EMBEDDINGS: {video_path} ===")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # --- Step 1: Segmentation (SAM 3) ---
    print("\n--- Step 1: SAM Segmentation ---")
    own_sam = sam_preprocessor is None
    if own_sam:
        sam_preprocessor = ChordAnalysisPreprocessor(device=device)
    
    masked_frames = sam_preprocessor.segment_video(
        video_path, 
        classes=["person", "guitar"]
    )
    
    # Cleanup SAM if we created it
    if own_sam:
        del sam_preprocessor
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Step 2: Feature Extraction (DINOv3) ---
    print("\n--- Step 2: DINO Feature Extraction ---")
    own_dino = dino_extractor is None
    if own_dino:
        dino_extractor = DINOv3FeatureExtractor(
            model_name="facebook/dinov3-vits16-pretrain-lvd1689m"
        )
    
    features = dino_extractor.extract_features(masked_frames, batch_size=8)  # Reduced batch size
    
    # Cleanup DINO if we created it
    if own_dino:
        del dino_extractor
        gc.collect()
        torch.cuda.empty_cache()
    
    # Clear masked_frames to free memory
    del masked_frames
    gc.collect()
    
    # Save
    torch.save(features.cpu(), output_path)  # Move to CPU before saving
    print(f"\nEmbeddings saved to {output_path}")
    print(f"Shape: {features.shape}")
    
    # Clear features
    result_shape = features.shape
    del features
    gc.collect()
    torch.cuda.empty_cache()
    
    return result_shape


def batch_extract_embeddings(video_dir: str, output_dir: str, device='cuda', skip_existing: bool = True):
    """
    Extract embeddings from all videos in a directory.
    
    Args:
        video_dir: directory containing downscaled videos
        output_dir: directory to save embeddings (.pt files)
        device: 'cuda' or 'cpu'
        skip_existing: skip videos that already have embeddings
    
    Returns:
        dict mapping chord_name -> embeddings_path
    """
    import time
    
    print("\n" + "="*60)
    print("BATCH EMBEDDING EXTRACTION")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(video_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return {}
    
    video_files = sorted(video_files)
    print(f"Found {len(video_files)} videos to process")
    print(f"Output directory: {output_dir}")
    print(f"Skip existing: {skip_existing}")
    
    # Track results
    processed = {}
    skipped = []
    failed = []
    start_time = time.time()
    
    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        chord_name = os.path.splitext(filename)[0]  # e.g., 'Am.mp4' -> 'Am'
        output_path = os.path.join(output_dir, f"{chord_name}.pt")
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(video_files)}] Processing: {chord_name}")
        print(f"{'='*60}")
        
        # Skip if already exists
        if skip_existing and os.path.exists(output_path):
            print(f"  Embeddings already exist at {output_path}, skipping...")
            processed[chord_name] = output_path
            skipped.append(chord_name)
            continue
        
        try:
            # AGGRESSIVE memory cleanup before each video
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Extract embeddings (each video gets fresh models)
            extract_and_save_embeddings(video_path, output_path, device)
            processed[chord_name] = output_path
            
            # AGGRESSIVE cleanup after each video
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Print memory status
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory: {mem_used:.2f}GB used, {mem_reserved:.2f}GB reserved")
            
            # ETA calculation
            elapsed = time.time() - start_time
            videos_done = i + 1 - len(skipped)
            if videos_done > 0:
                avg_time = elapsed / videos_done
                remaining = len(video_files) - i - 1
                eta_seconds = avg_time * remaining
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                print(f"\n  ETA for remaining {remaining} videos: {eta_min}m {eta_sec}s")
                
        except Exception as e:
            print(f"  ERROR processing {chord_name}: {e}")
            failed.append(chord_name)
            # AGGRESSIVE cleanup on error
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset CUDA to clear any fragmented memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("BATCH EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"Processed: {len(processed)} embeddings")
    if skipped:
        print(f"Skipped (already existed): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nEmbeddings saved to: {output_dir}/")
    print(f"Chords: {list(processed.keys())}")
    
    return processed