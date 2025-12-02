#!/usr/bin/env python3
"""
Main entry point for the Chord Classification Pipeline.

Usage:
    python main.py --mode <mode> [options]

Modes:
    train           - Train classifier on embeddings
    classify        - Classify video (full pipeline: SAM → DINO → Classifier)
    classify_emb    - Classify from pre-extracted embeddings
    extract         - Extract embeddings from a single video
    batch_extract   - Extract embeddings from all videos in a directory
    downscale       - Downscale videos for faster processing
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import torch
from glob import glob

from serve.classsify import (
    CHORD_CLASSES,
    classify_video,
    classify_from_embeddings,
    extract_and_save_embeddings,
    batch_extract_embeddings,
    downscale_videos,
)
from train.train import train_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Chord Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Downscale videos
    python main.py --mode downscale --input_dir dataset/data --output_dir dataset/data_downscaled
    
    # Extract embeddings from all videos
    python main.py --mode batch_extract --output_dir dataset/data_downscaled --embeddings_dir embeddings
    
    # Train classifier
    python main.py --mode train --embeddings_dir embeddings --epochs 500
    
    # Classify a new video (full pipeline)
    python main.py --mode classify --video path/to/video.mp4
    
    # Classify from pre-extracted embeddings (fast)
    python main.py --mode classify_emb --embeddings path/to/embeddings.pt
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "classify", "classify_emb", "extract", "batch_extract", "downscale"],
                        help="Operation mode")
    
    # Video/file paths
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (for classify/extract modes)")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to embeddings file (for classify_emb mode)")
    parser.add_argument("--model", type=str, default="chord_classifier.pt",
                        help="Path to model weights")
    parser.add_argument("--output", type=str, default="embeddings.pt",
                        help="Output path for extracted embeddings")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    
    # Directory paths
    parser.add_argument("--input_dir", type=str, default="dataset/data",
                        help="Input directory for videos")
    parser.add_argument("--output_dir", type=str, default="dataset/data_downscaled",
                        help="Output directory for downscaled videos")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings",
                        help="Directory for embeddings")
    
    # Processing options
    parser.add_argument("--fps", type=int, default=15,
                        help="Target FPS for downscaled videos")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip files that already exist")
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # === MODE HANDLERS ===
    
    if args.mode == "train":
        # Auto-discover embeddings from embeddings directory
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
            model_save_path=args.model
        )
        
    elif args.mode == "classify":
        if not args.video:
            print("Error: --video is required for classify mode")
            sys.exit(1)
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        classify_video(args.video, args.model, args.device)
        
    elif args.mode == "classify_emb":
        if not args.embeddings:
            print("Error: --embeddings is required for classify_emb mode")
            sys.exit(1)
        if not os.path.exists(args.embeddings):
            print(f"Error: Embeddings not found: {args.embeddings}")
            sys.exit(1)
        classify_from_embeddings(args.embeddings, args.model, args.device)
        
    elif args.mode == "extract":
        if not args.video:
            print("Error: --video is required for extract mode")
            sys.exit(1)
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        extract_and_save_embeddings(args.video, args.output, args.device)
        
    elif args.mode == "batch_extract":
        batch_extract_embeddings(
            video_dir=args.output_dir,
            output_dir=args.embeddings_dir,
            device=args.device,
            skip_existing=args.skip_existing
        )
        
    elif args.mode == "downscale":
        downscale_videos(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            resolution=None,  # Auto-detect based on VRAM
            fps=args.fps
        )


if __name__ == "__main__":
    main()
