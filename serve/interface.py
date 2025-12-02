import gradio as gr
import cv2
import numpy as np
import os
import sys
import tempfile
import subprocess
import shutil
from PIL import Image
import pandas as pd
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.classsify import classify_video_with_stages, CHORD_CLASSES, CLASS_NAMES, downscale_video, get_recommended_resolution


def preprocess_video(video_path: str, max_width: int = 640, max_height: int = 360, fps: int = 15) -> str:
    """
    Downscale video to manageable resolution before processing.
    Always re-encodes to ensure web compatibility.
    
    Args:
        video_path: path to input video
        max_width: maximum width (default 640 for 360p)
        max_height: maximum height (default 360 for 360p)
        fps: target frame rate
    
    Returns:
        path to downscaled video (or re-encoded original if already small enough)
    """
    # Check original video dimensions
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Original video: {orig_width}x{orig_height} @ {orig_fps:.1f}fps")
    
    # Create temp file for processed video
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "processed.mp4")
    
    # Always re-encode to ensure web compatibility (H.264 + yuv420p)
    if orig_width <= max_width and orig_height <= max_height:
        print(f"Video at target resolution, re-encoding for web compatibility...")
        # Just re-encode without scaling
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-r', str(fps),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-an',  # No audio
            output_path
        ]
    else:
        print(f"Downscaling to {max_width}x{max_height} @ {fps}fps...")
        # Scale and re-encode
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f'scale={max_width}:{max_height}:force_original_aspect_ratio=decrease,pad={max_width}:{max_height}:(ow-iw)/2:(oh-ih)/2',
            '-r', str(fps),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-an',  # No audio
            output_path
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(output_path):
            orig_size = os.path.getsize(video_path) / (1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Processed: {orig_size:.1f}MB -> {new_size:.1f}MB")
            return output_path
        else:
            print(f"ffmpeg error: {result.stderr[:500]}")
            print("Using original video")
            return video_path
    except Exception as e:
        print(f"Preprocessing failed: {e}, using original video")
        return video_path


def frames_to_video(frames, output_path, fps=15):
    """Convert list of PIL images to a web-compatible video file."""
    if not frames:
        return None
    
    # Get dimensions from first frame
    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        width, height = first_frame.size
    else:
        height, width = first_frame.shape[:2]
    
    # First write to a temp file with cv2
    temp_path = output_path + ".temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if isinstance(frame, Image.Image):
            # Convert PIL to cv2 format (RGB -> BGR)
            frame_np = np.array(frame)
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_np = frame
        
        out.write(frame_np)
    
    out.release()
    
    # Re-encode with ffmpeg to H.264 for web compatibility
    if shutil.which('ffmpeg'):
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',  # Required for browser compatibility
            '-movflags', '+faststart',  # Enable streaming
            output_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            os.remove(temp_path)
            return output_path
        except Exception as e:
            print(f"ffmpeg re-encode failed: {e}, using temp file")
            os.rename(temp_path, output_path)
            return output_path
    else:
        # Fallback if no ffmpeg
        os.rename(temp_path, output_path)
        return output_path


def recognize_chord(video):
    """
    Process video through full pipeline and return stage visualizations + classification.
    """
    if video is None:
        return None, None, None, None, None, "No video uploaded", None
    
    # Handle gr.File input (returns file path as string)
    video_path = video if isinstance(video, str) else video.name
    
    try:
        # Get recommended resolution based on VRAM
        rec_width, rec_height, desc = get_recommended_resolution()
        print(f"Using resolution: {rec_width}x{rec_height} ({desc})")
        
        # Downscale/convert video to web-compatible format
        processed_video_path = preprocess_video(video_path, max_width=rec_width, max_height=rec_height, fps=15)
        
        # Run the full classification pipeline with stage outputs
        predicted_chord, confidence, probs_dict, stage_data = classify_video_with_stages(
            processed_video_path,
            model_path="chord_classifier.pt",
            device='cuda'
        )
        
        # Create temporary directory for output videos
        temp_dir = tempfile.mkdtemp()
        
        # Convert stage frames to videos
        original_video_path = None
        person_video_path = None
        guitar_video_path = None
        combined_video_path = None
        
        # Original video - use the processed (web-compatible) version
        if stage_data.get('original'):
            original_video_path = os.path.join(temp_dir, "original.mp4")
            frames_to_video(stage_data['original'], original_video_path)
        
        # Person mask video
        if stage_data.get('class_masks', {}).get('person'):
            person_video_path = os.path.join(temp_dir, "person_mask.mp4")
            frames_to_video(stage_data['class_masks']['person'], person_video_path)
        
        # Guitar mask video
        if stage_data.get('class_masks', {}).get('guitar'):
            guitar_video_path = os.path.join(temp_dir, "guitar_mask.mp4")
            frames_to_video(stage_data['class_masks']['guitar'], guitar_video_path)
        
        # Combined mask video
        if stage_data.get('combined'):
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            frames_to_video(stage_data['combined'], combined_video_path)
        
        # Format result text
        result_text = f"**Predicted Chord: {predicted_chord}**\n\nConfidence: {confidence*100:.1f}%"
        
        # Create probability data for bar chart as DataFrame
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame({
            "Chord": [item[0] for item in sorted_probs],
            "Probability (%)": [item[1] * 100 for item in sorted_probs]
        })
        
        return (
            processed_video_path,  # Return the converted video for preview
            original_video_path,
            person_video_path,
            guitar_video_path,
            combined_video_path,
            result_text,
            df
        )
        
    except Exception as e:
        error_msg = f"Error during classification:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, None, None, error_msg, None


# Gradio interface
with gr.Blocks(title="Guitar Chord Recognition Demo") as demo:
    gr.Markdown("# Guitar Chord Recognition")
    gr.Markdown("### CS396 Final Project: Team 4")
    gr.Markdown("Upload a video of guitar playing to recognize the chord and see the segmentation pipeline stages.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Use File upload to avoid Chrome codec issues with MOV preview
            video_input = gr.File(
                label="Upload Video (MP4, MOV supported)",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
            )
            
            recognize_btn = gr.Button("Recognize Chord", variant="primary")
            
            gr.Markdown("---")
            
            # Show the converted video preview after processing
            gr.Markdown("### Uploaded Video (Converted)")
            input_preview = gr.Video(label="Input Preview", interactive=False)
            
            gr.Markdown("---")
            
            # Classification Result
            result_text = gr.Markdown(
                value="*Upload a video and click 'Recognize Chord' to see results*"
            )
    
    gr.Markdown("---")
    gr.Markdown("## Pipeline Visualization")
    gr.Markdown("See how the model processes your video through each stage:")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original Video")
            original_video = gr.Video(label="Original Input", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Person Mask")
            person_video = gr.Video(label="Person Segmentation", interactive=False)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("###  Guitar Mask")
            guitar_video = gr.Video(label="Guitar Segmentation", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Combined Mask")
            combined_video = gr.Video(label="Final Input to DINO", interactive=False)
    
    gr.Markdown("---")
    gr.Markdown("## Classification Probabilities")
    
    prob_chart = gr.BarPlot(
        x="Chord",
        y="Probability (%)",
        title="Chord Probabilities",
    )
    
    # Connect button to function
    recognize_btn.click(
        fn=recognize_chord,
        inputs=video_input,
        outputs=[
            input_preview,  # Show converted video
            original_video,
            person_video,
            guitar_video,
            combined_video,
            result_text,
            prob_chart
        ]
    )


if __name__ == "__main__":
    demo.launch(
        share=False,  
        show_error=True
    )