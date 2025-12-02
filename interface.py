import gradio as gr
import cv2
import librosa
import numpy as np
import os
import tempfile


chord_history = []

# Extract video frames and audio from mp4 file. Takes in the path to the video file
# and outputs frames and audio as a numpy array and the sample rate

def load_video_and_audio(video_path):
    print(f"Loading video from: {video_path}")
    
    # Extract video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    max_frames = 300  
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    try:
        print("Extracting audio...")
        audio_data, sample_rate = librosa.load(video_path, sr=None, duration=10) 
        print(f"Audio extracted: {audio_data.shape}, SR: {sample_rate}")
    except Exception as e:
        print(f"Audio extraction error: {e}")
        audio_data = np.array([])
        sample_rate = 22050
    
    return frames, audio_data, sample_rate

# Processes video and recognizes chords. Takes in video and classifies the current chord
# and keeps track of chord history
def recognize_chord(video):
    if video is None:
        return "No video recorded", "", ""
    
    # Load video and audio
    video_path = video
    frames, audio_data, sample_rate = load_video_and_audio(video_path)
    
    print(f"Loaded {len(frames)} frames")
    print(f"Audio shape: {audio_data.shape}, Sample rate: {sample_rate}")
    
 # I WILL EMBED MODEL HERE!
    # Example preprocessing steps:
    
    # 1. Process video frames 
    # processed_frames = []
    # for frame in frames:
    #     resized = cv2.resize(frame, (224, 224))
    #     normalized = resized / 255.0
    #     processed_frames.append(normalized)
    # video_tensor = np.array(processed_frames)
    
    # 2. Process audio 
    # if len(audio_data) > 0:
    #     mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    #     spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    #     chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    
    # 3. Run  model
    # chord = your_chord_model.predict({
    #     'video': video_tensor,
    #     'audio': mfcc
    # })

    
    chord = "C Major"  # Replace this with model output

    chord_history.append(chord)
    history_display = "\n".join([f"{i+1}. {c}" for i, c in enumerate(chord_history)])
    
    return chord, history_display

def clear_history():
    """Clear the chord history"""
    global chord_history
    chord_history = []
    return "", "", ""

# Gradio interface
with gr.Blocks(title=" Guitar Chord Recognition Demo") as demo:
    gr.Markdown("# Chord Recognition")
    gr.Markdown("# CS396 Final Project: Team 4")
    gr.Markdown("Record video to recognize chords and view history")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                sources=["webcam", "upload"],
                label=" Record Video",
                include_audio=True,
                height=400
            )
            
            gr.Markdown("Click the webcam icon above to start recording!")
            
            with gr.Row():
                recognize_btn = gr.Button("🎼 Recognize Chord", variant="primary")
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
        
        with gr.Column():

            current_chord = gr.Textbox(
                label="Current Chord",
                placeholder="Chord will appear here...",
                interactive=False,
                lines=2
            )
        
            chord_history_display = gr.Textbox(
                label="Chord History",
                placeholder="Previous chords will be listed here...",
                interactive=False,
                lines=10
            )
    
    recognize_btn.click(
        fn=recognize_chord,
        inputs=video_input,
        outputs=[current_chord, chord_history_display]
    )
    

    clear_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=[current_chord, chord_history_display]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,  
        show_error=True
    )