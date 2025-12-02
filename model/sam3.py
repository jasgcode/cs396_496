import os
import cv2
import gc

import torch
import numpy as np
import sys
import matplotlib
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,)

class ChordAnalysisPreprocessor:
    def __init__(self, device='cuda', model_path=None):
        """
        Initializes the SAM 3 predictor.
        """
        # Use all available GPUs or specific device
        if device == 'cuda':
            self.gpus = range(torch.cuda.device_count())
        else:
            self.gpus = [0] # Default to 0 if specific logic needed

        print("Initializing SAM 3 Predictor...")
        self.predictor = build_sam3_video_predictor(gpus_to_use=self.gpus)
        print("Predictor initialized.")

    def _propagate_video(self, session_id):
        """
        Internal helper to run propagation on the video and collect results.
        """
        outputs_per_frame = {}
        # Propagate from start to end
        stream = self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        )
        
        for response in stream:
            outputs_per_frame[response["frame_index"]] = response["outputs"]
            
        return outputs_per_frame

    def _load_video_frames(self, video_path):
        """
        Reads video frames into memory for final masking/export.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB (DINOv3/PIL)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def segment_video(self, video_path, classes=["person", "guitar"]):
        """
        Main pipeline method. 
        1. Loads video features.
        2. Segments each class individually.
        3. Combines masks.
        4. Returns masked images for DINOv3.
        """
        # 1. Start Session (Loads video & extracts features)
        print(f"Starting session for: {video_path}")
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]
        
        # Store raw masks for each class here
        class_masks = {} 

        try:
            # 2. Run inference for each class
            for class_name in classes:
                print(f"Segmenting class: {class_name}...")
                
                # A. Add text prompt to the first frame
                self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=class_name
                    )
                )
                
                # B. Propagate segmentation through video
                class_masks[class_name] = self._propagate_video(session_id)
                
                # C. Reset session to clear prompts but keep video embeddings
                self.predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=session_id,
                    )
                )
                
        finally:
            # Ensure session is closed to free GPU memory
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

        # 3. Load original frames for merging
        original_frames = self._load_video_frames(video_path)
        
        # 4. Process and Combine
        return self._generate_masked_output(original_frames, class_masks)

    def _generate_masked_output(self, frames, class_masks):
        """
        Combines masks from all classes and applies them to the original frames.
        
        SAM3 raw output structure per frame:
        {
            'out_obj_ids': np.array([0, 1, ...]),
            'out_binary_masks': np.array of shape (num_objects, H, W),
            'out_probs': np.array([...]),
            'out_boxes_xywh': np.array([...])
        }
        """
        processed_data = []
        
        for frame_idx, frame_img in enumerate(frames):
            # Create an empty mask for this frame (height, width)
            combined_mask = np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=bool)
            
            # Iterate through each class (Person, Guitar)
            for class_name, output_data in class_masks.items():
                if frame_idx in output_data:
                    frame_output = output_data[frame_idx]
                    
                    # SAM3 raw output has 'out_binary_masks' array
                    if 'out_binary_masks' in frame_output:
                        # Raw format: iterate through all detected objects
                        for idx, obj_id in enumerate(frame_output['out_obj_ids']):
                            binary_mask = frame_output['out_binary_masks'][idx]
                            # Logical OR to combine masks
                            combined_mask = np.logical_or(combined_mask, binary_mask)
                    else:
                        # Already processed format: {obj_id: mask}
                        for obj_id, binary_mask in frame_output.items():
                            combined_mask = np.logical_or(combined_mask, binary_mask)
            
            # Apply mask: Where mask is 0, set image pixel to 0 (Black background)
            # Expand mask to 3 channels for RGB multiplication
            mask_3c = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
            masked_image = frame_img * mask_3c
            
            # Convert to PIL for DINOv2/DINOv3
            pil_image = Image.fromarray(masked_image.astype('uint8'))
            processed_data.append(pil_image)
            
        return processed_data

    def segment_video_with_stages(self, video_path, classes=["person", "guitar"]):
        """
        Main pipeline method that returns individual stage outputs for visualization.
        
        Returns:
            tuple: (masked_frames, stage_data)
                - masked_frames: List of PIL images with combined mask (for DINO)
                - stage_data: dict with keys:
                    - 'original': list of original frames (PIL)
                    - 'class_masks': dict mapping class_name -> list of masked frames (PIL)
                    - 'combined': list of combined masked frames (PIL)
        """
        # 1. Start Session (Loads video & extracts features)
        print(f"Starting session for: {video_path}")
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]
        
        # Store raw masks for each class here
        class_masks = {} 

        try:
            # 2. Run inference for each class
            for class_name in classes:
                print(f"Segmenting class: {class_name}...")
                
                # A. Add text prompt to the first frame
                self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=class_name
                    )
                )
                
                # B. Propagate segmentation through video
                class_masks[class_name] = self._propagate_video(session_id)
                
                # C. Reset session to clear prompts but keep video embeddings
                self.predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=session_id,
                    )
                )
                
        finally:
            # Ensure session is closed to free GPU memory
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

        # 3. Load original frames for merging
        original_frames = self._load_video_frames(video_path)
        
        # 4. Generate all stage outputs
        stage_data = self._generate_stage_outputs(original_frames, class_masks, classes)
        
        # 5. Combined masked output (for DINO)
        combined_frames = stage_data['combined']
        
        return combined_frames, stage_data

    def _generate_stage_outputs(self, frames, class_masks, classes):
        """
        Generate visualization outputs for each stage of the pipeline.
        
        Returns:
            dict with keys:
                - 'original': list of original frames (PIL)
                - 'class_masks': dict mapping class_name -> list of masked frames (PIL)
                - 'combined': list of combined masked frames (PIL)
        """
        stage_data = {
            'original': [],
            'class_masks': {class_name: [] for class_name in classes},
            'combined': []
        }
        
        for frame_idx, frame_img in enumerate(frames):
            # Original frame
            stage_data['original'].append(Image.fromarray(frame_img.astype('uint8')))
            
            # Combined mask for this frame
            combined_mask = np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=bool)
            
            # Process each class
            for class_name in classes:
                class_mask = np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=bool)
                
                if class_name in class_masks and frame_idx in class_masks[class_name]:
                    frame_output = class_masks[class_name][frame_idx]
                    
                    if 'out_binary_masks' in frame_output:
                        for idx, obj_id in enumerate(frame_output['out_obj_ids']):
                            binary_mask = frame_output['out_binary_masks'][idx]
                            class_mask = np.logical_or(class_mask, binary_mask)
                    else:
                        for obj_id, binary_mask in frame_output.items():
                            class_mask = np.logical_or(class_mask, binary_mask)
                
                # Apply class mask to frame
                mask_3c = np.repeat(class_mask[:, :, np.newaxis], 3, axis=2)
                class_masked_image = frame_img * mask_3c
                stage_data['class_masks'][class_name].append(
                    Image.fromarray(class_masked_image.astype('uint8'))
                )
                
                # Add to combined mask
                combined_mask = np.logical_or(combined_mask, class_mask)
            
            # Combined masked frame
            mask_3c = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
            combined_masked_image = frame_img * mask_3c
            stage_data['combined'].append(
                Image.fromarray(combined_masked_image.astype('uint8'))
            )
        
        return stage_data


import matplotlib.pyplot as plt
import cv2

def test_masking(video_path, preprocessor):
    print(f"Testing masking on: {video_path}")
    
    # 1. Run the segmentation (this returns the black-background PIL images)
    #    We limit to first 30 frames for a quick test if possible, 
    #    but segment_video processes the whole file. 
    masked_pil_images = preprocessor.segment_video(video_path, classes=["person", "guitar"])
    
    # 2. Load original frames just for comparison
    original_frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # 3. Visualize Start, Middle, and End frames
    num_frames = len(masked_pil_images)
    indices_to_check = [0, num_frames // 2, num_frames - 1]
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices_to_check):
        if idx >= len(original_frames): continue
        
        # Original
        plt.subplot(3, 2, 2*i + 1)
        plt.imshow(original_frames[idx])
        plt.title(f"Frame {idx}: Original")
        plt.axis("off")
        
        # Masked Output (Result from Class)
        plt.subplot(3, 2, 2*i + 2)
        plt.imshow(masked_pil_images[idx])
        plt.title(f"Frame {idx}: Masked (DINO Input)")
        plt.axis("off")
        
    plt.tight_layout()
    
    # Save figure instead of showing (works in script mode)
    output_fig_path = "test_masking_output.png"
    plt.savefig(output_fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_fig_path}")
    plt.close()  # Close to free memory

# # === RUN THE TEST ===
# # Ensure you have your model path or parameters set correctly
# preprocessor = ChordAnalysisPreprocessor(device='cuda') # or 'cpu'

# # Replace with your actual video path
# video_path = "dataset/G_smaller.mp4" 

# test_masking(video_path, preprocessor)

# # Cleanup after testing
# import gc
# preprocessor.predictor.shutdown()
# del preprocessor
# gc.collect()
# torch.cuda.empty_cache()