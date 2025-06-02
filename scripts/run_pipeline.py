import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
from video_processing.video_extractor import extract_frames_per_second, segment_objects_yolo

def run_video_analysis(video_path: str, output_dir: str = "output", visualize: bool = True):
    # Step 1: Extract 1 frame per second
    frames = extract_frames_per_second(video_path, mode="pil")
    # Step 3: Save visualized results (optional)

# Run the pipeline if this script is executed directly
if __name__ == "__main__":
    video_path = "a.mp4"  # Replace with your input video path
    run_video_analysis(video_path)