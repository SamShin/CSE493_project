from typing import List, Tuple, Literal, Union, Optional
import os

import cv2
from PIL import Image

def extract_frames_per_second(
    video_path: str,
    mode: Literal["pil", "disk"] = "pil",
    output_dir: Optional[str] = None
) -> Union[List[Image.Image], List[Tuple[str, float]]]:
    """
    Extracts one frame per second from a video.

    Args:
        video_path (str): Path to the input video.
        mode (str): 'pil' to return list of PIL images,
                    'disk' to save frames to disk and return filenames with timestamps.
        output_dir (str, optional): Directory to save images if mode='disk'.

    Returns:
        If mode='pil':
            List[PIL.Image.Image]: List of PIL images.
        If mode='disk':
            List[Tuple[str, float]]: List of (filepath, timestamp) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(total_frames // fps)

    if mode == "disk":
        if output_dir is None:
            raise ValueError("output_dir must be specified when mode is 'disk'")
        os.makedirs(output_dir, exist_ok=True)

    results = []

    for sec in range(duration_seconds):
        frame_index = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at {sec}s")
            continue

        if mode == "pil":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            results.append(pil_image)
        elif mode == "disk":
            filename = os.path.join(output_dir, f"frame_{sec:04d}.jpg")
            cv2.imwrite(filename, frame)
            results.append((filename, sec))

    cap.release()
    return results