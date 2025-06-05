import os
import cv2
import math
import numpy as np
from PIL import Image
from typing import List, Tuple, Literal, Union, Optional

def resize_with_padding(frame, target_size=(1280, 720), color=(0, 0, 0)):
    original_h, original_w = frame.shape[:2]
    target_w, target_h = target_size

    # Compute scaling factor and resized dimensions
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Resize frame
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas and paste resized frame in the center
    padded_frame = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    return padded_frame

def extract_frames_per_second(
    video_path: str,
    mode: Literal["pil", "disk"] = "pil",
    output_dir: Optional[str] = None,
    resize_to_720p: bool = True
) -> Union[List[Image.Image], List[Tuple[str, float]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        print(f"Warning: Video {video_path} has 0 frames.")
        return []

    if fps <= 0:
        print(f"Warning: Video {video_path} has invalid FPS ({fps}). Will extract up to 20 frames based on total_frames.")
        num_frames_at_1fps = 0
    else:
        num_frames_at_1fps = int(total_frames / fps)

    frames_to_capture_details = []

    if num_frames_at_1fps == 0 and total_frames > 0:
        num_to_extract = min(total_frames, 20)
        for i in range(num_to_extract):
            frame_idx = int(i * (total_frames / num_to_extract))
            frame_idx = min(frame_idx, total_frames - 1)
            actual_time = frame_idx / fps if fps > 0 else float(i)
            frames_to_capture_details.append((frame_idx, actual_time, i))
    elif num_frames_at_1fps <= 20:
        for i in range(num_frames_at_1fps):
            frame_idx = int(i * fps)
            frame_idx = min(frame_idx, total_frames - 1)
            actual_time = float(i)
            frames_to_capture_details.append((frame_idx, actual_time, i))
    else:
        for i in range(20):
            frame_idx = int(i * (total_frames / 20))
            frame_idx = min(frame_idx, total_frames - 1)
            actual_time = frame_idx / fps if fps > 0 else float(i)
            frames_to_capture_details.append((frame_idx, actual_time, i))

    results = []
    if mode == "disk":
        if not output_dir:
            cap.release()
            raise ValueError("output_dir must be provided when mode is 'disk'")
        os.makedirs(output_dir, exist_ok=True)

    seen_frame_indices = set()
    unique_frames_to_capture = []
    for detail in frames_to_capture_details:
        frame_idx_to_seek = detail[0]
        if frame_idx_to_seek not in seen_frame_indices:
            unique_frames_to_capture.append(detail)
            seen_frame_indices.add(frame_idx_to_seek)

    if not unique_frames_to_capture and total_frames > 0:
        print(f"Warning: No frames selected for {video_path} despite {total_frames} total frames. Taking first frame.")
        unique_frames_to_capture.append((0, 0.0, 0))

    for frame_idx_to_seek, actual_timestamp_sec, filename_counter in unique_frames_to_capture:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_to_seek)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame at index {frame_idx_to_seek} (target time {actual_timestamp_sec:.2f}s) in {video_path}")
            continue

        if resize_to_720p and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            frame = resize_with_padding(frame, (1280, 720))
        elif frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"Warning: Frame at index {frame_idx_to_seek} is empty or invalid before resize in {video_path}")
            continue

        if mode == "pil":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            results.append(pil_image)
        elif mode == "disk":
            filename = os.path.join(output_dir, f"frame_{filename_counter:04d}.jpg")
            cv2.imwrite(filename, frame)
            results.append((filename, actual_timestamp_sec))

    cap.release()
    return results

def process_all_videos_in_parallel(data_root: str, max_workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor
    from glob import glob

    video_paths = glob(os.path.join(data_root, "video_*/video.mp4"))

    def process_one(video_path):
        video_dir = os.path.dirname(video_path)
        frames_dir = os.path.join(video_dir, "frames")
        try:
            print(f"[→] Extracting frames from {video_path}")
            extract_frames_per_second(
                video_path=video_path,
                mode="disk",
                output_dir=frames_dir,
                resize_to_720p=True
            )
            print(f"[✓] Done: {video_path}")
        except Exception as e:
            print(f"[X] Failed {video_path}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_one, video_paths)
