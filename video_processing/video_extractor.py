# video_processing/video_extractor.py

import os
import cv2
from PIL import Image
from typing import List, Tuple, Literal, Union, Optional
# from concurrent.futures import ThreadPoolExecutor # Not used directly in this file after edit
# from glob import glob # Not used directly in this file after edit
import math # For ceiling division if needed, or general math

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

    # If FPS is 0 or invalid, it's problematic for "1 frame per second" logic.
    # We can try to infer a duration or default to frame-based extraction.
    # For simplicity, if fps is 0, we treat duration_1fps as 0, pushing to the "extract N frames" logic.
    if fps <= 0:
        print(f"Warning: Video {video_path} has invalid FPS ({fps}). Will extract up to 20 frames based on total_frames.")
        # This will make num_frames_at_1fps = 0, handled by the first `if` block below
        num_frames_at_1fps = 0
    else:
        # Number of frames if we take 1 per second (effectively duration in whole seconds)
        num_frames_at_1fps = int(total_frames / fps)


    frames_to_capture_details = []  # List of (frame_index_to_seek, actual_timestamp_sec, filename_counter)

    # Determine how many frames to extract and their indices
    if num_frames_at_1fps == 0 and total_frames > 0:
        # Video is less than 1 second long OR FPS was invalid.
        # Extract up to 20 frames, or all frames if total_frames < 20.
        num_to_extract = min(total_frames, 20)
        if num_to_extract > 0: # Ensure we actually try to extract something if possible
            for i in range(num_to_extract):
                # Evenly space across total_frames
                frame_idx = int(i * (total_frames / num_to_extract))
                # Ensure frame_idx is within bounds, especially for the last frame
                frame_idx = min(frame_idx, total_frames - 1)
                actual_time = frame_idx / fps if fps > 0 else float(i) # Best effort timestamp
                frames_to_capture_details.append((frame_idx, actual_time, i))

    elif num_frames_at_1fps <= 20:
        # Standard 1 frame per second, video is 1-20 seconds long
        num_to_extract = num_frames_at_1fps
        for i in range(num_to_extract): # i will be sec_count: 0, 1, ..., num_to_extract-1
            frame_idx = int(i * fps)
            frame_idx = min(frame_idx, total_frames - 1) # Ensure in bounds
            actual_time = float(i) # Timestamp is the second itself
            frames_to_capture_details.append((frame_idx, actual_time, i))
    else:
        # Video is longer than 20 seconds (if sampled at 1 FPS), so extract 20 evenly spaced frames
        num_to_extract = 20
        if num_to_extract > 0: # Should always be true here (20)
            for i in range(num_to_extract):
                # Calculate the position as a fraction of total_frames
                frame_idx = int(i * (total_frames / num_to_extract))
                frame_idx = min(frame_idx, total_frames - 1) # Ensure in bounds
                actual_time = frame_idx / fps if fps > 0 else float(i) # Best effort timestamp
                frames_to_capture_details.append((frame_idx, actual_time, i))

    results = []
    if mode == "disk":
        if not output_dir:
            cap.release()
            raise ValueError("output_dir must be provided when mode is 'disk'")
        os.makedirs(output_dir, exist_ok=True)

    # Deduplicate frame indices to avoid processing the same frame multiple times,
    # which can happen with low FPS or specific total_frames / num_to_extract ratios.
    # We keep the first encountered (frame_idx, timestamp, filename_counter) for a given frame_idx.
    seen_frame_indices = set()
    unique_frames_to_capture = []
    for detail in frames_to_capture_details:
        frame_idx_to_seek = detail[0]
        if frame_idx_to_seek not in seen_frame_indices:
            unique_frames_to_capture.append(detail)
            seen_frame_indices.add(frame_idx_to_seek)

    # If after all calculations and deduplication, we have no frames but the video has frames,
    # try to get at least the first frame.
    if not unique_frames_to_capture and total_frames > 0:
        print(f"Warning: No frames selected for {video_path} despite {total_frames} total frames. Taking first frame.")
        unique_frames_to_capture.append((0, 0.0, 0))


    for frame_idx_to_seek, actual_timestamp_sec, filename_counter in unique_frames_to_capture:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_to_seek)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame at index {frame_idx_to_seek} (target time {actual_timestamp_sec:.2f}s) in {video_path}")
            continue

        if resize_to_720p:
            # Ensure frame is not empty before resize
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            else:
                print(f"Warning: Frame at index {frame_idx_to_seek} is empty or invalid before resize in {video_path}")
                continue


        if mode == "pil":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            results.append(pil_image) # Storing PIL image directly, timestamp might be lost unless part of a wrapper object
                                      # The original function signature implies results are just images or (filename, time)
                                      # If timestamp is needed with PIL, signature should change.
                                      # For now, adhering to original return type implies timestamp is not part of PIL result.
        elif mode == "disk":
            filename = os.path.join(output_dir, f"frame_{filename_counter:04d}.jpg") # Use filename_counter for sequential names
            cv2.imwrite(filename, frame)
            results.append((filename, actual_timestamp_sec))

    cap.release()
    return results

# The process_all_videos_in_parallel function remains unchanged as it calls the above.
# It's good practice to import ThreadPoolExecutor and glob here if this file could be run standalone
# or if this function were to be used by other modules directly.
# For now, they are effectively unused if only `extract_frames_per_second` is imported.

def process_all_videos_in_parallel(data_root: str, max_workers: int = 4):
    # Re-importing here as they were commented out at the top if not used by extract_frames_per_second
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