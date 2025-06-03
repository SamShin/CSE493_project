import os
import cv2
from PIL import Image
from typing import List, Tuple, Literal, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from glob import glob

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
    duration_seconds = int(total_frames // fps)

    if mode == "disk" and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []

    for sec in range(duration_seconds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at {sec}s in {video_path}")
            continue

        if resize_to_720p:
            frame = cv2.resize(frame, (1280, 720))

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

def process_all_videos_in_parallel(data_root: str, max_workers: int = 4):
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
