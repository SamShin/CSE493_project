import os
import sys
from concurrent.futures import ThreadPoolExecutor
from yt_dlp import YoutubeDL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from video_processing.video_extractor import extract_frames_per_second

# Config
URL_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "links.txt"))
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videos"))
FILENAME = "video.mp4"
MAX_PARALLEL_DOWNLOADS = 3

def get_urls(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [
            line.strip().replace("youtube.com/shorts/", "youtube.com/watch?v=")
            for line in f if line.strip()
        ]

def download_and_extract(index, url):
    subdir = os.path.join(DATA_ROOT, f"video_{index:03d}")
    os.makedirs(subdir, exist_ok=True)

    video_path = os.path.join(subdir, FILENAME)
    frames_dir = os.path.join(subdir, "frames")

    if os.path.exists(video_path):
        print(f"[✓] Already downloaded: video_{index:03d}")
        return

    ydl_opts = {
        'format': 'bv*[height<=720]+ba/b[height<=720]/bestaudio',
        'merge_output_format': 'mp4',
        'outtmpl': video_path,
        'quiet': True,
        'noplaylist': True,
        'concurrent_fragment_downloads': 4,
    }

    try:
        print(f"[↓] Downloading video_{index:03d} from {url}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"[✓] Downloaded video_{index:03d}")

        extract_frames_per_second(
            video_path=video_path,
            mode="disk",
            output_dir=frames_dir,
            resize_to_720p=True
        )
        print(f"[📸] Extracted frames for video_{index:03d}")
    except Exception as e:
        print(f"[X] Failed video_{index:03d}: {e}")

def main():
    urls = get_urls(URL_FILE)
    os.makedirs(DATA_ROOT, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOWNLOADS) as executor:
        futures = [
            executor.submit(download_and_extract, i + 1, url)
            for i, url in enumerate(urls)
        ]
        for f in futures:
            f.result()

if __name__ == "__main__":
    main()
