import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- Constants ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data/test_per_video_analysis"
VIDEO_DIR = ROOT_DIR / "videos"
OUTPUT_DIR = ROOT_DIR / "annotated"

# Load default font
try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except:
    FONT = ImageFont.load_default()


def overlay_boxes(video_number: str):
    json_path = DATA_DIR / f"{video_number}_analysis.json"
    frame_dir = VIDEO_DIR / video_number / "frames"
    output_dir = OUTPUT_DIR / video_number / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[!] JSON not found: {json_path}")
        return
    if not frame_dir.exists():
        print(f"[!] Frame directory missing: {frame_dir}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for frame_data in data["frames"]:
        frame_idx = frame_data["frame"]
        image_path = frame_dir / f"frame_{frame_idx:04d}.jpg"
        if not image_path.exists():
            print(f"[!] Skipping missing: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        W, H = image.size

        # People boxes (red)
        for person in frame_data.get("people", []):
            x, y, w, h = person["bbox"]
            box = [x * W, y * H, (x + w) * W, (y + h) * H]
            label = f"{person['type']} ({person['emotion']})"
            draw.rectangle(box, outline="red", width=4)
            draw.text((box[0], max(0, box[1] - 25)), label, fill="red", font=FONT)

        # Object boxes (blue)
        for obj in frame_data.get("detected_objects", []):
            x, y, w, h = obj["bbox"]
            box = [x * W, y * H, (x + w) * W, (y + h) * H]
            label = f"{obj['Name']}"
            draw.rectangle(box, outline="blue", width=3)
            draw.text((box[0], max(0, box[1] - 50)), label, fill="blue", font=FONT)

        out_path = output_dir / f"frame_{frame_idx:04d}.jpg"
        image.save(out_path)
        print(f"[✓] Saved: {out_path}")


def main():
    print("📽️  Video Annotation Overlay Tool")
    mode = input("Do you want to process a single video or all? (single/all): ").strip().lower()

    if mode == "single":
        video_number = input("Enter the video number (e.g., video_001): ").strip()
        overlay_boxes(video_number)
    elif mode == "all":
        all_jsons = sorted(DATA_DIR.glob("video_*_analysis.json"))
        for json_file in all_jsons:
            video_number = json_file.stem.replace("_analysis", "")
            print(f"\n[▶] Processing {video_number}")
            overlay_boxes(video_number)
    else:
        print("[!] Invalid input. Please type 'single' or 'all'.")


if __name__ == "__main__":
    main()
