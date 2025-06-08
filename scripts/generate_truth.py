import base64
import concurrent.futures
import json
import os

import boto3

# --- Determine Project Root and Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Directory Configuration
DATA_DIRECTORY = os.path.join(PROJECT_ROOT_DIR, "videos")
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT_DIR, "data\\per_video_analysis")

# AWS Configuration
AWS_PROFILE_NAME = "personal"
AWS_REGION = "us-west-1"
REKOGNITION_MIN_CONFIDENCE = 75.0

# Parallelism
MAX_WORKERS_PER_VIDEO = 10

# --- Initialize AWS Rekognition ---
rekognition_client = None
try:
    session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION)
    rekognition_client = session.client("rekognition")
    print("AWS Rekognition client initialized.")
except Exception as e:
    print(f"Warning: Could not initialize AWS Rekognition client: {e}")
    rekognition_client = None


def analyze_people_with_rekognition(image_path, min_confidence=75.0):
    if not rekognition_client:
        return []

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"[Error reading image] {image_path}: {e}")
        return []

    people_data = []
    try:
        response = rekognition_client.detect_faces(
            Image={"Bytes": image_bytes}, Attributes=["ALL"]
        )
        for face in response.get("FaceDetails", []):
            confidence = face.get("Confidence", 0)
            if confidence < min_confidence:
                continue

            age = (face["AgeRange"]["Low"] + face["AgeRange"]["High"]) // 2
            if age <= 2:
                person_type = "baby"
            elif age <= 12:
                person_type = "child"
            elif age <= 19:
                person_type = "teenager"
            else:
                person_type = "adult"

            gender = face.get("Gender", {}).get("Value", "unknown").lower()

            emotions = face.get("Emotions", [])
            if emotions:
                primary = max(emotions, key=lambda x: x.get("Confidence", 0))
                emotion = primary.get("Type", "CALM").lower()
                emotion = {
                    "happy": "happy", "sad": "sad", "angry": "angry",
                    "surprised": "surprised", "disgusted": "disgusted",
                    "confused": "confused", "calm": "calm", "fear": "fear"
                }.get(emotion, "neutral")
            else:
                emotion = "neutral"

            bbox = face.get("BoundingBox", {})
            people_data.append({
                "type": person_type,
                "age": age,
                "age_range": f"{face['AgeRange']['Low']}-{face['AgeRange']['High']}",
                "gender": gender,
                "emotion": emotion,
                "confidence": confidence,
                "bbox": [
                    bbox.get("Left", 0.0),
                    bbox.get("Top", 0.0),
                    bbox.get("Width", 0.0),
                    bbox.get("Height", 0.0),
                ]
            })
    except Exception as e:
        print(f"[Rekognition-People] Error on {image_path}: {e}")
    return people_data


def analyze_frame_with_rekognition(image_path, min_confidence=75.0):
    if not rekognition_client:
        return []

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"[Error reading image] {image_path}: {e}")
        return []

    objects = []
    try:
        response = rekognition_client.detect_labels(
            Image={"Bytes": image_bytes}, MaxLabels=20, MinConfidence=min_confidence
        )
        for label in response.get("Labels", []):
            for instance in label.get("Instances", []):
                if instance.get("Confidence", 0) >= min_confidence:
                    bbox = instance.get("BoundingBox", {})
                    objects.append({
                        "Name": label["Name"],
                        "Confidence": instance["Confidence"],
                        "bbox": [
                            bbox.get("Left", 0.0),
                            bbox.get("Top", 0.0),
                            bbox.get("Width", 0.0),
                            bbox.get("Height", 0.0),
                        ],
                        "Parents": [p["Name"] for p in label.get("Parents", []) if "Name" in p]
                    })
    except Exception as e:
        print(f"[Rekognition-Objects] Error on {image_path}: {e}")
    return objects


def _process_single_frame_task(frame_idx, image_path, image_file_name):
    print(f"  Processing frame {frame_idx}: {image_file_name}")
    return {
        "frame": frame_idx,
        "timestamp": float(frame_idx),
        "people": analyze_people_with_rekognition(image_path, min_confidence=REKOGNITION_MIN_CONFIDENCE),
        "detected_objects": analyze_frame_with_rekognition(image_path, min_confidence=REKOGNITION_MIN_CONFIDENCE)
    }


def process_video(video_folder_path):
    video_name = os.path.basename(video_folder_path)
    frames_dir = os.path.join(video_folder_path, "frames")
    print(f"\n[▶] Processing: {video_name}")

    try:
        image_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))],
            key=lambda x: int("".join(filter(str.isdigit, x)))
        )
    except Exception as e:
        print(f"[Error reading frames in {frames_dir}]: {e}")
        return None

    result_frames_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_PER_VIDEO) as executor:
        futures = {
            executor.submit(
                _process_single_frame_task,
                idx,
                os.path.join(frames_dir, fname),
                fname
            ): idx
            for idx, fname in enumerate(image_files)
        }

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                result_frames_data.append(result)
            except Exception as e:
                print(f"  Error in frame {futures[future]}: {e}")

    result_frames_data.sort(key=lambda x: x["frame"])

    people_types = set()
    max_people = 0
    for frame in result_frames_data:
        p = frame.get("people", [])
        max_people = max(max_people, len(p))
        people_types.update(person.get("type", "unknown") for person in p)

    return {
        "video_name": f"{video_name}.mp4",
        "total_frames": len(image_files),
        "duration_seconds": float(len(image_files)),
        "summary": {
            "total_people": max_people,
            "people_types": sorted(list(people_types))
        },
        "frames": result_frames_data
    }


def main():
    print("--- Video Frame Analysis (AWS Rekognition Only) ---")
    print(f"Input video frames directory: {DATA_DIRECTORY}")
    print(f"Output JSON directory: {OUTPUT_DIRECTORY}")

    if not rekognition_client:
        print("AWS Rekognition client is not initialized. Exiting.")
        return

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    mode = input("Process 'all' videos or a 'specific' one? (all/specific): ").strip().lower()
    video_paths = []

    if mode == "specific":
        vname = input("Enter video folder name (e.g., video_001): ").strip()
        path = os.path.join(DATA_DIRECTORY, vname)
        if not os.path.isdir(path):
            print(f"[!] Directory not found: {path}")
            return
        video_paths.append(path)
    else:
        video_paths = [
            os.path.join(DATA_DIRECTORY, d)
            for d in os.listdir(DATA_DIRECTORY)
            if os.path.isdir(os.path.join(DATA_DIRECTORY, d)) and d.startswith("video_")
        ]
        video_paths.sort()

    for path in video_paths:
        video_key = os.path.basename(path)
        output_path = os.path.join(OUTPUT_DIRECTORY, f"{video_key}_analysis.json")
        if os.path.exists(output_path):
            print(f"[⏩] Skipping {video_key} (already processed).")
            continue

        result = process_video(path)
        if result:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[✅] Saved: {output_path}")
        else:
            print(f"[❌] Failed to process {video_key}")


if __name__ == "__main__":
    main()
