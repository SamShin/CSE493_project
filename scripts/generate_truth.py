import base64
import concurrent.futures
import json
import os

import boto3
import dotenv
from openai import AzureOpenAI

# --- Determine Project Root and Paths ---
# Get the directory of the currently running script (e.g., Project_Root/Scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from SCRIPT_DIR, e.g., Project_Root)
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Path to the .env file in the project root
DOTENV_PATH = os.path.join(PROJECT_ROOT_DIR, ".env")
# Load environment variables from .env file in the project root
dotenv.load_dotenv(dotenv_path=DOTENV_PATH)
print(f"Attempting to load .env from: {DOTENV_PATH}")
if os.path.exists(DOTENV_PATH):
    print(".env file found and loaded.")
else:
    print(f"Warning: .env file not found at {DOTENV_PATH}. Environment variables might not be set.")


# --- Configuration ---
# AWS Configuration
AWS_PROFILE_NAME = "personal"
AWS_REGION = "us-west-1"
REKOGNITION_MIN_CONFIDENCE = 75.0

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_API_KEY = str(os.getenv("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_DEPLOYMENT = str(os.getenv("AZURE_DEPLOYMENT_NAME"))
AZURE_API_VERSION = str(os.getenv("AZURE_API_VERSION", "2024-02-01"))

# Directory Configuration (now relative to PROJECT_ROOT_DIR)
DATA_DIRECTORY = os.path.join(PROJECT_ROOT_DIR, "videos")
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT_DIR, "data\\per_video_analysis")

# Image Analysis Configuration
IMAGE_DETAIL = "high"  # For OpenAI vision
OPENAI_MAX_TOKENS = 1000 # Increased for potentially larger JSON with bboxes

# Parallelism Configuration
MAX_WORKERS_PER_VIDEO = (
    10  # Number of frames to process in parallel for a single video. Adjust as needed.
)

# --- Initialize Clients ---
# Initialize AWS Rekognition
rekognition_client = None
try:
    if AWS_PROFILE_NAME and AWS_REGION:
        session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION)
        rekognition_client = session.client("rekognition")
        print("AWS Rekognition client initialized.")
    else:
        print(
            "Warning: AWS_PROFILE_NAME or AWS_REGION is not configured. AWS Rekognition features will be skipped."
        )
except Exception as e:
    print(
        f"Warning: Could not initialize AWS Rekognition client: {e}. Rekognition features will be skipped."
    )
    rekognition_client = None

# Initialize Azure OpenAI
azure_client = None
if not (
    not AZURE_OPENAI_ENDPOINT
    or AZURE_OPENAI_ENDPOINT == "None"
    or not AZURE_OPENAI_API_KEY
    or AZURE_OPENAI_API_KEY == "None"
    or not AZURE_OPENAI_DEPLOYMENT
    or AZURE_OPENAI_DEPLOYMENT == "None"
    or not AZURE_API_VERSION
    or AZURE_API_VERSION == "None"
):
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
        )
        print("Azure OpenAI client initialized.")
    except Exception as e:
        print(
            f"Critical Error: Could not initialize Azure OpenAI client: {e}\nExiting."
        )
        exit()
else:
    print(
        "Critical Error: Azure OpenAI client could not be initialized due to missing .env configuration.\nExiting."
    )
    if __name__ == "__main__": # Keep this check for direct execution context
        pass
    else: # If imported and critical config missing, still exit
        exit()


def analyze_frame_with_openai(image_path, detail="low"):
    if not azure_client:
        print(
            f"    [OpenAI] Client not initialized. Skipping for {os.path.basename(image_path)}"
        )
        return []
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"    [OpenAI] Error: File not found {image_path}")
        return []
    except Exception as e:
        print(f"    [OpenAI] Error reading/encoding {image_path}: {e}")
        return []

    prompt = """Analyze this image and identify all people. For each person, provide:
    - type: "baby" (0-2 years), "child" (3-12), "teenager" (13-19), or "adult" (20+)
    - age: estimated age as a number
    - age_range: format as "X-Y" (e.g., "25-35", "0-2")
    - gender: "male", "female", or "unknown"
    - emotion: primary emotion (happy, sad, neutral, calm, e xcited, angry, etc.)

    Return ONLY a JSON array of people objects. If no people are present, return an empty array [].
    Example: [{"type": "adult", "age": 30, "age_range": "25-35", "gender": "female", "emotion": "happy"}]"""

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": detail,
                            },
                        },
                    ],
                } # type: ignore
            ],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=0.0,
        )
        raw_response_content = response.choices[0].message.content
    except Exception as e:
        print(f"    [OpenAI] API Error for '{os.path.basename(image_path)}': {e}")
        return []

    people = []
    try:
        if raw_response_content is None:
            print(f"    [OpenAI] Error: Received None response content for '{os.path.basename(image_path)}'")
            return []
        parsed_data = json.loads(raw_response_content) # type: ignore
        if isinstance(parsed_data, list):
            # Basic validation for bbox if present
            for person_obj in parsed_data:
                if "bbox" in person_obj and person_obj["bbox"] is not None:
                    if not (isinstance(person_obj["bbox"], list) and len(person_obj["bbox"]) == 4 and all(isinstance(n, (int, float)) for n in person_obj["bbox"])):
                        print(f"    [OpenAI] Warning: Malformed bbox for a person in '{os.path.basename(image_path)}': {person_obj['bbox']}. Setting to null.")
                        person_obj["bbox"] = None
            people = parsed_data
        else:
            print(
                f"    [OpenAI] Warning: Parsed response not a list for '{os.path.basename(image_path)}'. Content: {parsed_data}"
            )
    except json.JSONDecodeError as e:
        print(
            f"    [OpenAI] JSON Decode Error for '{os.path.basename(image_path)}': {e}. Content: {raw_response_content}"
        )
    except Exception as e:
        print(
            f"    [OpenAI] Unexpected parse error for '{os.path.basename(image_path)}': {e}. Content: {raw_response_content}"
        )
    return people


def analyze_frame_with_rekognition(image_path, min_confidence=75.0):
    if not rekognition_client:
        print(
            f"    [Rekognition] Client not initialized. Skipping for {os.path.basename(image_path)}"
        )
        return []
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    except FileNotFoundError:
        print(f"    [Rekognition] Error: File not found {image_path}")
        return []
    except Exception as e:
        print(f"    [Rekognition] Error reading image {image_path}: {e}")
        return []

    try:
        response = rekognition_client.detect_labels(
            Image={"Bytes": image_bytes}, MaxLabels=20, MinConfidence=min_confidence
        )
    except Exception as e:
        print(f"    [Rekognition] API Error for '{os.path.basename(image_path)}': {e}")
        return []

    detected_objects = []
    if "Labels" in response:
        for label in response["Labels"]:
            # Process instances for bounding boxes
            if label.get("Instances"):
                for instance in label["Instances"]:
                    if "BoundingBox" in instance and "Confidence" in instance:
                        # Check if instance confidence meets the general min_confidence
                        # (Rekognition API applies MinConfidence to labels, not always to instances directly in older versions, but good practice)
                        if instance["Confidence"] >= min_confidence:
                            bbox_rek = instance["BoundingBox"]
                            detected_objects.append({
                                "Name": label["Name"],
                                "Confidence": instance["Confidence"],
                                "bbox": [ # Standardized format [x, y, w, h] (Left, Top, Width, Height)
                                    bbox_rek.get("Left", 0.0),
                                    bbox_rek.get("Top", 0.0),
                                    bbox_rek.get("Width", 0.0),
                                    bbox_rek.get("Height", 0.0)
                                ],
                                "Parents": [p.get("Name") for p in label.get("Parents", []) if p.get("Name")]
                            })
            # Optionally, include labels without instances if needed (e.g., general scene labels)
            # else:
            #     if label["Confidence"] >= min_confidence: # Check top-level label confidence
            #         detected_objects.append({
            #             "Name": label["Name"],
            #             "Confidence": label["Confidence"],
            #             "bbox": None, # No specific bounding box for this instance-less label
            #             "Parents": [p.get("Name") for p in label.get("Parents", []) if p.get("Name")]
            #         })
    else:
        print(
            f"    [Rekognition] No 'Labels' in response for '{os.path.basename(image_path)}'."
        )
    return detected_objects


def _process_single_frame_task(frame_idx, image_path, image_file_name):
    """Helper function to process a single frame using both services."""
    print(f"    Starting analysis for frame {frame_idx}: {image_file_name}")
    people_data = analyze_frame_with_openai(image_path, detail=IMAGE_DETAIL)
    object_details_data = analyze_frame_with_rekognition(
        image_path, min_confidence=REKOGNITION_MIN_CONFIDENCE
    )
    print(
        f"    Finished analysis for frame {frame_idx}: {image_file_name}. OpenAI: {len(people_data)} people, Rekognition: {len(object_details_data)} objects."
    )
    return {
        "frame": frame_idx,
        "timestamp": float(frame_idx),  # Assuming 1fps
        "people": people_data, # Will contain bboxes from OpenAI
        "detected_objects": object_details_data, # Will contain bboxes from Rekognition
    }


def process_video(video_folder_path):
    video_name = os.path.basename(video_folder_path)
    frames_dir = os.path.join(video_folder_path, "frames")

    print(
        f"\nStarting processing for video: '{video_name}' (Parallelism: {MAX_WORKERS_PER_VIDEO} workers)"
    )
    if not os.path.isdir(frames_dir):
        print(f"Error: Frames directory '{frames_dir}' not found. Cannot proceed.")
        return None

    try:
        image_files = sorted(
            [
                f
                for f in os.listdir(frames_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ],
            key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
        )
    except Exception as e:
        print(f"Error reading/sorting files in '{frames_dir}': {e}")
        return None

    total_frames = len(image_files)
    if total_frames == 0:
        print(f"Warning: No image files found in '{frames_dir}'.")
    else:
        print(f"Found {total_frames} frames to process in '{frames_dir}'.")

    result_frames_data = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS_PER_VIDEO
    ) as executor:
        future_to_frame_idx = {
            executor.submit(
                _process_single_frame_task,
                idx,
                os.path.join(frames_dir, image_file),
                image_file,
            ): idx
            for idx, image_file in enumerate(image_files)
        }
        for future in concurrent.futures.as_completed(future_to_frame_idx):
            frame_idx = future_to_frame_idx[future]
            try:
                frame_data = future.result()
                result_frames_data.append(frame_data)
            except Exception as exc:
                print(f"    Frame {frame_idx} generated an exception: {exc}")
                result_frames_data.append(
                    {
                        "frame": frame_idx,
                        "timestamp": float(frame_idx),
                        "people": [],
                        "detected_objects": [], # Updated key name
                        "error": str(exc),
                    }
                )

    result_frames_data.sort(key=lambda x: x["frame"])

    all_people_types_in_video = set()
    max_people_in_any_frame = 0
    for frame_data in result_frames_data:
        people_in_frame = frame_data.get("people", [])
        if isinstance(people_in_frame, list):
            if len(people_in_frame) > max_people_in_any_frame:
                max_people_in_any_frame = len(people_in_frame)
            for person in people_in_frame:
                all_people_types_in_video.add(person.get("type", "unknown"))
        else:
            print(
                f"Warning: 'people' data for frame {frame_data.get('frame')} is not a list: {people_in_frame}"
            )

    final_result = {
        "video_name": f"{video_name}.mp4",
        "total_frames": total_frames,
        "duration_seconds": float(total_frames),
        "summary": {
            "total_people": max_people_in_any_frame,
            "people_types": sorted(
                list(all_people_types_in_video)
            ),
        },
        "frames": result_frames_data,
    }
    print(f"Finished processing for video: '{video_name}'.")
    return final_result


def main():
    print("--- Video Frame Analysis Script (OpenAI & AWS Rekognition) ---")
    print(f"Using DATA_DIRECTORY: {DATA_DIRECTORY}")
    print(f"Using OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")

    if not azure_client:
        print(
            "Exiting script as Azure OpenAI client is not available (check .env and initialization logs)."
        )
        return

    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        print(f"Output directory '{OUTPUT_DIRECTORY}' is ready.")
    except OSError as e:
        print(
            f"Error: Could not create output directory '{OUTPUT_DIRECTORY}': {e}. Exiting."
        )
        return

    while True:
        choice = (
            input(
                "Process 'all' video directories or a 'specific' one? (all/specific): "
            )
            .strip()
            .lower()
        )
        if choice in ["all", "specific"]:
            break
        print("Invalid choice. Please enter 'all' or 'specific'.")

    video_paths_to_process = []
    if choice == "specific":
        video_dir_name = input(
            f"Enter video folder name in '{DATA_DIRECTORY}' (e.g., video_001): "
        ).strip()
        if not video_dir_name:
            print("Error: No folder name. Exiting.")
            return
        target_video_path = os.path.join(DATA_DIRECTORY, video_dir_name)
        if not os.path.isdir(target_video_path):
            print(f"Error: Directory '{target_video_path}' not found. Exiting.")
            return
        video_paths_to_process.append(target_video_path)
    else:  # "all"
        print(
            f"Finding video directories in '{DATA_DIRECTORY}' starting with 'video_'..."
        )
        if not os.path.isdir(DATA_DIRECTORY):
            print(f"Error: Data directory '{DATA_DIRECTORY}' not found. Exiting.")
            return
        try:
            all_subdirs = [
                d
                for d in os.listdir(DATA_DIRECTORY)
                if os.path.isdir(os.path.join(DATA_DIRECTORY, d))
                and d.startswith("video_")
            ]
            if not all_subdirs:
                print(f"No 'video_*' directories found in '{DATA_DIRECTORY}'. Exiting.")
                return
            video_paths_to_process = [
                os.path.join(DATA_DIRECTORY, d) for d in sorted(all_subdirs)
            ]
            print(
                f"Found {len(video_paths_to_process)} video directories: {', '.join(os.path.basename(p) for p in video_paths_to_process)}"
            )
        except Exception as e:
            print(f"Error listing directories in '{DATA_DIRECTORY}': {e}. Exiting.")
            return

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for video_path in video_paths_to_process:
        video_name_out = os.path.basename(video_path)
        expected_output_file = os.path.join(
            OUTPUT_DIRECTORY, f"{video_name_out}_analysis.json"
        )

        if os.path.exists(expected_output_file):
            print(
                f"\nSkipping video '{video_name_out}': Analysis file '{expected_output_file}' already exists."
            )
            skipped_count += 1
            continue

        analysis_result = process_video(video_path)
        if analysis_result:
            try:
                with open(expected_output_file, "w") as f:
                    json.dump(analysis_result, f, indent=2)
                print(f"  Successfully saved results to: {expected_output_file}")
                processed_count += 1
            except IOError as e:
                print(f"  Error saving to {expected_output_file}: {e}")
                failed_count += 1
        else:
            print(f"No analysis result for '{video_name_out}'.")
            failed_count += 1

    print(f"\n--- Overall Analysis Complete ---")
    print(
        f"Successfully processed: {processed_count} video(s)."
    )
    print(
        f"Skipped (already analyzed): {skipped_count} video(s)."
    )
    print(
        f"Failed/No result: {failed_count} video(s)."
    )
    print(f"Results are in '{OUTPUT_DIRECTORY}'.")


if __name__ == "__main__":
    missing_vars = []
    if not AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_ENDPOINT == "None":
        missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY or AZURE_OPENAI_API_KEY == "None":
        missing_vars.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT == "None":
        missing_vars.append("AZURE_DEPLOYMENT_NAME")
    if not AZURE_API_VERSION or AZURE_API_VERSION == "None":
        missing_vars.append("AZURE_API_VERSION")

    if missing_vars:
        print(
            "=" * 70
            + f"\n!!! CRITICAL WARNING: Azure OpenAI Configuration Missing from .env file ({DOTENV_PATH}) !!!"
        )
        print("Required environment variables in '.env' file (expected at project root):")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nExample '.env' file content:\n" + "-" * 70)
        print(
            "AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/\nAZURE_OPENAI_API_KEY=your_actual_api_key_here"
        )
        print(
            "AZURE_DEPLOYMENT_NAME=your_gpt_model_deployment_name_here\nAZURE_API_VERSION=2024-02-01\n"
            + "-" * 70
        )
        print("Script will exit due to missing configuration.\n" + "=" * 70)
    elif not azure_client:
        print(
            "Azure OpenAI client failed to initialize despite .env variables appearing set. Check client init logs. Exiting."
        )
    else:
        main()