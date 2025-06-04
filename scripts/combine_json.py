import os
import json

# --- Configuration ---
INPUT_DIR = "analysis_results"  # Directory containing individual .json analysis files
FINAL_FRAMES_FILE = "output_frames.jsonl" # Final output with one frame per line

# --- Helper Function to process a single video object ---
def process_video_object_to_frame_json_strings(video_data):
    """
    Transforms a single video data object (Python dictionary) into a list of
    JSON strings, where each string is a processed frame with a custom ID.

    Args:
        video_data (dict): A Python dictionary representing one video's analysis.

    Returns:
        list: A list of JSON strings, each representing a processed frame.
              Returns an empty list if errors occur or no frames are found.
    """
    frame_json_strings = []
    video_name = video_data.get("video_name")

    if not video_name:
        print(f"Warning: 'video_name' missing in a video object. Cannot create unique ID. Skipping this video's frames.")
        return []

    # Remove .mp4 extension from video_name
    if video_name.endswith(".mp4"):
        base_video_name = video_name[:-4]
    else:
        base_video_name = video_name

    if "frames" in video_data and isinstance(video_data["frames"], list):
        for frame_object in video_data["frames"]:
            current_frame_data = frame_object.copy()
            original_frame_number_val = current_frame_data.get("frame")

            if original_frame_number_val is None:
                print(f"Warning: 'frame' number missing in a frame from video '{video_name}'. Skipping this frame.")
                continue

            try:
                frame_num_int = int(original_frame_number_val)
            except ValueError:
                print(f"Warning: 'frame' number '{original_frame_number_val}' is not a valid integer in video '{video_name}'. Skipping this frame.")
                continue

            formatted_frame_number = f"{frame_num_int:04d}"
            unique_id = f"{base_video_name}_frame_{formatted_frame_number}"

            current_frame_data.pop("frame", None)
            current_frame_data.pop("timestamp", None)

            frame_with_id = {"id": unique_id}
            frame_with_id.update(current_frame_data)

            frame_json_strings.append(json.dumps(frame_with_id))
    else:
        print(f"Warning: 'frames' key missing or not a list in video '{video_name}'. No frames processed for this video.")

    return frame_json_strings

# --- Main Script Logic ---

print(f"Processing JSON files from '{INPUT_DIR}' and writing frames to '{FINAL_FRAMES_FILE}'...")

if not os.path.exists(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' does not exist. Exiting.")
    exit()

# Open the final output file once
with open(FINAL_FRAMES_FILE, "w", encoding="utf-8") as outfile:
    files_in_dir = sorted(os.listdir(INPUT_DIR))
    if not files_in_dir:
        print(f"Warning: Input directory '{INPUT_DIR}' is empty.")

    total_frames_written = 0
    for fname in files_in_dir:
        if fname.endswith(".json"):
            file_path = os.path.join(INPUT_DIR, fname)
            print(f"Processing file: {fname}")
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    video_obj = json.load(infile) # Each .json file contains one video object

                # Process this single video object to get its frame strings
                processed_frame_strings = process_video_object_to_frame_json_strings(video_obj)

                # Write each processed frame string to the output file
                for frame_line in processed_frame_strings:
                    outfile.write(frame_line + "\n")
                    total_frames_written += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file '{file_path}': {e}. Skipping this file.")
            except Exception as e:
                print(f"An unexpected error occurred processing file '{file_path}': {e}. Skipping this file.")

print(f"\nProcessing complete. {total_frames_written} frames written to '{FINAL_FRAMES_FILE}'.")