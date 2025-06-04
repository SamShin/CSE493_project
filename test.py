import json

def transform_video_jsonl_to_frames_custom_id(input_jsonl_content):
    """
    Transforms a JSONL string where each line is a video object
    into a JSONL string where each line is a frame object.
    The 'frame' and 'timestamp' keys are replaced with a unique 'id'
    formatted as 'videoNameWithoutExtension_frame_0000'.

    Args:
        input_jsonl_content (str): A string containing the input JSONL data,
                                   with each video object on a new line.

    Returns:
        str: A string containing the output JSONL data,
             with each frame object (with its new unique ID) on a new line.
    """
    output_lines = []
    input_lines = input_jsonl_content.strip().split('\n')

    for line_number, line in enumerate(input_lines):
        if not line.strip():
            continue  # Skip empty lines
        try:
            video_data = json.loads(line)
            video_name = video_data.get("video_name")

            if not video_name:
                print(f"Warning: 'video_name' missing in line {line_number + 1}. Cannot create unique ID. Skipping video.")
                continue

            # Remove .mp4 extension from video_name
            if video_name.endswith(".mp4"):
                base_video_name = video_name[:-4]  # Slice off the last 4 characters
            else:
                base_video_name = video_name # Use as is if .mp4 is not at the end

            if "frames" in video_data and isinstance(video_data["frames"], list):
                for frame_object in video_data["frames"]:
                    original_frame_number_val = frame_object.get("frame")

                    if original_frame_number_val is None: # Check for None, as 0 is a valid frame number
                        print(f"Warning: 'frame' number missing in a frame from video '{video_name}'. Skipping this frame.")
                        continue

                    try:
                        # Ensure frame number is an integer for formatting
                        frame_num_int = int(original_frame_number_val)
                    except ValueError:
                        print(f"Warning: 'frame' number '{original_frame_number_val}' is not a valid integer in video '{video_name}'. Skipping this frame.")
                        continue

                    # Format frame number to 4 digits with leading zeros
                    formatted_frame_number = f"{frame_num_int:04d}"

                    # Create the unique ID
                    unique_id = f"{base_video_name}_frame_{formatted_frame_number}"

                    # Remove original 'frame' and 'timestamp' keys
                    frame_object.pop("frame", None)
                    frame_object.pop("timestamp", None)

                    # Create the new frame object with 'id' as the first key
                    frame_object_with_id = {"id": unique_id}
                    frame_object_with_id.update(frame_object) # Add remaining keys

                    output_lines.append(json.dumps(frame_object_with_id))
            else:
                print(f"Warning: 'frames' key missing or not a list in video '{video_name}' (line {line_number + 1}). Skipping.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number + 1}: {e}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred processing line {line_number + 1}: {e}. Skipping.")

    return "\n".join(output_lines)

# --- Example Usage with your provided data snippet ---

# Simulating the input JSONL content from your example
# (In a real scenario, you would read this from a file)
# Transform the data
# output_jsonl_content = transform_video_jsonl_to_frames_jsonl(input_data)

# Print the result (or write to a new file)
# print(output_jsonl_content)

# To use with files:
#
with open("combined_analysis.jsonl", "r") as infile:
    input_content = infile.read()
#
output_content = transform_video_jsonl_to_frames_custom_id(input_content)
#
with open("output_frames.jsonl", "w") as outfile:
    outfile.write(output_content)
#
print("Transformation complete. Output written to output_frames.jsonl")