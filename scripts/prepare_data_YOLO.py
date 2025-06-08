import json
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml

# --- Configuration ---
# Paths are now relative to the project root, assuming the script is in a 'scripts' subdirectory.
project_root = Path(__file__).parent.parent # Goes up one level from 'scripts' to project root

jsonl_file_path = project_root / 'data\\per_frame_analysis.jsonl'
base_data_dir = project_root / 'videos'           # Base directory containing video_XXX folders
dataset_root = project_root / 'yolo_dataset'    # Where the YOLO formatted dataset will be created
image_extension = '.jpg'                        # Or .png, or whatever your images are

# Create dataset directories (these paths are relative to where dataset_root points)
img_train_dir = dataset_root / 'images' / 'train'
lbl_train_dir = dataset_root / 'labels' / 'train'
img_val_dir = dataset_root / 'images' / 'val'
lbl_val_dir = dataset_root / 'labels' / 'val'

for p in [img_train_dir, lbl_train_dir, img_val_dir, lbl_val_dir]:
    p.mkdir(parents=True, exist_ok=True)

# --- 1. Parse JSONL and Extract Class Names ---
all_data_from_jsonl = []
class_names = set()
object_counts = {} # To count instances per class

print(f"Reading JSONL file: {jsonl_file_path}")
if not jsonl_file_path.exists():
    print(f"Error: JSONL file not found at {jsonl_file_path}")
    print("Please ensure the path is correct and the file exists at the project root.")
    exit()

with open(jsonl_file_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            all_data_from_jsonl.append(data)
            for obj in data.get('detected_objects', []):
                class_name = obj['Name']
                class_names.add(class_name)
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
            continue

class_list = sorted(list(class_names))
class_to_id = {name: i for i, name in enumerate(class_list)}

print("\n--- Class Summary ---")
print(f"Found {len(class_list)} unique classes: {class_list}")
print("Instance counts per class:")
for name, count in sorted(object_counts.items(), key=lambda item: item[1], reverse=True):
    print(f"- {name}: {count}")

if not class_list:
    print("No classes found. Exiting.")
    exit()

# --- 2. Filter data to include only frames for which we have images & Split ---
if not base_data_dir.exists():
    print(f"Error: Base data directory '{base_data_dir}' not found!")
    print("Please ensure the 'data' directory exists at the project root.")
    exit()

frame_entries_with_images = []
print("\nLocating image files...")
for data_entry in all_data_from_jsonl:
    frame_id = data_entry['id'] # e.g., "video_001_frame_0000"

    try:
        parts = frame_id.split('_frame_')
        if len(parts) != 2:
            print(f"Warning: Malformed frame_id '{frame_id}'. Expected 'video_XXX_frame_YYYY'. Skipping.")
            continue
        video_folder_name = parts[0]
        frame_number_str = parts[1]

        image_actual_name_in_source = f"frame_{frame_number_str}{image_extension}"
        original_image_path = base_data_dir / video_folder_name / 'frames' / image_actual_name_in_source

        if original_image_path.exists():
            data_entry['original_image_path'] = original_image_path
            frame_entries_with_images.append(data_entry)
        else:
            print(f"Warning: Image file not found at {original_image_path} for ID {frame_id}. Skipping this entry.")
    except Exception as e:
        print(f"Error processing frame_id '{frame_id}': {e}. Skipping.")
        continue

if not frame_entries_with_images:
    print("No valid image frames found based on the JSONL entries and directory structure. Exiting.")
    exit()

print(f"\nProcessing {len(frame_entries_with_images)} frames with available images.")

train_data, val_data = train_test_split(frame_entries_with_images, test_size=0.2, random_state=42)
print(f"Splitting into {len(train_data)} training samples and {len(val_data)} validation samples.")

# --- 3. Process and Write Labels ---
def process_data_split(data_split, img_dir, lbl_dir, split_name):
    print(f"\nProcessing {split_name} data...")
    frames_processed_count = 0
    for data_entry in data_split:
        frame_id = data_entry['id']

        yolo_image_file_name = f"{frame_id}{image_extension}"
        yolo_label_file_name = f"{frame_id}.txt"

        original_image_path = data_entry['original_image_path']
        yolo_image_path = img_dir / yolo_image_file_name
        yolo_label_path = lbl_dir / yolo_label_file_name

        shutil.copy(original_image_path, yolo_image_path)

        with open(yolo_label_path, 'w') as lf:
            for obj in data_entry.get('detected_objects', []):
                class_name = obj['Name']
                if class_name not in class_to_id:
                    print(f"Warning: Class '{class_name}' in frame {frame_id} not in master class list. Skipping object.")
                    continue

                class_id = class_to_id[class_name]
                bbox = obj['bbox']
                x_min, y_min, w, h = bbox

                x_center = x_min + (w / 2)
                y_center = y_min + (h / 2)

                if not all(0.0 <= val <= 1.0 for val in [x_center, y_center, w, h]):
                    print(f"Warning: Invalid bbox for {class_name} in {frame_id}: {[x_center, y_center, w, h]}. Skipping object.")
                    continue
                if w <= 0 or h <= 0:
                    print(f"Warning: Non-positive width/height for {class_name} in {frame_id}: w={w}, h={h}. Skipping object.")
                    continue

                lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        frames_processed_count +=1
    print(f"Finished processing {frames_processed_count} frames for {split_name}.")

process_data_split(train_data, img_train_dir, lbl_train_dir, "training")
process_data_split(val_data, img_val_dir, lbl_val_dir, "validation")


# --- 4. Create dataset.yaml ---
# 'path' in dataset.yaml should be the absolute path to the yolo_dataset directory
# 'train' and 'val' paths are relative to this 'path'
dataset_yaml_content = {
    'path': str(dataset_root.resolve()),
    'train': (Path('images') / 'train').as_posix(),  # Corrected: .as_posix() on Path object, comma added
    'val': (Path('images') / 'val').as_posix(),    # Corrected: .as_posix() on Path object, comma added
    'names': {v: k for k, v in class_to_id.items()}   # YOLO expects {id: 'name'}
}

yaml_path = dataset_root / 'dataset.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_yaml_content, f, sort_keys=False) # sort_keys=False to maintain order for integer keys

print(f"\nDataset preparation complete. YOLO dataset created at: {dataset_root.resolve()}")
print(f"YAML file created at: {yaml_path.resolve()}")
print("Please verify the paths in dataset.yaml, especially if you move the dataset folder.")
print("\nNext steps: Install ultralytics and run YOLOv8 training from your project root.")
print("Example training command (run from project_root, not from scripts/):")
# The data path in the command should be relative to where you run 'yolo train'
# Or use the absolute path from yaml_path.resolve()
print(f"  yolo train model=yolov8n.pt data={str(yaml_path.relative_to(project_root)).replace(os.sep, '/')} epochs=100 imgsz=640 batch=16 device=0")
# Or, more robustly using the absolute path:
# print(f"  yolo train model=yolov8n.pt data={yaml_path.resolve()} epochs=100 imgsz=640 batch=16 device=0")

