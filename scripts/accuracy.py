import os
import sys
import json
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import time

# --- DYNAMIC PATH CONFIGURATION ---
# This makes the script runnable from anywhere, as long as the project structure is consistent.
# The script is in CSE493_project/scripts, so we go up one level to get the project root.
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
except NameError:
    # Fallback for interactive environments (like Jupyter)
    SCRIPT_DIR = os.getcwd()
    # You might need to adjust this if running interactively
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

sys.path.append(PROJECT_ROOT) # Allows finding modules in the project root if needed

# --- CONFIGURATION ---

# 1. File Paths (now relative to the dynamic PROJECT_ROOT)
GROUND_TRUTH_FILE = os.path.join(PROJECT_ROOT, "data", "per_frame_analysis.jsonl")
MODELS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "qwen2.5-output")

MODELS_CONFIG = {
    "Qwen2.5-VL-3B": os.path.join(MODELS_OUTPUT_DIR, "Qwen_Qwen2.5-VL-3B-Instruct", "qwen2.5-3b-instruct_combined.jsonl"),
    "Qwen2.5-VL-7B": os.path.join(MODELS_OUTPUT_DIR, "Qwen_Qwen2.5-VL-7B-Instruct", "qwen2.5-7b-instruct_combined.jsonl"),
    "Qwen2.5-VL-32B": os.path.join(MODELS_OUTPUT_DIR, "Qwen_Qwen2.5-VL-32B-Instruct", "qwen2.5-32b-instruct_combined.jsonl"),
}

# 2. Comparison Parameters
IOU_THRESHOLD = 0.5
AGE_TOLERANCE = 3
SEMANTIC_SIMILARITY_THRESHOLD = 0.8

# 3. Output Files (will be saved in a new 'reports' directory in the project root)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the directory if it doesn't exist

EMBEDDING_CACHE_FILE = os.path.join(OUTPUT_DIR, "embeddings_cache.json")
DETAILED_RESULTS_FILE = os.path.join(OUTPUT_DIR, "comparison_results.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, "comparison_report.txt")

# 4. Azure AI Setup
# load_dotenv will search up from the script's directory and find the .env in the project root
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# --- HELPER FUNCTIONS (No changes needed in the functions below) ---

def load_jsonl_to_dict(file_path):
    """Loads a JSONL file into a dictionary keyed by the 'id' field."""
    data_dict = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'id' in record:
                        data_dict[record['id']] = record
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    return data_dict

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) for two bounding boxes.
    Bbox format: [x_center, y_center, width, height]
    """
    box1_x1, box1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    box1_x2, box1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    box2_x2, box2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x1, inter_y1 = max(box1_x1, box2_x1), max(box1_y1, box2_y1)
    inter_x2, inter_y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    vec1, vec2 = np.array(v1), np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embeddings_with_cache(texts, cache_file):
    """
    Gets embeddings for a list of texts, using a cache to avoid redundant API calls.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    texts_to_fetch = [text for text in set(texts) if text not in cache]

    if texts_to_fetch:
        print(f"Fetching {len(texts_to_fetch)} new embeddings from Azure AI...")
        try:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2023-05-15"
            )
            response = client.embeddings.create(input=texts_to_fetch, model=AZURE_DEPLOYMENT_NAME)
            for text, embedding_data in zip(texts_to_fetch, response.data):
                cache[text] = embedding_data.embedding

            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print("Embedding cache updated.")
        except Exception as e:
            print(f"An error occurred with Azure API: {e}")
            print("Proceeding without new embeddings. Results for new objects may be inaccurate.")
    return cache

def calculate_metrics(tp, fp, fn):
    """Calculate Precision, Recall, F1-Score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# --- MAIN ANALYSIS LOGIC (No changes needed) ---

def analyze_model(model_name, gt_data, model_data, embeddings_cache):
    """Performs a full analysis of one model against the ground truth."""
    print(f"\n--- Analyzing {model_name} ---")

    people_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'correct_type': 0, 'correct_age': 0, 'correct_gender': 0, 'correct_emotion': 0, 'total_matched': 0}
    object_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'correct_name_semantic': 0, 'total_matched': 0}

    for frame_id, gt_frame in gt_data.items():
        if frame_id not in model_data:
            people_stats['fn'] += len(gt_frame.get('people', []))
            object_stats['fn'] += len(gt_frame.get('detected_objects', []))
            continue

        pred_frame = model_data[frame_id]

        gt_people = gt_frame.get('people', [])
        pred_people = pred_frame.get('people', [])
        people_matches = find_best_matches(gt_people, pred_people)

        people_stats['tp'] += len(people_matches)
        people_stats['fn'] += len(gt_people) - len(people_matches)
        people_stats['fp'] += len(pred_people) - len(people_matches)

        for gt_idx, pred_idx, _ in people_matches:
            people_stats['total_matched'] += 1
            gt_person, pred_person = gt_people[gt_idx], pred_people[pred_idx]
            if gt_person.get('type') == pred_person.get('type'): people_stats['correct_type'] += 1
            if abs(gt_person.get('age', -99) - pred_person.get('age', -99)) <= AGE_TOLERANCE: people_stats['correct_age'] += 1
            if gt_person.get('gender') == pred_person.get('gender'): people_stats['correct_gender'] += 1
            if gt_person.get('emotion') == pred_person.get('emotion'): people_stats['correct_emotion'] += 1

        gt_objects = gt_frame.get('detected_objects', [])
        pred_objects = pred_frame.get('detected_objects', [])
        object_matches = find_best_matches(gt_objects, pred_objects)

        object_stats['tp'] += len(object_matches)
        object_stats['fn'] += len(gt_objects) - len(object_matches)
        object_stats['fp'] += len(pred_objects) - len(object_matches)

        for gt_idx, pred_idx, _ in object_matches:
            object_stats['total_matched'] += 1
            gt_obj, pred_obj = gt_objects[gt_idx], pred_objects[pred_idx]
            gt_name, pred_name = gt_obj.get('Name', '').lower(), pred_obj.get('Name', '').lower()
            if gt_name in embeddings_cache and pred_name in embeddings_cache:
                sim = cosine_similarity(embeddings_cache[gt_name], embeddings_cache[pred_name])
                if sim >= SEMANTIC_SIMILARITY_THRESHOLD:
                    object_stats['correct_name_semantic'] += 1

    return {'people_stats': people_stats, 'object_stats': object_stats}

def find_best_matches(gt_list, pred_list):
    """Finds the best IoU matches between two lists of detections."""
    matches, used_preds = [], set()
    for i, gt_item in enumerate(gt_list):
        best_iou, best_pred_idx = -1, -1
        for j, pred_item in enumerate(pred_list):
            if j in used_preds: continue
            iou = calculate_iou(gt_item['bbox'], pred_item['bbox'])
            if iou > best_iou:
                best_iou, best_pred_idx = iou, j
        if best_iou > IOU_THRESHOLD:
            matches.append((i, best_pred_idx, best_iou))
            used_preds.add(best_pred_idx)
    return matches

def generate_report(all_results):
    """Creates a formatted text report from the analysis results."""
    # This function remains unchanged
    report_lines = ["="*80, "      IMAGE ANNOTATION MODEL COMPARISON REPORT", "="*80, "\n"]
    for model_name, results in all_results.items():
        report_lines.append(f"\n--- MODEL: {model_name} ---\n")
        p_stats = results['people_stats']
        p_precision, p_recall, p_f1 = calculate_metrics(p_stats['tp'], p_stats['fp'], p_stats['fn'])
        report_lines.append("  PEOPLE ANALYSIS:")
        report_lines.append(f"    - Detection Metrics (IoU > {IOU_THRESHOLD}):")
        report_lines.append(f"      - Precision: {p_precision:.2%}, Recall: {p_recall:.2%}, F1-Score: {p_f1:.2%}")
        report_lines.append(f"      - (TP: {p_stats['tp']}, FP: {p_stats['fp']}, FN: {p_stats['fn']})\n")
        if p_stats['total_matched'] > 0:
            total = p_stats['total_matched']
            report_lines.append("    - Attribute Accuracy (for matched people):")
            report_lines.append(f"      - Type: {p_stats['correct_type']/total:.2%}, Age (±{AGE_TOLERANCE}): {p_stats['correct_age']/total:.2%}, Gender: {p_stats['correct_gender']/total:.2%}, Emotion: {p_stats['correct_emotion']/total:.2%}")
        report_lines.append("")
        o_stats = results['object_stats']
        o_precision, o_recall, o_f1 = calculate_metrics(o_stats['tp'], o_stats['fp'], o_stats['fn'])
        report_lines.append("  OBJECT ANALYSIS:")
        report_lines.append(f"    - Detection Metrics (IoU > {IOU_THRESHOLD}):")
        report_lines.append(f"      - Precision: {o_precision:.2%}, Recall: {o_recall:.2%}, F1-Score: {o_f1:.2%}")
        report_lines.append(f"      - (TP: {o_stats['tp']}, FP: {o_stats['fp']}, FN: {o_stats['fn']})\n")
        if o_stats['total_matched'] > 0:
            total = o_stats['total_matched']
            report_lines.append("    - Classification Accuracy (for matched objects):")
            report_lines.append(f"      - Semantic Name (Sim > {SEMANTIC_SIMILARITY_THRESHOLD}): {o_stats['correct_name_semantic']/total:.2%}")
        report_lines.append("\n" + "-"*60)
    report_lines.append("\nReport generated successfully.")
    return "\n".join(report_lines)


# --- MAIN EXECUTION (No changes needed) ---
if __name__ == "__main__":
    start_time = time.time()

    print("Project Root:", PROJECT_ROOT)
    print("Loading ground truth data...")
    gt_data = load_jsonl_to_dict(GROUND_TRUTH_FILE)
    if not gt_data: exit("Failed to load ground truth data. Exiting.")

    print("Loading model prediction data...")
    all_model_data = {name: load_jsonl_to_dict(path) for name, path in MODELS_CONFIG.items()}
    all_model_data = {k: v for k, v in all_model_data.items() if v is not None}
    if not all_model_data: exit("Failed to load any model data. Exiting.")

    print("Collecting unique object names for semantic comparison...")
    all_object_names = set(obj.get('Name', '').lower() for frame in gt_data.values() for obj in frame.get('detected_objects', []))
    for model_data in all_model_data.values():
        all_object_names.update(obj.get('Name', '').lower() for frame in model_data.values() for obj in frame.get('detected_objects', []))
    all_object_names.discard('')

    embeddings_cache = get_embeddings_with_cache(list(all_object_names), EMBEDDING_CACHE_FILE)

    all_results = {name: analyze_model(name, gt_data, data, embeddings_cache) for name, data in all_model_data.items()}

    print(f"\nSaving detailed results to {DETAILED_RESULTS_FILE}...")
    with open(DETAILED_RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Generating and saving text report to {REPORT_FILE}...")
    report_content = generate_report(all_results)
    with open(REPORT_FILE, 'w') as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"\nAnalysis complete. Total time taken: {time.time() - start_time:.2f} seconds.")