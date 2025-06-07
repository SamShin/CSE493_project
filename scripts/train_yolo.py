from pathlib import Path

import torch
from ultralytics import YOLO

# --- Configuration ---
project_root = Path(__file__).parent.parent
dataset_yaml_path = str(project_root / 'yolo_dataset' / 'dataset.yaml')
model_to_finetune = 'yolov8n.pt'

# --- Training Hyperparameters ---
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16  # Good starting value for Colab; lower if OOM
LEARNING_RATE = 0.01
OPTIMIZER = 'AdamW'
PATIENCE = 20
WORKERS = 2
PROJECT_NAME = 'yolo_colab_runs'
RUN_NAME = 'finetune_colab'

# --- Determine Device (CUDA only) ---
if torch.cuda.is_available():
    DEVICE = 0  # CUDA: use first GPU
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Using CUDA device: {DEVICE}")
else:
    DEVICE = 'cpu'
    print("No CUDA GPU found. Training on CPU (very slow).")

def train_model():
    print("--- YOLOv8 Custom Fine-tuning on GPU ---")
    print(f"Project Root: {project_root}")
    print(f"Dataset YAML: {dataset_yaml_path}")
    if not Path(dataset_yaml_path).exists():
        print(f"ERROR: Dataset YAML not found at {dataset_yaml_path}")
        return

    print(f"Pretrained Model: {model_to_finetune}")
    print(f"Epochs: {EPOCHS}, Image Size: {IMG_SIZE}, Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}, Optimizer: {OPTIMIZER}, Patience: {PATIENCE}")
    print(f"Workers: {WORKERS}")
    print(f"Using device: {DEVICE}")

    model = YOLO(model_to_finetune)
    print(f"\nLoaded pretrained model: {model_to_finetune}")
    print(f"Training device set to: {DEVICE}")

    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            project=str(project_root / 'runs' / 'train' / PROJECT_NAME),
            name=RUN_NAME,
            optimizer=OPTIMIZER,
            lr0=LEARNING_RATE,
            patience=PATIENCE,
        )
        print("\n--- Training Complete ---")
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
            print(f"Training run artifacts saved in: {model.trainer.save_dir}")
        else:
            print("Training completed. Check your output directory for results.")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        if "out of memory" in str(e).lower():
            print("\nRECOMMENDATION: Out Of Memory (OOM) error. Reduce BATCH_SIZE or IMG_SIZE.")

if __name__ == '__main__':
    train_model()
