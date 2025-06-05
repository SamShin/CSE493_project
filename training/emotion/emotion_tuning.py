import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.transforms import ToPILImage
from facenet_pytorch import MTCNN

# Labels have a bunch of classes but we definetly cant support so many with small dataset
emotion_mapping = {
    'calm': 'calm',
    'neutral': 'calm',
    'sad': 'crying',
    'angry': 'crying',
    'upset': 'crying',
    'scared': 'crying',
    'serious': 'calm',
    'happy': 'laughing',
    'excited': 'laughing',
    'curious': 'calm',
}
emotions = ['calm', 'crying', 'laughing']
emotion_to_idx = {e: i for i, e in enumerate(emotions)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BABIES_PER_FRAME = 3

# Transforms from milestone to be applied only to crying and laughing since they are underrepresented
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomResizedCrop(160, scale=(0.9, 1.0)),
])

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_eval_augment = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class EmotionDataset(Dataset):
    def __init__(self, jsonl_path, frames_root, train=True, cache_path=None):
        self.samples = []
        self.train = train
        self.frames_root = frames_root
        self.to_pil = ToPILImage()
        self.mtcnn = MTCNN(image_size=160, margin=0, device=device)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if os.path.exists(cache_path):
                print(f"Loading cached dataset from {cache_path}")
                self.samples = torch.load(cache_path)
                return

        print("Processing dataset (this might take some time)...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                frame_id = data['id']
                people = data.get('people', [])
                video_id, frame_num = frame_id.split('_frame_')
                frame_path = os.path.join(frames_root, video_id, "frames", f"frame_{frame_num}.jpg")

                if not os.path.exists(frame_path):
                    continue

                try:
                    img = Image.open(frame_path).convert("RGB")
                except Exception:
                    continue

                babies = [
                    p for p in people
                    if (p.get("type") == "baby" or p.get("type") == "child") and p.get("emotion") in emotion_mapping
                ]
                if not babies:
                    continue

                faces = self.mtcnn(img)
                if faces is None:
                    continue
                if isinstance(faces, torch.Tensor):
                    faces = [faces]

                count = min(len(faces), len(babies), MAX_BABIES_PER_FRAME)

                frame_faces = []
                frame_labels = []

                for i in range(count):
                    face_tensor = faces[i].cpu()
                    face_pil = self.to_pil(face_tensor)

                    mapped_emotion = emotion_mapping[babies[i]["emotion"]]
                    label_idx = emotion_to_idx[mapped_emotion]

                    if self.train:
                        if mapped_emotion in ['crying', 'laughing']:
                            face_pil = transform_augment(face_pil)
                        else:
                            face_pil = transforms.Resize((160, 160))(face_pil)
                        face_pil = face_pil.convert("RGB")
                        face_tensor_norm = transform_norm(face_pil)
                    else:
                        face_pil = transform_eval_augment(face_pil)
                        face_tensor_norm = face_pil if isinstance(face_pil, torch.Tensor) else transform_norm(face_pil)

                    frame_faces.append(face_tensor_norm)
                    frame_labels.append(label_idx)

                pad_count = MAX_BABIES_PER_FRAME - count
                if pad_count > 0:
                    frame_faces.extend([torch.zeros_like(frame_faces[0])] * pad_count)
                    frame_labels.extend([-1] * pad_count)

                frame_faces = torch.stack(frame_faces)
                frame_labels = torch.tensor(frame_labels)

                self.samples.append((frame_faces, frame_labels))

        if cache_path is not None:
            print(f"Saving processed dataset to {cache_path}")
            torch.save(self.samples, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Create batches from dataset
def make_batch(batch):
    faces_batch = torch.stack([item[0] for item in batch])
    labels_batch = torch.stack([item[1] for item in batch])
    return faces_batch, labels_batch

# Pretrained Model with added classifier layer and dropout
class EmotionModel(nn.Module):
    def __init__(self, num_emotions, max_babies):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.dropout = nn.Dropout(p=0.6)
        self.backbone.classifier = nn.Sequential(
            self.dropout, # Increasing chance of dropout since val loss was bad
            nn.Linear(self.backbone.last_channel, num_emotions)
        )
        self.max_babies = max_babies

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size * self.max_babies, 3, 160, 160)
        out = self.backbone(x)
        out = out.view(batch_size, self.max_babies, -1)
        return out

# Fine tuning model
def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for faces, labels in train_loader:
            faces, labels = faces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(faces).permute(0, 2, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for faces, labels in val_loader:
                faces, labels = faces.to(device), labels.to(device)
                outputs = model(faces).permute(0, 2, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)


# Processing data to ensure proper representation of each emotion class given that our dataset is primarily calm
dataset = EmotionDataset("output_frames.jsonl", frames_root="data", train=True, cache_path="cache/emotion_dataset.pt")

total_size = len(dataset)
test_size = int(total_size//10)
val_size = int(total_size//5)
train_size = total_size - test_size - val_size

# Using the 70, 20, 10 split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=make_batch)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=make_batch)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=make_batch)

# Training
model = EmotionModel(num_emotions=len(emotions), max_babies=MAX_BABIES_PER_FRAME).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train(model, train_loader, val_loader, criterion, optimizer, epochs=10)


# Test ~ 17 samples attempted - getting around 80% test accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (faces, labels) in enumerate(test_loader):
        faces, labels = faces.to(device), labels.to(device)
        outputs = model(faces)
        _, preds = torch.max(outputs, 2)

        labels = labels[0]
        preds = preds[0]

        print(f"\nTest Sample #{batch_idx + 1}")
        for i in range(len(labels)):
            if labels[i].item() == -1:
                continue
            expected_label = emotions[labels[i].item()]
            predicted_label = emotions[preds[i].item()]
            print(f"Baby {i+1}: Expected: {expected_label}, Predicted: {predicted_label}")

        mask = labels != -1
        correct += (preds == labels)[mask].sum().item()
        total += mask.sum().item()

if total > 0:
    print(f"\nOverall Test Accuracy: {100 * correct / total:.2f}%")
else:
    print("No valid test samples.")


# # To save model if it good
# os.makedirs("checkpoints", exist_ok=True)
# save_path = "checkpoints/emotion_model_weights.pth"
# torch.save(model.state_dict(), save_path)
# print(f"Model weights saved to {save_path}")
