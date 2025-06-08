import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import ToPILImage
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

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
print(device)
MAX_BABIES_PER_FRAME = 3


transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.01),
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
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def make_batch(batch):
    faces_batch = torch.stack([item[0] for item in batch])
    labels_batch = torch.stack([item[1] for item in batch])
    return faces_batch, labels_batch

class EmotionModel(nn.Module):
    def __init__(self, num_emotions, max_babies):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.dropout = nn.Dropout(p=0.3)
        self.backbone.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(self.backbone.last_channel, num_emotions)
        )
        self.max_babies = max_babies

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size * self.max_babies, 3, 160, 160)
        out = self.backbone(x)
        out = out.view(batch_size, self.max_babies, -1)
        return out
    


def process_by_video(jsonl_path, frames_root, cache_dir="cache", overwrite_cache=False):
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train.pt")

    if not overwrite_cache and os.path.exists(train_cache):
        train_samples = torch.load(train_cache)
        return EmotionDataset(train_samples)

    video_to_samples = {}
    to_pil = ToPILImage()

    count_laughing = 0
    count_calm = 0
    count_crying = 0

    mtcnn = MTCNN(image_size=160, margin=0, device=device)

    for line in open(jsonl_path, 'r'):
        data = json.loads(line)
        frame_id = data['id']
        video_id, frame_num = frame_id.split('_frame_')
        frame_path = os.path.join(frames_root, video_id, "frames", f"frame_{frame_num}.jpg")

        if not os.path.exists(frame_path):
            continue

        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception:
            continue

        people = data.get("people", [])
        babies = [p for p in people if p.get("type") in ["baby", "child"] and p.get("emotion") in emotion_mapping]
        if not babies:
            continue

        for baby in babies:
            mapped_emotion = emotion_mapping[baby["emotion"]]
            if mapped_emotion == 'laughing':
                count_laughing += 1
            elif mapped_emotion == 'calm':
                count_calm += 1
            elif mapped_emotion == 'crying':
                count_crying += 1

        faces = mtcnn(img)
        if faces is None:
            continue
        if isinstance(faces, torch.Tensor):
            faces = [faces]

        count = min(len(faces), len(babies), MAX_BABIES_PER_FRAME)
        if count == 0:
            continue

        augment_times_list = []
        for i in range(count):
            mapped_emotion = emotion_mapping[babies[i]["emotion"]]
            if mapped_emotion == 'calm':
                augment_times_list.append(1)
            elif mapped_emotion == 'crying':
                augment_times_list.append(2)
            elif mapped_emotion == 'laughing':
                augment_times_list.append(3)
            else:
                augment_times_list.append(1)
        max_augment_times = max(augment_times_list)

        for aug_i in range(max_augment_times):
            frame_faces = []
            frame_labels = []

            for i in range(count):
                face_tensor = faces[i].cpu()
                face_pil = to_pil(face_tensor)

                mapped_emotion = emotion_mapping[babies[i]["emotion"]]
                label_idx = emotion_to_idx[mapped_emotion]

                if aug_i < augment_times_list[i]:
                    if mapped_emotion in ['crying', 'laughing', 'calm']:
                        face_pil_aug = transform_augment(face_pil)
                    else:
                        face_pil_aug = transforms.Resize((160, 160))(face_pil)
                    face_tensor_norm = transform_norm(face_pil_aug.convert("RGB"))
                else:
                    face_tensor_norm = torch.zeros((3, 160, 160))
                    label_idx = -1

                frame_faces.append(face_tensor_norm)
                frame_labels.append(label_idx)

            pad_count = MAX_BABIES_PER_FRAME - count
            if pad_count > 0:
                dummy_face = torch.zeros((3, 160, 160))
                frame_faces.extend([dummy_face] * pad_count)
                frame_labels.extend([-1] * pad_count)

            frame_faces = torch.stack(frame_faces)
            frame_labels = torch.tensor(frame_labels)

            if video_id not in video_to_samples:
                video_to_samples[video_id] = []
            if any(label != -1 for label in frame_labels):
                video_to_samples[video_id].append((frame_faces, frame_labels))

    train_samples = []
    for samples in video_to_samples.values():
        train_samples.extend(samples)

    torch.save(train_samples, train_cache)

    print(f"Number of laughing labels before augmentation: {count_laughing}")
    print(f"Number of calm labels before augmentation: {count_calm}")
    print(f"Number of crying labels before augmentation: {count_crying}")

    return EmotionDataset(train_samples)


jsonl_path = "data/per_frame_analysis.jsonl"
frames_root = "data"

train_dataset = process_by_video(jsonl_path, frames_root, overwrite_cache=True)

def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for faces, labels in train_loader:
            faces, labels = faces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(faces).permute(0, 2, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=make_batch)

model = EmotionModel(num_emotions=len(emotions), max_babies=MAX_BABIES_PER_FRAME).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.4, ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model, train_loader, criterion, optimizer, epochs=10)

# os.makedirs("checkpoints", exist_ok=True)
# torch.save(model.state_dict(), "checkpoints/emotion_model.pth")
# print("Model saved to Google Drive.")