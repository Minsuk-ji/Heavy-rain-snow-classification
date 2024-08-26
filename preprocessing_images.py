import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import labeling_images

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로딩
train_dataset = datasets.ImageFolder(root=labeling_images.train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=labeling_images.val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
