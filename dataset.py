import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import random
from PIL import ImageFilter, Image

# Custom transform to add motion blur
class RandomMotionBlur:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=3))  # Simulates motion blur
        return img

# Custom transform to add random noise
class RandomNoise:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            noise = Image.effect_noise(img.size, random.uniform(0.1, 1.5))
            return Image.blend(img, noise, alpha=0.3)
        return img

def load_datasets(train_dir, val_dir, batch_size=32, num_workers=4, dataset_type="general"):
    if dataset_type == "heavy_rain":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            RandomMotionBlur(p=0.5),  # 추가된 Motion Blur
            RandomNoise(p=0.5),  # 추가된 Random Noise
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # For "heavy_snow" and "normal" classes, use default augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader
