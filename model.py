import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import preprocessing_images

def initialize_model():
    # ResNet50 모델 로드 및 가중치 설정
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 두 개의 클래스로 출력 레이어 수정

    # 학습 장치 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device

def train_one_epoch(model, device, train_loader, criterion, optimizer, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0:
            tqdm.write(f'Step [{i+1}/{len(train_loader)}], '
                       f'Loss: {running_loss/(i+1):.4f}, Accuracy: {100 * correct / total:.2f}%')

    print(f'Epoch [{epoch+1}/{num_epochs}] finished with Loss: {running_loss/len(train_loader):.4f}')

def validate_model(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def train_model(num_epochs=10):
    model, device = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        train_one_epoch(model, device, preprocessing_images.train_loader, criterion, optimizer, epoch, num_epochs)
        validate_model(model, device, preprocessing_images.val_loader)
    
    save_model(model, 'resnet50_rain_snow_classifier.pth')

if __name__ == "__main__":
    train_model()
