import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from labeling_images import train_dir, val_dir

def load_datasets(train_dir, val_dir, batch_size=32):
    # 이미지 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로딩
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class ResNetTrainer:
    def __init__(self, num_classes=2, learning_rate=0.001, checkpoint_file='checkpoint.pth'):
        self.model, self.device = self.initialize_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.checkpoint_file = checkpoint_file
        self.result_record = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}
        self.best_accuracy = 0.0

    def initialize_model(self, num_classes):
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model.to(device), device

    def train_one_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                tqdm.write(f'Step [{i+1}/{len(train_loader)}], '
                           f'Loss: {running_loss/(i+1):.4f}, Accuracy: {100 * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        self.result_record['Train Loss'].append(epoch_loss)
        self.result_record['Train Acc'].append(epoch_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}] finished with Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    def validate_model(self, val_loader):
        self.model.eval()
        cost = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                cost += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_loss = cost / len(val_loader)
        validation_accuracy = 100 * correct / total

        self.result_record['Validation Loss'].append(validation_loss)
        self.result_record['Validation Acc'].append(validation_accuracy)

        print(f'Validation Accuracy: {validation_accuracy:.2f}%\tValidation Loss: {validation_loss:.4f}')
        return validation_accuracy

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded: start training from epoch {start_epoch}")
            return start_epoch
        else:
            print("No checkpoint found, start training from scratch")
            return 0

    def train(self, num_epochs=10):
        start_epoch = self.load_checkpoint()

        for epoch in range(start_epoch, num_epochs):
            self.train_one_epoch(train_loader, epoch, num_epochs)
            accuracy = self.validate_model(val_loader)

            self.save_checkpoint(epoch)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print('* Saving the best model *')
                torch.save(self.model.state_dict(), 'resnet50_rain_snow_classifier.pth')

        print("Training complete.")
        self.plot_results()

    def plot_results(self):
        plt.figure(1)
        plt.plot(self.result_record['Train Loss'], 'b', label='training loss')
        plt.plot(self.result_record['Validation Loss'], 'r', label='validation loss')
        plt.title("LOSS")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(['training set', 'validation set'], loc='center right')
        plt.savefig('Loss_ResNet50.png', dpi=300, bbox_inches='tight')

        plt.figure(2)
        plt.plot(self.result_record['Train Acc'], 'b', label='training accuracy')
        plt.plot(self.result_record['Validation Acc'], 'r', label='validation accuracy')
        plt.title("ACCURACY")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(['training set', 'validation set'], loc='center right')
        plt.savefig('Accuracy_ResNet50.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    # 데이터셋 로딩
    train_loader, val_loader = load_datasets(train_dir, val_dir)
    
    # 데이터셋 로딩이 잘 되었는지 확인
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")
    
    # 학습 시작
    trainer = ResNetTrainer()
    trainer.train(num_epochs=10)
