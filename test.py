import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

def load_model(model_path, num_classes=2):
    # ResNet50 모델 구조 로드
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # 클래스 수에 맞게 출력 레이어 수정

    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path))

    # 학습 장치 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    
    return model, device

def prepare_test_data(test_dir, batch_size=10):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

def test_model(model, device, test_loader):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def evaluate_accuracy(predictions, labels, class_names):
    total = len(predictions)
    correct_counts = {class_name: 0 for class_name in class_names}
    total_counts = {class_name: 0 for class_name in class_names}
    
    for pred, label in zip(predictions, labels):
        class_name = class_names[label]
        total_counts[class_name] += 1
        if pred == label:
            correct_counts[class_name] += 1
    
    for class_name in class_names:
        accuracy = 100 * correct_counts[class_name] / total_counts[class_name] if total_counts[class_name] > 0 else 0
        print(f"{class_name} 사진의 정확도: {accuracy:.2f}%")

if __name__ == "__main__":
    # 모델 로드
    model_path = 'resnet50_rain_snow_classifier.pth'
    model, device = load_model(model_path)
    
    # 테스트 데이터 준비 (폭우 또는 폭설 이미지 경로 지정)
    test_dir = 'labeled_images/test'  # 폭우 및 폭설 이미지가 있는 테스트 폴더 경로
    test_loader = prepare_test_data(test_dir, batch_size=10)
    
    # 모델 테스트
    accuracy, predictions, labels = test_model(model, device, test_loader)
    
    # 총 정확도 출력
    print(f"전체 테스트 데이터셋 정확도: {accuracy:.2f}%")
    
    # 폭우 및 폭설에 대한 정확도 평가
    class_names = ['폭우', '폭설']  # 클래스 이름 지정 (0: 폭우, 1: 폭설)
    evaluate_accuracy(predictions, labels, class_names)
