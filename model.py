import torch
from torchvision import models
import torch.nn as nn

def initialize_model(num_classes=3, dropout_prob=0.5):
    # EfficientNet-B0 모델 불러오기 (사전 학습된 가중치 사용)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # EfficientNet의 마지막 FC 레이어의 입력 피처 수
    num_ftrs = model.classifier[1].in_features

    # 드롭아웃을 추가하여 FC 레이어 커스터마이즈
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_prob),            # 드롭아웃 추가 (p=0.5)
        nn.Linear(num_ftrs, num_classes)       # 최종 클래스 레이어
    )

    # CUDA 사용 가능 여부에 따라 장치를 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return model.to(device), device
