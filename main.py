import argparse
import torch.optim as optim
import torch.nn as nn
from dataset import load_datasets
from model import initialize_model
from trainer import EfficientNetTrainer
from utils import save_checkpoint, load_checkpoint, plot_results
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet model for image classification.")
    parser.add_argument('--train_dir', type=str, default='labeled_images/train', help='학습 데이터가 저장된 디렉터리.')
    parser.add_argument('--val_dir', type=str, default='labeled_images/val', help='검증 데이터가 저장된 디렉터리.')
    parser.add_argument('--batch_size', type=int, default=32, help='학습 및 검증의 배치 크기.')
    parser.add_argument('--num_classes', type=int, default=3, help='클래스의 수.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='학습률.')
    parser.add_argument('--num_epochs', type=int, default=20, help='에포크 수.')
    parser.add_argument('--checkpoint_file', type=str, default='logs/checkpoint.pth', help='체크포인트 파일 경로.')
    parser.add_argument('--use_checkpoint', action='store_true', help='체크포인트를 사용하여 학습을 재개할지 여부.')
    parser.add_argument('--patience', type=int, default=5, help='조기 종료를 위한 patience 값.')
    
    args = parser.parse_args()

    # 데이터셋 로딩
    train_loader, val_loader = load_datasets(args.train_dir, args.val_dir, args.batch_size)
    
    # 모델 초기화
    model, device = initialize_model(args.num_classes)
    
    # 옵티마이저 및 손실 함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 학습기 초기화
    trainer = EfficientNetTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_file=args.checkpoint_file,
        use_checkpoint=args.use_checkpoint,
        patience=args.patience  # 조기 종료를 위한 patience 값 설정
    )
    
    # 체크포인트 로딩 및 결과 기록 로딩
    start_epoch, result_record = load_checkpoint(model, optimizer, args.checkpoint_file, args.use_checkpoint)

    # 결과 기록을 저장하기 위한 result_record 초기화
    if not result_record:
        result_record = {
            'Train Loss': [],
            'Train Acc': [],
            'Validation Loss': [],
            'Validation Acc': []
        }

    # 학습 시작
    for epoch in range(start_epoch, args.num_epochs):
        # 한 에폭 동안 학습 진행
        train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch, args.num_epochs)
        validation_result = trainer.validate_model(val_loader)

        # 조기 종료 여부 확인
        if validation_result == "early_stop":
            print(f"Training stopped early at epoch {epoch+1} due to early stopping.")
            break
        
        val_loss, val_acc = validation_result
        
        # 결과 기록
        result_record['Train Loss'].append(train_loss)
        result_record['Train Acc'].append(train_acc)
        result_record['Validation Loss'].append(val_loss)
        result_record['Validation Acc'].append(val_acc)
        
        # 체크포인트 저장
        save_checkpoint(trainer.model, trainer.optimizer, epoch, args.checkpoint_file, result_record)
        
        # 최적 모델 저장
        if val_acc > trainer.best_accuracy:
            trainer.best_accuracy = val_acc
            print('* Saving the best model *')
            torch.save(trainer.model.state_dict(), 'save_model/EfficientNet_rain_snow_classifier.pth')
    
    print("Training complete.")
    
    # 결과 시각화
    plot_results(result_record)
