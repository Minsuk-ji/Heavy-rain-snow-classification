import torch
import os
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, checkpoint_file, result_record):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'result_record': result_record  # 결과 기록도 저장
    }
    torch.save(checkpoint, checkpoint_file)

def load_checkpoint(model, optimizer, checkpoint_file, use_checkpoint):
    if use_checkpoint and os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        result_record = checkpoint.get('result_record', {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []})
        print(f"Checkpoint loaded: start training from epoch {start_epoch}")
        return start_epoch, result_record  # result_record도 함께 반환
    else:
        print("No checkpoint found or not using checkpoint, start training from scratch")
        result_record = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}
        return 0, result_record

def plot_results(result_record):
    plt.figure(1)
    plt.plot(result_record['Train Loss'], 'b', label='training loss')
    plt.plot(result_record['Validation Loss'], 'r', label='validation loss')
    plt.title("LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['training set', 'validation set'], loc='center right')
    plt.savefig('Loss_ResNet50.png', dpi=300, bbox_inches='tight')

    plt.figure(2)
    plt.plot(result_record['Train Acc'], 'b', label='training accuracy')
    plt.plot(result_record['Validation Acc'], 'r', label='validation accuracy')
    plt.title("ACCURACY")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['training set', 'validation set'], loc='center right')
    plt.savefig('Accuracy_ResNet50.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()