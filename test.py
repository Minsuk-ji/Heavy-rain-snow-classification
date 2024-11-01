import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import argparse
import logging
import torchvision.utils as vutils
import time
import os

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 역정규화를 위한 변환 정의
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# 역정규화 함수
def unnormalize_image(tensor_image):
    """이미지를 원래 상태로 복원"""
    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    # 역정규화 적용
    return inv_norm(tensor_image)

# Load the EfficientNet model with a Dropout layer
def load_model(model_path, num_classes=3, dropout_prob=0.5):
    """EfficientNet 모델 로드"""
    try:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(num_ftrs, num_classes)
        )

        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        logging.info(f"Model loaded successfully from {model_path}")
        return model, device
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

# Prepare test data
def prepare_test_data(test_dir=None, test_image_path=None, cctv_dir=None, public_dir=None, batch_size=10):
    """테스트 데이터를 준비"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if test_image_path:
        try:
            # 단일 이미지 로드
            image = Image.open(test_image_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # 배치 차원 추가
            test_loader = [(image, None)]  # 단일 이미지용 페이크 데이터 로더
            logging.info(f"Single image loaded from {test_image_path}")
            return test_loader, None
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            raise
    elif cctv_dir and public_dir:
        try:
            # CCTV와 public 데이터셋 로드
            cctv_dataset = datasets.ImageFolder(root=cctv_dir, transform=transform)
            logging.info(f"CCTV dataset loaded from {cctv_dir}")
            public_dataset = datasets.ImageFolder(root=public_dir, transform=transform)
            logging.info(f"Public dataset loaded from {public_dir}")

            combined_dataset = ConcatDataset([cctv_dataset, public_dataset])
            combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

            # 클래스 라벨이 일치하는지 확인
            if cctv_dataset.classes != public_dataset.classes:
                raise ValueError("Class mismatch between CCTV and Public datasets!")
            classes = cctv_dataset.classes

            return combined_loader, classes
        except Exception as e:
            logging.error(f"Failed to load datasets: {e}")
            raise
    else:
        try:
            # 데이터셋 로드
            test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            logging.info(f"Dataset loaded from {test_dir}")
            return test_loader, test_dataset.classes
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

# Test the model
def test_model(model, device, test_loader, classes, single_image=False):
    """모델 테스트"""
    model.eval()
    total_correct = 0
    total_images = 0
    total_inference_time = 0

    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    predicted_all = []
    labels_all = []
    confidence_all = []
    images_all = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing Model"):
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            # 추론 시간 계산
            inference_time = end_time - start_time
            total_inference_time += inference_time * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            predicted_all.extend(predicted.cpu().numpy())

            if labels is not None:
                labels_all.extend(labels.cpu().numpy())
                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                # 클래스별 정확도 계산
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

            confidence_all.extend(torch.nn.functional.softmax(outputs, dim=1).max(1).values.cpu().numpy())
            images_all.extend(inputs.cpu())

    # 각 클래스별 정확도 계산
    accuracy_per_class = [(100 * class_correct[i] / class_total[i]) if class_total[i] > 0 else 0 for i in range(len(class_total))]
    avg_inference_time = total_inference_time / total_images
    overall_accuracy = 100 * total_correct / total_images if total_images > 0 else 0

    return images_all, predicted_all, labels_all, confidence_all, accuracy_per_class, avg_inference_time, overall_accuracy

# Visualize predictions in grid (역정규화 적용)
def visualize_predictions_grid(images, predicted, labels, classes, confidence, nrow=2):
    """이미지를 그리드로 시각화하고 예측 결과를 표시"""
    # 역정규화 이미지 리스트 생성
    unnormalized_images = [unnormalize_image(image) for image in images]

    # 그리드 이미지 생성
    grid_img = vutils.make_grid(unnormalized_images[:nrow * nrow], nrow=nrow)
    grid_img = transforms.ToPILImage()(grid_img)
    width, height = grid_img.size
    font = ImageFont.truetype("arial.ttf", 15)
    draw = ImageDraw.Draw(grid_img, 'RGBA')

    # 텍스트 박스 크기
    x_border = 180
    y_border = 50
    i = 0
    for row in range(1, nrow + 1):
        for col in range(1, nrow + 1):
            x = width * col / nrow - width / nrow * 0.96
            y = height * row / nrow - 219

            # 텍스트 뒤에 사각형 그리기
            draw.rectangle((x - 3, y - 1, x + x_border, y + y_border), fill=(0, 0, 0, 180))

            # 예측된 라벨, 실제 라벨, 신뢰도 표시
            pred_msg = f'Prediction: {classes[predicted[i]]}'
            conf_msg = f'Confidence: {confidence[i]:.2f}'
            lab_msg = f'Label: {classes[labels[i]]}'
            draw.text((x, y), lab_msg, fill='white', font=font)
            draw.text((x, y + 15), pred_msg, fill='white', font=font)
            draw.text((x, y + 30), conf_msg, fill='white', font=font)

            i += 1

    plt.figure(figsize=(20, 20))
    plt.imshow(grid_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Print the results for each class and save to a file
def print_class_results(classes, accuracy_per_class, avg_inference_time, overall_accuracy, output_dir="results", output_file="폭설_폭우 상황 인식 정확도.txt"):
    """각 클래스의 결과를 출력 및 저장"""
    results = []
    results.append(f'Overall Accuracy: {overall_accuracy:.2f}%')
    results.append(f'Average Inference Time: {avg_inference_time:.4f} seconds per image\n')
    for i, class_name in enumerate(classes):
        results.append(f'Class: {class_name}')
        results.append(f'  - Accuracy: {accuracy_per_class[i]:.2f}%\n')
    
    # 화면에 출력
    for line in results:
        print(line)

    # 결과 파일 저장 경로 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)
    
    # 파일에 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")
    
    print(f"Results saved to {output_path}")
# Visualize a single image prediction
def visualize_single_prediction(image_path, prediction, confidence, class_names):
    """단일 이미지 예측 결과 시각화"""
    image = Image.open(image_path)
    plt.imshow(np.array(image))
    plt.title(f"Predicted: {class_names[prediction]} (Confidence: {confidence:.2f})", fontsize=16, loc='left', pad=20)
    plt.axis('off')  # 축 숨김
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an image or a dataset using a trained EfficientNet model.")
    parser.add_argument('--test_image_path', type=str, help='Path to the image to test.')
    parser.add_argument('--test_dir', type=str, help='Directory of images to test.')
    parser.add_argument('--cctv_dir', type=str, help='Directory of CCTV test images.')
    parser.add_argument('--public_dir', type=str, help='Directory of public test images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for testing.')
    parser.add_argument('--class_names', type=str, nargs='+', default=['폭우', '폭설', '일반'], help='List of class names.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the result files.')


    args = parser.parse_args()

    # 모델 로드
    model, device = load_model(args.model_path, num_classes=len(args.class_names))

    if args.test_image_path and os.path.isfile(args.test_image_path):
        # 단일 이미지 테스트
        test_loader, _ = prepare_test_data(test_image_path=args.test_image_path)
        prediction, confidence = test_model(model, device, test_loader, classes=args.class_names, single_image=True)
        visualize_single_prediction(args.test_image_path, prediction[0], confidence[0], args.class_names)
    elif args.cctv_dir and args.public_dir:
        # CCTV 및 Public 데이터셋 테스트
        test_loader, classes = prepare_test_data(cctv_dir=args.cctv_dir, public_dir=args.public_dir, batch_size=args.batch_size)
        images, predicted, labels, confidence, accuracy_per_class, avg_inference_time, overall_accuracy = test_model(model, device, test_loader, classes)
        visualize_predictions_grid(images, predicted, labels, classes, confidence, nrow=5)
        print_class_results(classes, accuracy_per_class, avg_inference_time, overall_accuracy)
    elif args.test_dir and os.path.isdir(args.test_dir):
        # 전체 데이터셋 테스트
        test_loader, classes = prepare_test_data(test_dir=args.test_dir, batch_size=args.batch_size)
        images, predicted, labels, confidence, accuracy_per_class, avg_inference_time, overall_accuracy = test_model(model, device, test_loader, classes)
        visualize_predictions_grid(images, predicted, labels, classes, confidence, nrow=2)
        print_class_results(classes, accuracy_per_class, avg_inference_time, overall_accuracy, output_dir=args.output_dir, output_file="폭설_폭우 상황 인식 정확도.txt")
    else:
        logging.error("Error: You must provide either --test_image_path for a single image, --test_dir for a dataset, or --cctv_dir and --public_dir for combined testing.")
