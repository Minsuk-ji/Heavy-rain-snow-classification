import os
import csv

def create_csv_from_images(image_folder, csv_filename, label_mapping):
    """
    주어진 이미지 폴더 내의 이미지 파일 경로와 라벨을 기반으로 CSV 파일을 생성합니다.
    
    Parameters:
    - image_folder: 이미지 파일들이 저장된 폴더 경로
    - csv_filename: 생성할 CSV 파일 이름 (경로 포함)
    - label_mapping: 폴더 이름을 라벨로 매핑한 딕셔너리 (예: {"heavy_rain": 0, "snow": 1, "normal": 2})
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['file_path', 'label'])  # CSV 헤더 작성
        
        # 각 폴더 내의 이미지 파일 경로와 라벨을 CSV에 작성
        for label_name, label_id in label_mapping.items():
            folder_path = os.path.join(image_folder, label_name)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist.")
                continue
            
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_path, filename)
                    csvwriter.writerow([file_path, label_id])
    
    print(f"CSV 파일 '{csv_filename}'이(가) 성공적으로 생성되었습니다.")

# 사용 예시
image_folder = "labeled_images"  # 이미지 폴더 경로
csv_filename = "labels.csv"  # 생성할 CSV 파일 이름
label_mapping = {
    "heavy_rain": 0,  # "heavy_rain" 폴더 내의 이미지들은 라벨 0으로 지정
    "heavy_snow": 1,        # "snow" 폴더 내의 이미지들은 라벨 1으로 지정
    "normal": 2       # "normal" 폴더 내의 이미지들은 라벨 2으로 지정
}

create_csv_from_images(image_folder, csv_filename, label_mapping)
