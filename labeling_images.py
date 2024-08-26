import os
import shutil
import random

# 원본 폴더 경로 설정
dataset_dir = r'labeled_images'
heavy_rain_dir = os.path.join(dataset_dir, 'heavy_rain')
snow_dir = os.path.join(dataset_dir, 'heavy_snow')

# 분할된 데이터셋을 저장할 폴더 경로
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

# 학습/검증 폴더가 이미 존재하는지 확인
if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
    # 각 클래스별로 학습/검증 폴더 생성
    os.makedirs(os.path.join(train_dir, 'heavy_rain'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'heavy_snow'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'heavy_rain'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'heavy_snow'), exist_ok=True)

    # 학습/검증 비율 설정 (80% 학습, 20% 검증)
    split_ratio = 0.8

    # heavy_rain 이미지를 학습/검증 데이터로 분할
    heavy_rain_images = os.listdir(heavy_rain_dir)
    random.shuffle(heavy_rain_images)
    train_size = int(len(heavy_rain_images) * split_ratio)

    for i, img_name in enumerate(heavy_rain_images):
        src_path = os.path.join(heavy_rain_dir, img_name)
        if i < train_size:
            shutil.copy(src_path, os.path.join(train_dir, 'heavy_rain', img_name))
        else:
            shutil.copy(src_path, os.path.join(val_dir, 'heavy_rain', img_name))

    # snow 이미지를 학습/검증 데이터로 분할
    snow_images = os.listdir(snow_dir)
    random.shuffle(snow_images)
    train_size = int(len(snow_images) * split_ratio)

    for i, img_name in enumerate(snow_images):
        src_path = os.path.join(snow_dir, img_name)
        if i < train_size:
            shutil.copy(src_path, os.path.join(train_dir, 'heavy_snow', img_name))
        else:
            shutil.copy(src_path, os.path.join(val_dir, 'heavy_snow', img_name))

    print("데이터셋이 학습(train)과 검증(val)으로 분할되었습니다.")
else:
    print("이미 학습(train)과 검증(val) 데이터셋이 존재합니다.")