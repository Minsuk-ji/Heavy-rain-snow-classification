import os
import shutil
import random

# 원본 폴더 경로 설정
dataset_dir = r'labeled_images'
heavy_rain_dir = os.path.join(dataset_dir, 'heavy_rain')
snow_dir = os.path.join(dataset_dir, 'heavy_snow')
normal_dir = os.path.join(dataset_dir, 'normal')

# 분할된 데이터셋을 저장할 폴더 경로
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

# 학습/검증 폴더가 이미 존재하는지 확인
if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
    # 각 클래스별로 학습/검증 폴더 생성
    os.makedirs(os.path.join(train_dir, 'heavy_rain'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'heavy_snow'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'heavy_rain'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'heavy_snow'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'normal'), exist_ok=True)

    # 학습/검증 비율 설정 (80% 학습, 20% 검증)
    train_ratio = 0.8
    val_ratio = 0.2

    def split_dataset(images, src_dir, train_dest, val_dest):
        random.shuffle(images)
        train_size = int(len(images) * train_ratio)

        for i, img_name in enumerate(images):
            src_path = os.path.join(src_dir, img_name)
            if i < train_size:
                shutil.copy(src_path, os.path.join(train_dest, img_name))
            else:
                shutil.copy(src_path, os.path.join(val_dest, img_name))

    # heavy_rain 이미지를 학습/검증 데이터로 분할
    heavy_rain_images = os.listdir(heavy_rain_dir)
    split_dataset(
        heavy_rain_images,
        heavy_rain_dir,
        os.path.join(train_dir, 'heavy_rain'),
        os.path.join(val_dir, 'heavy_rain')
    )

    # heavy_snow 이미지를 학습/검증 데이터로 분할
    snow_images = os.listdir(snow_dir)
    split_dataset(
        snow_images,
        snow_dir,
        os.path.join(train_dir, 'heavy_snow'),
        os.path.join(val_dir, 'heavy_snow')
    )

    # normal 이미지를 학습/검증 데이터로 분할
    normal_images = os.listdir(normal_dir)
    split_dataset(
        normal_images,
        normal_dir,
        os.path.join(train_dir, 'normal'),
        os.path.join(val_dir, 'normal')
    )

    print("데이터셋이 학습(train)과 검증(val)으로 분할되었습니다.")
else:
    print("이미 학습(train)과 검증(val) 데이터셋이 존재합니다.")
