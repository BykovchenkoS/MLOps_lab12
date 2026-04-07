import os
import shutil
from pathlib import Path
from PIL import ImageEnhance, ImageOps, ImageFilter
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import json

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_MAPPING = {
    # Trail_Camera_Wildlife_v4
    'Urs': 'Bear',
    'Cerb Comun': 'Deer',
    'Mistret': 'Boar',

    # Wildlife_v2
    'SunBear': 'Bear',
    'Elephant': 'Elephant',
    'Tiger': 'Tiger',
    'Tapir': 'Tapir',
    'ForestBG': 'Background'
}


TARGET_PER_CLASS = 150
DATASET_1 = Path('Trail Camera Wildlife.v4-4.folder')
DATASET_2 = Path('Wildlife.v2i.folder')
OUTPUT_DIR = Path('dataset/')


def collect_images(dataset_path, class_mapping):
    images_by_class = {}
    print(f"Собираем изображения из {dataset_path.name}...")
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}

    for img_file in tqdm(list(dataset_path.rglob('*')), desc="Поиск"):
        if img_file.suffix not in extensions:
            continue

        skip_names = {'train', 'test', 'val', 'valid', 'images', 'labels'}
        current = img_file.parent

        while current != dataset_path and current.name.lower() in skip_names:
            current = current.parent

        if current == dataset_path:
            continue

        class_name = current.name
        mapped_class = class_mapping.get(class_name, class_name)

        if mapped_class not in images_by_class:
            images_by_class[mapped_class] = []
        images_by_class[mapped_class].append(img_file)

    return images_by_class


def augment_image_only(img_path, base_output_path, n_augmentations=5):
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            ext = img_path.suffix

            created = 0

            for i in range(n_augmentations):
                aug_img = img.copy()

                # Горизонтальное отражение
                if random.random() > 0.5:
                    aug_img = ImageOps.mirror(aug_img)

                # Поворот (-20 до +20 градусов)
                angle = random.uniform(-20, 20)
                aug_img = aug_img.rotate(angle, expand=False, resample=Image.BICUBIC)

                # Random crop
                if random.random() > 0.5:
                    width, height = aug_img.size
                    left = random.randint(0, int(width * 0.15))
                    top = random.randint(0, int(height * 0.15))
                    right = width - random.randint(0, int(width * 0.15))
                    bottom = height - random.randint(0, int(height * 0.15))
                    aug_img = aug_img.crop((left, top, right, bottom))
                    aug_img = aug_img.resize((width, height), Image.BICUBIC)

                # Яркость (0.7 до 1.3)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(aug_img)
                    aug_img = enhancer.enhance(random.uniform(0.7, 1.3))

                # Контраст (0.7 до 1.3)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Contrast(aug_img)
                    aug_img = enhancer.enhance(random.uniform(0.7, 1.3))

                # Цвет (0.7 до 1.3)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Color(aug_img)
                    aug_img = enhancer.enhance(random.uniform(0.7, 1.3))

                # Резкость (1.0 до 2.0)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Sharpness(aug_img)
                    aug_img = enhancer.enhance(random.uniform(1.0, 2.0))

                # Grayscale
                if random.random() > 0.8:
                    aug_img = aug_img.convert('L').convert('RGB')

                # Gaussian Blur
                if random.random() > 0.85:
                    aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

                # Gaussian Noise
                if random.random() > 0.8:
                    img_array = np.array(aug_img)
                    noise = np.random.normal(0, random.randint(10, 30), img_array.shape).astype(np.uint8)
                    img_array = np.clip(img_array + noise, 0, 255)
                    aug_img = Image.fromarray(img_array)

                # Solarize
                if random.random() > 0.9:
                    aug_img = ImageOps.solarize(aug_img, threshold=random.randint(100, 200))

                # Posterize
                if random.random() > 0.9:
                    aug_img = ImageOps.posterize(aug_img, bits=random.randint(4, 6))

                aug_name = f"{base_output_path.stem}_aug{i + 1}{ext}"
                aug_path = base_output_path.parent / aug_name
                aug_img.save(aug_path, quality=85)
                created += 1

            return created

    except Exception as e:
        print(f"ошибка {img_path} {e}")
        return 0


def merge_and_balance(datasets, output_dir, target_per_class=150):
    print(f"\n{'=' * 60}")
    print("Объединение и балансировка датасетов")
    print(f"{'=' * 60}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = {}

    for dataset_path, class_mapping in datasets:
        images = collect_images(dataset_path, class_mapping)
        for cls, img_list in images.items():
            if cls not in all_images:
                all_images[cls] = []
            all_images[cls].extend(img_list)

    print(f"\nДо балансировки")
    print(f"{'Класс':<20} {'Изображений':>12} {'Цель':>12} {'Нужно добавить':>15}")
    print("-" * 60)

    for cls in sorted(all_images.keys()):
        count = len(all_images[cls])
        need = max(0, target_per_class - count)
        print(f"{cls:<20} {count:>12} {target_per_class:>12} {need:>15}")

    print(f"\nКопирование и аугментация...")

    final_counts = {}

    for cls, img_list in tqdm(all_images.items(), desc="Классы"):
        class_dir = output_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        current_count = len(img_list)

        if current_count < target_per_class:
            need = target_per_class - current_count
            augment_per_image = (need + current_count - 1) // current_count

            print(f"\n{cls}: {current_count} изображений → {augment_per_image} аугментаций each")

            # Для каждого изображения создаём оригинал + аугментации
            for i, img_path in enumerate(img_list):
                # Сначала копируем оригинал
                try:
                    orig_dest = class_dir / f"{cls}_{i:04d}{img_path.suffix}"
                    shutil.copy2(img_path, orig_dest)
                except Exception as e:
                    print(f"Ошибка копирования оригинала {img_path} {e}")
                    continue

                base_name = f"{cls}_{i:04d}"
                output_path = class_dir / f"{base_name}{img_path.suffix}"
                created = augment_image_only(img_path, output_path, n_augmentations=augment_per_image)
            final_counts[cls] = current_count + (current_count * augment_per_image)

        else:
            for i, img_path in enumerate(img_list[:target_per_class]):
                try:
                    dest = class_dir / f"{cls}_{i:04d}{img_path.suffix}"
                    shutil.copy2(img_path, dest)
                except Exception as e:
                    print(f"Ошибка копирования {img_path} {e}")
            final_counts[cls] = min(current_count, target_per_class)

    # Финальная статистика
    print(f"\nПосле балансировки")
    print(f"{'Класс':<20} {'Изображений':>12}")
    print("-" * 60)

    for cls in sorted(final_counts.keys()):
        count = final_counts[cls]
        print(f"{cls:<20} {count:>12}")

    metadata = {
        'total_images': sum(final_counts.values()),
        'num_classes': len(final_counts),
        'class_counts': final_counts,
        'target_per_class': target_per_class,
        'class_mapping': CLASS_MAPPING,
        'source_datasets': [str(DATASET_1), str(DATASET_2)],
        'augmentations': [
            'mirror', 'rotate', 'crop', 'brightness', 'contrast', 'color',
            'sharpness', 'grayscale', 'gaussian_blur', 'gaussian_noise',
            'solarize', 'posterize'
        ]
    }

    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return final_counts


def create_splits(dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    dataset_dir = Path(dataset_dir)

    for split in ['train', 'val', 'test']:
        for cls_dir in dataset_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            (dataset_dir / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

    for cls_dir in tqdm(list(dataset_dir.iterdir()), desc="Классы"):
        if not cls_dir.is_dir() or cls_dir.name in ['train', 'val', 'test']:
            continue

        images = list(cls_dir.glob('*.[jJ][pP][gG]')) + list(cls_dir.glob('*.[pP][nN][gG]')) + list(cls_dir.glob('*.[wW][eE][bB][pP]'))
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img, split_name in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
            for img_path in img:
                dest = dataset_dir / split_name / cls_dir.name / img_path.name
                shutil.copy2(img_path, dest)

    # Статистика split'ов
    print(f"\nРазмеры сплитов")
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        total = sum(1 for _ in split_dir.rglob('*.jpg'))
        total += sum(1 for _ in split_dir.rglob('*.png'))
        print(f"   {split}: {total} изображений")


def main():
    datasets = [(DATASET_1, CLASS_MAPPING), (DATASET_2, CLASS_MAPPING)]
    final_counts = merge_and_balance(datasets=datasets, output_dir=OUTPUT_DIR, target_per_class=TARGET_PER_CLASS)
    create_splits(OUTPUT_DIR)


if __name__ == "__main__":
    main()
