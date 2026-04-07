import re
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def is_roboflow_augmented(filename):
    return '.rf.' in filename.lower()


def get_base_image_name(filename):
    if not is_roboflow_augmented(filename):
        return Path(filename).stem

    name = Path(filename).stem
    match = re.match(r'^(.+?)\.rf\.[a-f0-9]+$', name, re.IGNORECASE)
    if match:
        return match.group(1)
    return name


def clean_roboflow_augmentations(dataset_path, dry_run=True):
    dataset_path = Path(dataset_path)
    class_base_to_versions = defaultdict(list)

    print("\nСканирование файлов...")
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}

    for img_file in tqdm(list(dataset_path.rglob('*'))):
        if img_file.suffix not in extensions:
            continue

        current = img_file.parent
        class_name = None

        while current != dataset_path:
            if current.name.lower() not in {'train', 'test', 'val', 'valid', 'validation', 'images', 'labels'}:
                class_name = current.name
                break
            current = current.parent

        if class_name is None:
            continue

        base_name = get_base_image_name(img_file.name)
        key = (class_name, base_name)
        class_base_to_versions[key].append(img_file)

    print(f"\nСтатистика:")
    total_bases = len(class_base_to_versions)
    total_files = sum(len(versions) for versions in class_base_to_versions.values())
    duplicates = sum(len(versions) - 1 for versions in class_base_to_versions.values() if len(versions) > 1)

    print(f"    Уникальных базовых изображений: {total_bases}")
    print(f"    Всего файлов: {total_files}")
    print(f"    Дубликатов (будет удалено): {duplicates}")

    if duplicates == 0:
        print("Дубликатов не найдено")
        return

    if not dry_run:
        print(f"\nУдаление дубликатов...")
        removed_count = 0

        for (class_name, base_name), versions in tqdm(class_base_to_versions.items(), desc="Очистка"):
            if len(versions) <= 1:
                continue

            keep = versions[0]
            to_remove = versions[1:]

            for img_path in to_remove:
                try:
                    img_path.unlink()
                    removed_count += 1

                    parent = img_path.parent
                    while parent != dataset_path and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent

                except Exception as e:
                    print(f"Ошибка удаления {img_path}: {e}")

        print(f"\nУдалено файлов: {removed_count}")
        remaining = sum(1 for _ in dataset_path.rglob('*.jpg'))
        remaining += sum(1 for _ in dataset_path.rglob('*.png'))
        print(f"Осталось файлов: {remaining}")


def count_originals_per_class(dataset_path):
    dataset_path = Path(dataset_path)
    class_bases = defaultdict(set)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}

    for img_file in dataset_path.rglob('*'):
        if img_file.suffix not in extensions:
            continue

        current = img_file.parent
        class_name = None

        while current != dataset_path:
            if current.name.lower() not in {'train', 'test', 'val', 'valid', 'validation', 'images', 'labels'}:
                class_name = current.name
                break
            current = current.parent

        if class_name is None:
            continue

        base_name = get_base_image_name(img_file.name)
        class_bases[class_name].add(base_name)

    print(f"\nОригинальных изображений по классам:")
    print(f"{'Класс':<20} {'Количество':>12}")
    print("-" * 35)
    for cls, bases in sorted(class_bases.items(), key=lambda x: -len(x[1])):
        bar = '█' * min(len(bases) // 2, 40)
        print(f"{cls:<20} {len(bases):>12} {bar}")

    total = sum(len(bases) for bases in class_bases.values())
    print(f"\n   ВСЕГО: {total}")

    return {cls: len(bases) for cls, bases in class_bases.items()}


if __name__ == "__main__":
    dataset_path = Path("Wildlife.v2i.folder")
    clean_roboflow_augmentations(dataset_path, dry_run=True)
    original_counts = count_originals_per_class(dataset_path)
    print("\n" + "=" * 60)
    response = input("Удалить дубликаты? (y/n): ").strip().lower()
    if response == 'y':
        clean_roboflow_augmentations(dataset_path, dry_run=False)
        print("\nОчистка завершена")
    else:
        print("\nПропущено удаление")
