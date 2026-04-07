import zipfile
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP'}
SPLIT_NAMES = {'train', 'test', 'val', 'valid', 'validation', 'images', 'labels'}


def extract_zip(zip_path, extract_to=None):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"Не найден: {zip_path}")
        return None

    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Распаковка: {zip_path.name} -> {extract_to.name}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file in tqdm(zf.namelist(), desc="Извлечение"):
            zf.extract(file, extract_to)
    return extract_to


def get_class_name(img_path, dataset_root):
    cur = img_path.parent
    while cur != dataset_root and cur.name.lower() in SPLIT_NAMES:
        cur = cur.parent
    if cur == dataset_root:
        return "unknown"
    return cur.name


def scan_dataset(dataset_path, dataset_name):
    print(f"\nСканирование: {dataset_name}")
    print(f"{dataset_path}")
    class_counts = Counter()
    image_sizes = []
    corrupted = []
    class_samples = {}
    dataset_path = Path(dataset_path)

    for img_file in tqdm(list(dataset_path.rglob('*')), desc="Поиск файлов"):
        if img_file.suffix not in EXTENSIONS:
            continue
        class_name = get_class_name(img_file, dataset_path)
        class_counts[class_name] += 1
        if class_name not in class_samples and class_counts[class_name] == 1:
            class_samples[class_name] = img_file

        try:
            with Image.open(img_file) as img:
                image_sizes.append(img.size)
                img.load()
        except Exception as e:
            corrupted.append({'path': str(img_file), 'error': str(e)[:80]})

    total = sum(class_counts.values())
    n_classes = len(class_counts)

    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"        Всего изображений: {total:,}")
    print(f"        Классов: {n_classes}")
    print(f"        Повреждённых: {len(corrupted)}")

    if total > 0 and n_classes > 0:
        counts = list(class_counts.values())
        print(f"        Среднее на класс: {np.mean(counts):.1f}")
        print(f"        min/max: {min(counts)} / {max(counts)}")
        print(f"        Дисбаланс: {max(counts) / max(min(counts), 1):.2f}x")

        print(f"\nКлассы:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            bar = '█' * min(cnt // 5, 40)
            print(f"   {cls:20s} {cnt:4d} {bar}")

        poor = [(c, cnt) for c, cnt in class_counts.items() if cnt < 20]
        if poor:
            print(f"\n  Мало данных:")
            for c, cnt in sorted(poor, key=lambda x: x[1]):
                print(f"   • {c}: {cnt}")

    return {
        'name': dataset_name, 'total': total, 'n_classes': n_classes, 'class_counts': dict(class_counts),
        'corrupted': len(corrupted), 'avg': np.mean(list(class_counts.values())) if class_counts else 0,
        'min': min(class_counts.values()) if class_counts else 0, 'max': max(class_counts.values()) if class_counts else 0,
        'imbalance': max(class_counts.values()) / max(min(class_counts.values()), 1) if class_counts else 0,
        'poor_classes': [c for c, cnt in class_counts.items() if cnt < 20], 'class_samples': {c: str(p) for c, p in class_samples.items()}
    }


def visualize(report, save=True):
    if not report['class_counts']:
        return

    classes = sorted(report['class_counts'].keys(), key=lambda x: report['class_counts'][x])
    counts = [report['class_counts'][c] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{report['name']}", fontsize=14, fontweight='bold')

    # Распределение
    ax = axes[0]
    ax.barh(classes, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Изображений')
    ax.set_title('Распределение по классам')
    ax.tick_params(axis='y', labelsize=8)

    # Проблемные классы
    ax = axes[1]
    poor = [(c, cnt) for c, cnt in zip(classes, counts) if cnt < 20]
    if poor:
        poor_c, poor_cnt = zip(*poor)
        ax.barh(poor_c, poor_cnt, color='coral', alpha=0.7)
        ax.set_xlabel('Изображений')
        ax.set_title('Классы с малым количеством (<20)')
        ax.tick_params(axis='y', labelsize=8)
    else:
        ax.text(0.5, 0.5, 'Все классы имеют достаточно данных',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Качество данных')
        ax.axis('off')

    plt.tight_layout()
    if save:
        fname = f"analysis_{report['name'].replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    zip_files = {
        'Trail_Camera_Wildlife_v4': 'Trail Camera Wildlife.v4-4.folder.zip',
        'Wildlife_v2': 'Wildlife.v2i.folder.zip'
    }

    reports = {}
    for name, zip_path in zip_files.items():
        print(f"\n{'#' * 60}\n {name}\n{'#' * 60}")

        extracted = extract_zip(zip_path)
        if not extracted:
            continue

        report = scan_dataset(extracted, name)
        if report:
            reports[name] = report
            visualize(report, save=True)

    if len(reports) >= 2:
        print(f"\n{'=' * 60}\nСРАВНЕНИЕ\n{'=' * 60}")
        df = pd.DataFrame([
            {
                'Dataset': r['name'],
                'Images': r['total'],
                'Classes': r['n_classes'],
                'Avg/Class': f"{r['avg']:.1f}",
                'Imbalance': f"{r['imbalance']:.2f}x",
                'Poor (<20)': len(r['poor_classes']),
                'Corrupted': r['corrupted']
            }
            for r in reports.values()
        ])
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
