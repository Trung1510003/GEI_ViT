import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Thư mục dữ liệu
data_dir = Path("./CASIA_B/output")
save_path = Path("./GEI_IMG")


def load_data(data_dir):
    """Load all PNG image paths from CASIA-B dataset."""
    data = []
    for subject_path in sorted(data_dir.iterdir()):
        if subject_path.is_dir():
            for condition_path in sorted(subject_path.iterdir()):
                if condition_path.is_dir():
                    for view_path in sorted(condition_path.iterdir()):
                        if view_path.is_dir():
                            for img_path in sorted(view_path.glob("*.png")):
                                data.append({
                                    'image_path': str(img_path),
                                    'subject_id': subject_path.name,
                                    'condition': condition_path.name,
                                    'view': view_path.name
                                })
    return data


def crop_and_center(img, size=(128, 88)):
    """Crop the person from image and center them using padding."""
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros(size, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    # Resize while maintaining aspect ratio
    h_ratio = size[0] / h
    w_ratio = size[1] / w
    scale = min(h_ratio, w_ratio)
    resized = cv2.resize(cropped, (int(w * scale), int(h * scale)))

    # Pad to final size
    pad_vert = (size[0] - resized.shape[0]) // 2
    pad_horiz = (size[1] - resized.shape[1]) // 2
    padded = np.zeros(size, dtype=np.uint8)
    padded[pad_vert:pad_vert + resized.shape[0], pad_horiz:pad_horiz + resized.shape[1]] = resized
    return padded


def generate_gei(data, save_path, size=(128, 88)):
    """Generate GEI images from grouped sequences."""
    save_path.mkdir(parents=True, exist_ok=True)
    grouped_data = defaultdict(list)

    for item in data:
        key = (item['subject_id'], item['condition'], item['view'])
        grouped_data[key].append(item['image_path'])
    for (subject_id, condition, view), image_paths in tqdm(grouped_data.items(), desc="Generating GEIs"):
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            processed = crop_and_center(img, size)
            images.append(processed)
        if images:
            gei = np.mean(images, axis=0).astype(np.uint8)
            out_name = f"gei_{subject_id}_{condition}_{view}.png"
            out_path = save_path / out_name
            if not cv2.imwrite(str(out_path), gei):
                print(f"Failed to save: {out_path}")
            else:
                print(f"Saved GEI: {out_path}")


def main():
    data = load_data(data_dir)
    generate_gei(data, save_path)
    pd.DataFrame(data).to_csv(save_path / 'metadata.csv', index=False)
    print(f"Metadata saved to {save_path / 'metadata.csv'}")


if __name__ == "__main__":
    main()
