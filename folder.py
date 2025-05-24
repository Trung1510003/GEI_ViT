import os
import shutil
import re
from pathlib import Path

# Configuration
config = {
    'input_dir': '.\GEI_IMG',  # Thư mục chứa ảnh GEI chưa phân loại
    'output_dir': '.\CASIA_B123',     # Thư mục đầu ra với cấu trúc ID/condition/view
}

def extract_metadata_from_filename(filename):
    """
    Suy ra ID, condition, và view từ tên file.
    Giả định tên file có dạng: gei_{ID}_{condition}_{view}.png (ví dụ: gei_123_cl-01_036.png)
    """
    try:
        # Bỏ phần mở rộng (.png)
        name = os.path.splitext(filename)[0]
        # Sử dụng regex để tách thành các phần
        match = re.match(r'gei_(\d+)_([a-z]+-\d+)_(\d+)', name)
        if not match:
            raise ValueError(f"Tên file không đúng định dạng: {filename}")
        subject_id, condition, view = match.groups()
        # Chuẩn hóa ID và view
        subject_id = subject_id.zfill(3)
        view = view.zfill(3)
        return subject_id, condition, view
    except Exception as e:
        print(f"Lỗi phân tích tên file {filename}: {e}")
        return None, None, None

def create_directory_structure(output_dir, subject_id, condition, view):
    """
    Tạo thư mục theo cấu trúc: output_dir/subject_id/condition/view
    """
    target_dir = os.path.join(output_dir, subject_id, condition, view)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def organize_gei_images():
    """
    Tổ chức ảnh GEI vào cấu trúc thư mục ID/condition/view dựa trên tên file
    """
    # Kiểm tra thư mục đầu vào
    if not os.path.exists(config['input_dir']):
        raise FileNotFoundError(f"Thư mục {config['input_dir']} không tồn tại")

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(config['output_dir'], exist_ok=True)

    # Thu thập danh sách ảnh
    samples = []
    for filename in os.listdir(config['input_dir']):
        if filename.endswith('.png'):
            subject_id, condition, view = extract_metadata_from_filename(filename)
            if subject_id and condition and view:
                samples.append({
                    'image_path': os.path.join(config['input_dir'], filename),
                    'subject_id': subject_id,
                    'condition': condition,
                    'view': view
                })

    if not samples:
        raise ValueError("Không tìm thấy ảnh GEI hợp lệ trong thư mục")

    # Tổ chức ảnh vào thư mục
    success_count = 0
    for sample in samples:
        image_path = sample['image_path']
        subject_id = sample['subject_id']
        condition = sample['condition']
        view = sample['view']

        # Kiểm tra ảnh tồn tại
        if not os.path.exists(image_path):
            print(f"Ảnh không tồn tại: {image_path}")
            continue

        # Tạo thư mục đích
        target_dir = create_directory_structure(config['output_dir'], subject_id, condition, view)

        # Tạo tên file đích
        target_filename = os.path.basename(image_path)
        target_path = os.path.join(target_dir, target_filename)

        # Sao chép ảnh
        try:
            shutil.copy(image_path, target_path)  # Sử dụng copy để giữ nguyên ảnh gốc
            print(f"Đã sao chép: {image_path} -> {target_path}")
            success_count += 1
        except Exception as e:
            print(f"Lỗi sao chép {image_path}: {e}")

    print(f"Hoàn tất! Đã tổ chức {success_count}/{len(samples)} ảnh GEI")

def main():
    try:
        organize_gei_images()
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()