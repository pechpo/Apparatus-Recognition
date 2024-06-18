import json
import shutil
from pathlib import Path

count = 0

def process_single_object_images(dir_path, label):
    global count
    annotations = []
    for filepath in dir_path.iterdir():
        count += 1
        new_filename = f"{count}" + filepath.suffix
        shutil.copy(filepath, Path(f"./train/pic/{new_filename}"))
        annotation = {"image": new_filename, "labels": [label]}
        annotations.append(annotation)
    return annotations

def process_multi_object_images(dir_path):
    global count
    annotations = []
    for filepath in dir_path.iterdir():
        count += 1
        new_filename = f"{count}" + filepath.suffix
        shutil.copy(filepath, Path(f"./train/pic/{new_filename}"))
        labels = filepath.stem.split(' ')
        annotation = {"image": new_filename, "labels": labels}
        annotations.append(annotation)
    return annotations

def main():
    base_path = Path("./raw")
    single_object_path = base_path / "单物体"
    multi_object_path = base_path / "多物体"

    annotations = []

    train_pic_path = Path("./train/pic")
    train_pic_path.mkdir(parents=True, exist_ok=True)

    # Process single object images
    for folder in single_object_path.iterdir():
        if folder.is_dir():
            annotations.extend(process_single_object_images(folder, folder.name))

    # Process multi object images
    annotations.extend(process_multi_object_images(multi_object_path))

    # Save annotations to a JSON file
    with open('./train/label.json', 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
