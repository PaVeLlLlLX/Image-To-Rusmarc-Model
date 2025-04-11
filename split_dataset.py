from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(root_dir: str, 
                 train_ratio: float = 0.9,
                 test_ratio: float = 0.1,
                 seed: int = 42) -> None:
    """
        root_dir (str): Путь к корневой директории датасета
        train_ratio (float): Доля тренировочных данных
        test_ratio (float): Доля тестовых данных
        seed (int): Random seed для воспроизводимости
    """
    cards = []
    with open("/home/pavel/VSC/Torch/GPNTB/AKO - введенные.TXT", "r", encoding="windows-1251") as file:
        cards = file.read().split("*****")

    src_dir = Path(root_dir)
    class_dirs = [d for d in src_dir.iterdir() if d.is_dir() and d.name != 'labels']
    
    splits = ['train', 'test']
    filtered_labels_train = []
    filtered_labels_test = []
    for split in splits:
        (src_dir.parent / split).mkdir(exist_ok=True)
        for class_dir in class_dirs:
            (src_dir.parent / split / class_dir.name).mkdir(exist_ok=True)
            subclass_dirs = [dd for dd in class_dir.iterdir() if dd.is_dir() and dd.name != 'labels']
            for subclass_dir in subclass_dirs:
                (src_dir.parent / split / class_dir.name / subclass_dir.name).mkdir(exist_ok=True, parents=True)
    
            
                print(subclass_dir)
                images = list(subclass_dir.glob('*.jpg'))
                if len(images) == 1:
                    print(subclass_dir, images)
                    for img in images:
                        img_target = str(img).split('/')
                        img_target.remove("GPNTB")
                        img_target.insert(5, "train")
                        img_target = '/'.join(img_target)
                        shutil.copy(img, img_target)
                        #print(class_dir)
                        #print(subclass_dir)
                        #print(img_target)
                        for card in cards:
                            #print(f"{class_dir.name}\{subclass_dir.name}\{img.name}")
                            if f"{class_dir.name}\{subclass_dir.name}\{img.name}" in card:
                                filtered_labels_train.append(card)
                    continue
                
                train, test = train_test_split(
                    images, 
                    test_size=test_ratio,
                    train_size=train_ratio,
                    random_state=seed
                )
                if split == 'train':
                    for img in train:
                        img_target = str(img).split('/')
                        img_target.remove("GPNTB")
                        img_target.insert(5, "train")
                        img_target = '/'.join(img_target)
                        shutil.copy(img, img_target)
                        #print(class_dir)
                        #print(subclass_dir)
                        #print(img_target)
                        for card in cards:
                            #print(f"{class_dir.name}\{subclass_dir.name}\{img.name}")
                            if f"{class_dir.name}\{subclass_dir.name}\{img.name}" in card:
                                filtered_labels_train.append(card)
                else:
                    for img in test:
                        img_target = str(img).split('/')
                        img_target.remove("GPNTB")
                        img_target.insert(5, "test")
                        img_target = '/'.join(img_target)
                        shutil.copy(img, img_target)
                        for card in cards:
                            if f"{class_dir.name}\{subclass_dir.name}\{img.name}" in card:
                                filtered_labels_test.append(card)

    my_file = open("/home/pavel/VSC/Torch/train_labels.txt", "w+")
    filtered_labels_train = '*****'.join(filtered_labels_train)
    my_file.write(filtered_labels_train)
    my_file.close()

    my_file = open("/home/pavel/VSC/Torch/test_labels.txt", "w+")
    filtered_labels_test = '*****'.join(filtered_labels_test)
    my_file.write(filtered_labels_test)
    my_file.close()


if __name__ == '__main__':
    split_dataset(
        root_dir="/home/pavel/VSC/Torch/GPNTB",
        train_ratio=0.9,
        test_ratio=0.1,
        seed=42,
    )