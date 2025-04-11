from pathlib import Path
import os
import shutil

cards = []
with open('/home/pavel/VSC/Torch/train_labels.txt', "r", encoding="utf-8") as f:
    cards = f.read().split("*****")
filtered_labels_test = []
src_dir = Path("/home/pavel/VSC/Torch/GPNTB")
splits = ['train', 'test']
(src_dir.parent / splits[1]).mkdir(exist_ok=True)
class_dirs = [d for d in src_dir.iterdir() if d.is_dir() and d.name != 'labels']
for class_dir in class_dirs:
    (src_dir.parent / splits[1] / class_dir.name).mkdir(exist_ok=True)
    subclass_dirs = [dd for dd in class_dir.iterdir() if dd.is_dir() and dd.name != 'labels']
    for subclass_dir in subclass_dirs:
        (src_dir.parent / splits[1] / class_dir.name / subclass_dir.name).mkdir(exist_ok=True, parents=True)
        images = list(subclass_dir.glob('*.jpg'))
        if len(images) == 1:
            #print(subclass_dir, images)
            for img in images:
                img_target = str(img).split('/')
                img_target.remove("GPNTB")
                img_target.insert(5, "test")
                img_target = '/'.join(img_target)
                #print(class_dir)
                #print(subclass_dir)
                #print(img_target)
                flag = 0
                for card in cards:
                    #print(f"{class_dir.name}\{subclass_dir.name}\{img.name}")
                    if f"{class_dir.name}\{subclass_dir.name}\{img.name}" in card:
                        shutil.copy(img, img_target)
                        print(img)
                        flag = 1
                if flag == 0:
                    filtered_labels_test.append(card)
            continue
        for img in images:
            img_target = str(img).split('/')
            img_target.remove("GPNTB")
            img_target.insert(5, "test")
            img_target = '/'.join(img_target)
            #print(img)
            flag = 0
            for card in cards:
                if f"{class_dir.name}\{subclass_dir.name}\{img.name}" in card:
                    shutil.copy(img, img_target)
                    print(img)
                    flag = 1
            if flag == 0:
                filtered_labels_test.append(card)

my_file = open("/home/pavel/VSC/Torch/test_labels.txt", "w+")
filtered_labels_test = '*****'.join(filtered_labels_test)
my_file.write(filtered_labels_test)
my_file.close()