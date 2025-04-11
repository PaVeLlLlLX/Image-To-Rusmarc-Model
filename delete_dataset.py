from pathlib import Path
import os
import shutil

src_dir = Path("/home/pavel/VSC/Torch/test")
splits = ['train', 'test']
class_dirs = [d for d in src_dir.iterdir() if d.is_dir() and d.name != 'labels']
for split in splits:
        for class_dir in class_dirs:
            subclass_dirs = [dd for dd in class_dir.iterdir() if dd.is_dir() and dd.name != 'labels']
            for subclass_dir in subclass_dirs:
                images = list(subclass_dir.glob('*.jpg'))
                for img in images:
                    img_target = str(img).split('/')
                    img_target.remove("test")
                    img_target.insert(5, "test")
                    img_target = '/'.join(img_target)
                    #shutil.copy(img, img_target)
                    print(f"{class_dir.name}\{subclass_dir.name}\{img.name}")
                    os.remove(Path(f"{src_dir}/{class_dir.name}/{subclass_dir.name}/{img.name}"))