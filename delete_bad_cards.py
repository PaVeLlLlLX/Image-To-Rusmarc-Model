import re, os
from pathlib import Path
images_dir = '/home/pavel/VSC/Torch/train'
images_dir = Path(images_dir)
with open('/home/pavel/VSC/Torch/train_labels.txt', "r", encoding="utf-8") as f:
    cards = f.read().split("*****")
    filtered_cards = []
    removed_images = []
    for i, card in enumerate(cards):
        if ".jpg\n#952:" in card:
            pattern = re.compile(r"([^\\/]+)[\\/]([^\\/]+)[\\/]([^\\/]+\.jpg)")
            match = pattern.search(card)

            if match:
                rel_path = Path(*match.groups())
                full_path = images_dir / rel_path
                if full_path.exists():
                    os.remove(full_path)
                    removed_images.append(str(full_path))
                continue 
        filtered_cards.append(card)
    
    filtered_cards = [c for c in filtered_cards if c.strip()]
    
with open('/home/pavel/VSC/Torch/train_labels.txt', "w", encoding="utf-8") as f:
    f.write("*****".join(filtered_cards))

print(f"Удалено {len(removed_images)} изображений:")
for img in removed_images:
    print(f" - {img}")
print(f"\nСохранено {len(filtered_cards)} меток в файле")