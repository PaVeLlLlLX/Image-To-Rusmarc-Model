import cv2
import opendatasets as od
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import numpy as np

#od.download(
#    "https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset/data",
#    force=True,
#    data_dir="/home/pavel/VSC/Torch/"
#)

class CyrillicDataset(Dataset):
    def __init__(self, images_dir, tsv_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        self.labels_df = pd.read_csv(tsv_file, sep='\t', header=None, names=['image', 'label'])
        self.labels_df = self.labels_df.dropna()

        all_chars = ''.join(self.labels_df['label'].astype(str).unique())
        self.alphabet = ''.join(sorted(set(all_chars)))
        self.alphabet += ' '
        for label in self.labels_df['label']:
            for char in label:
                if char not in self.alphabet:
                    raise ValueError(f"Символ '{char}' отсутствует в алфавите.")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        image_filename = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        image_path = os.path.join(self.images_dir, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found!")

        image = self.load_image(image_path)
        label_indices = self.encode_text(label)

        return image, torch.tensor(label_indices, dtype=torch.long)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

    def encode_text(self, text):
        label_indices = [self.alphabet.find(char) for char in text]
        return label_indices

transform = transforms.Compose([
    #transforms.RandomRotation(10),
    #transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    #transforms.GaussianBlur(kernel_size=(3, 7)),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((64, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

images_dir = '/home/pavel/VSC/Torch/cyrillic-handwriting-dataset/train'
tsv_file = '/home/pavel/VSC/Torch/cyrillic-handwriting-dataset/train.tsv'

dataset = CyrillicDataset(images_dir, tsv_file, transform=transform)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

class CRNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=108, hidden_size=256):
        super(CRNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.rnn = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5,)

        self.fc = nn.Linear(hidden_size*2, output_channels)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.max_pool2d(x, (2, 2))
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.max_pool2d(x, (2, 2))
        x = self.batch_norm2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = F.max_pool2d(x, (2, 2))
        x = self.batch_norm2(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        x = self.batch_norm3(x)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.01)

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, width, channels * height)

        x, _ = self.rnn(x)

        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def loss_with_constraints(output, target, input_lengths, target_lengths,):
    #year_loss = year_constraint_loss(output, year_info)

    #author_loss = author_constraint_loss(output, author_info)

    #total_loss = ce_loss + alpha * year_loss + beta * author_loss
    #return total_loss
    print(output.size(), target.size())
    output = output.permute(1, 0, 2)

    loss = F.ctc_loss(output, target, input_lengths, target_lengths, blank=0, zero_infinity=True)
    return loss

def year_constraint_loss(output, year_info):
    predicted_year = output[:, -4:]
    target_year = year_info
    year_loss = F.mse_loss(predicted_year, target_year)
    return year_loss


def author_constraint_loss(output, author_info):
    predicted_author = output[:, :-4]
    target_author = author_info
    author_loss = F.mse_loss(predicted_author, target_author)
    return author_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN().to(device)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

num_epochs = 20
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if torch.isnan(images).any() or torch.isinf(images).any():
            print("NaN or Inf in images")
            continue

        optimizer.zero_grad()

        output = model(images)
        output = F.log_softmax(output, dim=2)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("NaN or Inf in model output")
            continue

        input_lengths = torch.full((images.size(0),), output.size(1), dtype=torch.long, device=device)
        target_lengths = torch.tensor([len(label[label != 0]) for label in labels], dtype=torch.long, device=device)

        if (input_lengths <= 0).any() or (target_lengths <= 0).any():
            print("Invalid input or target lengths")
            continue

        loss = loss_with_constraints(output, labels, input_lengths, target_lengths)
        loss.backward()



        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print("NaN or Inf in gradients")
                    optimizer.zero_grad()
                    continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

'''
checkpoint = torch.load('/home/pavel/VSC/crnn_checkpoint.pth', map_location=torch.device('cpu'))

model = CRNN(
    input_channels=checkpoint['input_channels'],
    output_channels=len(checkpoint['alphabet']),
    hidden_size=checkpoint['hidden_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

alphabet = checkpoint['alphabet']
'''

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'alphabet': dataloader.dataset.alphabet,
    'hidden_size': 256,
    'input_channels': 1
}, '/home/pavel/VSC/crnn_checkpoint.pth')

def ctc_decode(output, alphabet):
    _, max_indices = torch.max(output, dim=2)

    batch_texts = []
    for indices in max_indices:
        indices = torch.unique_consecutive(indices, dim=-1)

        indices = indices[indices != 0]

        text = ''.join([alphabet[i] for i in indices if i < len(alphabet)])
        batch_texts.append(text)

    return batch_texts

from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_path = "/content/drive/MyDrive/Снимок экрана от 2025-01-27 16-37-15.png"
image = Image.open(test_path).convert('L')
image = transform(image).to("cpu")
alphabet = dataloader.dataset.alphabet

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

show_image(image[0])

model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0).to("cuda"))
    print(output)
decoded_texts = ctc_decode(output, alphabet)
print(decoded_texts)

def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

image = cv2.imread("/content/drive/MyDrive/карточка.jpeg", 0)
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = image[y:y+h, x:x+w]
    show_image(cropped)
    #resized = cv2.resize(cropped, (256, 64))

    resized = resize_with_padding(cropped, (256, 64))
    #resized = resized[:, np.newaxis]
    #resized = torch.Tensor(resized)
    #print(resized.size())
    #resized = resized.permute(1, 0, 2)
    resized = Image.fromarray(resized)
    show_image(resized)
    resized = transform(resized)

    output = model(resized.unsqueeze(0).to("cuda"))
    decoded_texts = ctc_decode(output, alphabet)
    print(decoded_texts)