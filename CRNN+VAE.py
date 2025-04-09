import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import gc
import numpy as np
import os, re, cv2
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from os.path import exists
from torch.utils.data import Dataset
import Levenshtein
from functools import lru_cache
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import BatchSampler
from collections import defaultdict
from torchvision.transforms.functional import crop
from nltk.translate.bleu_score import corpus_bleu
from TorchCRF import CRF
torch.cuda.empty_cache()
gc.collect()
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))  # [B, C]
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))  # [B, C]
        channel_weights = avg_out + max_out  # [B, C]
        channel_weights = torch.sigmoid(channel_weights).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * channel_weights 
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_weights = self.conv(concat)  # [B, 1, H, W]
        spatial_weights = self.sigmoid(spatial_weights)  # [B, 1, H, W]
        return x * spatial_weights
    

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class OCRModel(nn.Module):
    def __init__(self, num_chars, hidden_dim=256):
        super(OCRModel, self).__init__()
        # [B, N, 1, 256, 384]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 256, 384]
        #     nn.Dropout(0.3),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(2, 2),
        #     CBAM(32),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 128, 192]
        #     nn.Dropout(0.3),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(2, 2),
        #     CBAM(64),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 64, 96]
        #     nn.Dropout(0.3),
        #     #nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(2, 2),
        #     CBAM(128),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 32, 48]
        #     #nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(2, 2),
        #     CBAM(256),
        #     nn.AdaptiveAvgPool2d((1, 2161))
        # )

        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children()))[:-2]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 2400))
        #self.attn = nn.Conv1d(256, 1, kernel_size=1)
        #self.rnn = ResidualLSTM(64 * 4, 256)
        #self.attention = LocationAwareAttention(512)
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=1024,
        #     num_heads=4,
        #     dropout=0.1
        # )
        self.rnn = nn.LSTM(input_size=512* (1), 
            hidden_size=hidden_dim, 
            bidirectional=True, 
            num_layers=4, 
            dropout=0.3,
        )
        self.crf_head = nn.Linear(2 * hidden_dim, 60)
        self.crf = CRF(70)

        #self.proj = nn.Linear(256 * 24, 512)
        #config = GPT2Config(n_layer=4, n_embd=512, n_head=4, vocab_size=1000)
        #self.transformer = GPT2LMHeadModel(config)
        #self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        #self.vedm = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        # self.key_layer = nn.Linear(256, 256)
        # self.query_layer = nn.Linear(256, 256)
        # self.energy_layer = nn.Linear(256, 1)
        self.fc = nn.Linear(hidden_dim*2, num_chars + 1)
        #self.fc = nn.Linear(1000, num_chars + 1)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 2161))

    def forward(self, x):
        #x = x.unsqueeze(0)
        feat = self.cnn(x)  # [B, 256, H', W']
        #print(feat.size())
        #print(self.cnn[20].spatial_attention(torch.randint(1, 356, 512)).detach().cpu().numpy())
        # (B, C, H', W') => (W', B, C*H')
        feat = self.adaptive_pool(feat) 
        b, c, h_prime, w_prime = feat.size()
        #print(feat.size())
        feat = feat.permute(1, 0, 3, 2)  # [C, B, W', H']
        feat = feat.contiguous().view(w_prime*h_prime, b, c)  # [W'*H', B, C]

        rnn_out, _ = self.rnn(feat)  # [W', B, 2*hidden_dim]

        crf_emissions = self.crf_head(rnn_out.permute(1, 0, 2))  # [W'*H', B, num_crf_tags]

        logits = self.fc(rnn_out)    # [W', B, num_chars]
        return logits, crf_emissions
        #features = features.permute(3, 0, 1, 2)  # [2161, B, C, H']
        #print(features.size())
        #features = features.flatten(2) # [2161, B, C*H']
        #print(features.size())
        #features = self.proj(features)  # [2161, B, 1024]
        #logits = self.transformer(inputs_embeds=features)
        # print(logits.logits.size())
        #logits = self.fc(logits.logits)
        #return logits, features # [B, 2161, num_chars]
    
    def decode_crf(self, crf_emissions, mask=None):
        return self.crf.viterbi_decode(crf_emissions, mask=mask)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

cudnn.benchmark = True
cudnn.deterministic = False

class GPNTBDataset(Dataset):
    def __init__(self, images_dir, txt_file, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_seq_len = 2400
        self.cards = self._load_cards(txt_file)
        self.image_paths = self._validate_image_paths()
        # " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~¦§©«®°±»ЁЎАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёљў—„•€№™"
        self.alphabet, self.char_to_idx = self._build_alphabet()
        #print(len(self.alphabet), self.alphabet)

    def _load_cards(self, txt_file):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read().split("*****")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки файла {txt_file}: {str(e)}")

    def _validate_image_paths(self):
        paths = []
        pattern = re.compile(r"([^\\/]+)[\\/]([^\\/]+)[\\/]([^\\/]+\.jpg)")
        
        for idx, card in enumerate(self.cards):
            match = pattern.search(card)
            if not match:
                raise ValueError(f"Неверный формат карточки {idx}: {card}")
            # if len(card) > self.max_seq_len:
            #     self.max_seq_len = len(card)
            #     print(self.max_seq_len)
            #     print(card)
            #     print(len(card))
            rel_path = Path(*match.groups())
            full_path = self.images_dir / rel_path
            if len(self.cards[idx]) > self.max_seq_len or len(self.cards[idx]) == 0:
                print(len(self.cards[idx]))
                #print(self.cards[idx])
                #self.cards.pop(idx)
                #continue
            if not full_path.exists():
                #print(f"Изображение {full_path} не найдено")
                # print(match)
                # print(pattern.search(card))
                #print(card)
                #print(len(card))
                raise FileNotFoundError(f"Изображение {full_path} не найдено")
                #print(self.cards[idx])
                #self.cards.pop(idx)
            paths.append(full_path)

        return paths
    
    def _build_alphabet(self):
        unique_chars = set()
        for i, card in enumerate(self.cards):
            card_chars = set(card)
            unique_chars.update(card_chars)
        uc = unique_chars.copy()
        for char in unique_chars:
            if char in "@QWYqЎљў":
                uc.remove(char)

        unique_chars = uc
        #print(len(unique_chars), unique_chars)
        sorted_chars = sorted(unique_chars)
        sorted_chars.insert(0, '@')
        alphabet_str = ''.join(sorted_chars)
        char_mapping = {char: idx for idx, char in enumerate(sorted_chars)}
        print(char_mapping['@'])
        #print(alphabet_str)
        return alphabet_str, char_mapping

    def __len__(self):
        return len(self.cards)

    #@lru_cache(maxsize=2000)
    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        label = self._encode_text(self.cards[idx])
        # print(self.cards[idx])
        # plt.imshow(image[0], cmap='gray')
        # plt.axis('off')
        # plt.show()
        crf_labels = []
        for i, line in enumerate(self.cards[idx].split("\n"), start=1):
            if i >= 60:
                print(f"BIG SIZE OF LINES IN CARD: {self.cards[idx]}")
                break
            #print(line)
            if not line:
                continue
            #if line[:4] == "#22:" or line[:4] == " #22" and i != 1:
            #    i = i - 1

            crf_labels.extend([i] * len(line))
        if len(crf_labels) < self.max_seq_len:
            crf_labels.extend([0] * (self.max_seq_len - len(crf_labels)))
        else:
            print(len(crf_labels))
            #print(crf_labels)
        #print(self._encode_text(''))
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(crf_labels, dtype=torch.long)
    #-----------------------------------------------
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        # Если передается одиночный индекс
        return self._get_single_item(idx)

    # def _get_single_item(self, idx):
    #     image = self._load_image(self.image_paths[idx])
    #     label = self._encode_text(self.cards[idx])
    #     # print(self.cards[idx])
    #     # plt.imshow(image[0], cmap='gray')
    #     # plt.axis('off')
    #     # plt.show()
    #     crf_labels = []
    #     for i, line in enumerate(self.cards[idx].split("\n"), start=1):
    #         #print(line)
    #         if not line:
    #             continue
    #         if line[:3] == "#22:" or line[:3] == " #22" and i != 1:
    #             i = i - 1

    #         crf_labels.extend([i] * len(line))
    #     if len(crf_labels) < self.max_seq_len:
    #         crf_labels.extend([0] * (self.max_seq_len - len(crf_labels)))
    #     else:
    #         print(len(crf_labels))
    #         #print(crf_labels)
    #     #print(self._encode_text(''))
    #     label = label + [0] * (self.max_seq_len - len(label))
    #     return image, torch.tensor(label, dtype=torch.long), torch.tensor(crf_labels, dtype=torch.long)

    @lru_cache(maxsize=1000)
    def _load_image(self, path):
        try:
            image = Image.open(path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки изображения {path}: {str(e)}")

    def _encode_text(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]
  
transform = transforms.Compose([
    #transforms.RandomRotation(4),
    #transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.GaussianBlur(kernel_size=(3, 7)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# line_transform = transforms.Compose([
#     transforms.Resize((24, 256)),
# ])
    
images_dir = '/home/pavel/VSC/Torch/train'
txt_file = '/home/pavel/VSC/Torch/GPNTB/train_labels.txt'

dataset = GPNTBDataset(images_dir, txt_file, transform=transform)
test_dataset = GPNTBDataset('/home/pavel/VSC/Torch/test', '/home/pavel/VSC/Torch/GPNTB/test_labels.txt', transform=transform_test)

def collate_fn(batch):
    try:
        images, labels, crf_labels = zip(*batch)

        images = torch.stack(images)

        max_label_len = max(len(l) for l in labels)
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long, device=images.device)

        for i, l in enumerate(labels):
            if isinstance(l, torch.Tensor):
                padded_labels[i, :len(l)] = l.clone().detach()
            else:
                padded_labels[i, :len(l)] = torch.tensor(l, dtype=torch.long, device=images.device)

        max_crf_len = max(len(c) for c in crf_labels)
        padded_crf = torch.zeros(len(batch), max_crf_len, dtype=torch.long, device=images.device)

        for i, c in enumerate(crf_labels):
            if isinstance(c, torch.Tensor):
                padded_crf[i, :len(c)] = c.clone().detach()
            else:
                padded_crf[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=images.device)

        return images, padded_labels, padded_crf

    except Exception as e:
        print("Ошибка в collate_fn:")
        print(f"Типы элементов: {[type(x) for x in batch[0]]}")
        print(f"Размеры images: {[img.size() if hasattr(img, 'shape') else None for img in images]}")
        print(f"Длины labels: {[len(l) for l in labels]}")
        raise

batch_size = 32

# buckets = defaultdict(list)
# for idx, (_, label, _) in enumerate(dataset):
#     buckets[len(label)].append(idx)

# batch_sampler = []
# for bucket in buckets.values():
#     np.random.shuffle(bucket)
#     batch_sampler.extend([bucket[i:i+batch_size] for i in range(0, len(bucket), batch_size)])

train_loader = DataLoader(dataset, num_workers=3, collate_fn=collate_fn, pin_memory=False, batch_size=batch_size, drop_last=True, shuffle=True)#batch_sampler=BatchSampler(batch_sampler, batch_size=batch_size, drop_last=False))

test_loader = DataLoader(test_dataset, num_workers=3, collate_fn=collate_fn, pin_memory=False,  batch_size=batch_size, drop_last=True)#batch_sampler=BatchSampler(batch_sampler, batch_size=batch_size, drop_last=False))


model = OCRModel(num_chars=len(dataset.alphabet)).to(device, non_blocking=True)
#model = torch.compile(model)

for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

def ctc_decode(output, alphabet):
    _, max_indices = torch.max(output, dim=2)
    batch_texts = []
    for indices in max_indices:
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = indices[indices != 0]
        text = ''.join([alphabet[i] for i in indices if i < len(alphabet)])
        batch_texts.append(text)
    return batch_texts


def calculate_cer(preds, targets, alphabet):
    cer = 0
    true_texts = []
    pred_text = ctc_decode(preds, alphabet)
    for target in targets:
        true_text = ''.join([alphabet[i] for i in target if i != 0])
        true_texts.append(true_text)
    for pred, text in zip(pred_text, true_texts):
        distance = Levenshtein.distance(pred, text)
        cer += distance / max(len(true_text), 1)
    return cer / len(preds)

def loss_with_constraints(output, target, input_lengths, target_lengths):
    # print(output.size())
    # print(target.size())
    # print(input_lengths)
    # print(target_lengths)

    # ce = nn.CrossEntropyLoss(ignore_index=0)
    # T, B, C = output.size()
    # output_ce = output.reshape(B*T, C)
    # #print(output_ce.size(), target.view(-1).size())
    # ce_loss = ce(output_ce, target.view(-1))

    output = F.log_softmax(output, dim=2)
    ctc_loss = F.ctc_loss(
        output,  # [T, B, C]
        target,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        zero_infinity=True,
        blank=0
    ) 
    return 1 * ctc_loss #+ 0.3 * ce_loss

scaler = torch.amp.GradScaler(device=device)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)

num_epochs = 140
for epoch in range(num_epochs):
    for images, labels, crf_labels in train_loader:
        images, labels, crf_labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True), crf_labels.to(device, non_blocking=True)
        #torch.autograd.set_detect_anomaly(True)
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("NaN or Inf in images")
            continue
        optimizer.zero_grad()
        # all_lines = detect_lines_batch(images)
        # cropped_batch = crop_lines_batch(images, all_lines)
        # #print(images.size(), len(cropped_batch), "SIZE")
        # all_texts = []
        # for cropped_lines in cropped_batch:
        #     texts = []
        #     for line_img in cropped_lines:
        #         line_img = line_transform(line_img)
        #         text = model(line_img)
        #         texts.append(text)
        #         #print(ctc_decode(text, dataset.alphabet))
        #     if texts == []:
        #         continue
        #     texts = torch.cat(texts, dim=0)
        #     if texts.size(0) < 2161:
        #         pad_size = 2161 - texts.size(0)
        #         T, B, C = texts.size()
        #         padding = torch.zeros((pad_size, B, C), dtype=texts.dtype, device=device)
        #         texts = torch.cat([texts, padding], dim=0)
        #     elif texts.size(0) > 2161:
        #         texts = texts[:2161, :, :]

        #     #print(texts.size())
        #     all_texts.append(texts)
        # if len(all_texts) != batch_size:
        #     for i in range(batch_size-len(all_texts)):
        #         all_texts.append(torch.randn(2161, 1, 172).to(device=device))
        #     #print(len(all_texts))
        # logits = torch.cat(all_texts, dim=1)
        #print(logits.size())

        with torch.amp.autocast(device_type='cuda'):
            logits, crf_emissions = model(images)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN or Inf in model output")
                continue
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(label[label != 0]) for label in labels], dtype=torch.long, device=device)
            #print(input_lengths, target_lengths)
            if (input_lengths <= 0).any() or (target_lengths <= 0).any():
                print("Invalid input or target lengths")
                continue
            
            if torch.isnan(crf_emissions).any() or torch.isinf(crf_emissions).any():
                print("NaN or inf in emissions!")
                crf_emissions = torch.nan_to_num(crf_emissions)
            loss_ctc = loss_with_constraints(logits.float(), labels, input_lengths, target_lengths)

            #masks = (crf_labels != 0).to(torch.bool).to(device)
            #print(crf_emissions.size(), crf_labels.size(), masks.size())
            #crf_loss = -model.crf(crf_emissions, crf_labels, mask=masks)
            #crf_loss = crf_loss.mean()
            total_loss = loss_ctc #+ 0.5 * crf_loss

        scaler.scale(total_loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #print(loss_ctc.item(), crf_loss)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("NaN or Inf in gradients")
            print(input_lengths, target_lengths)
            print(loss_ctc.item())#, crf_loss)
            optimizer.zero_grad()
            continue

        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    candidates = []
    references = []
    with torch.no_grad():
        cer = []
        for images, labels, crf_labels in test_loader:
            images, labels, crf_labels = images.to(device), labels.to(device), crf_labels.to(device)
            # all_lines = detect_lines_batch(images)
            # cropped_batch = crop_lines_batch(images, all_lines)
            # #print(images.size(), len(cropped_batch), "SIZE")
            # all_texts = []
            # for cropped_lines in cropped_batch:
            #     texts = []
            #     for line_img in cropped_lines:
            #         line_img = line_transform(line_img)
            #         text = model(line_img)
            #         texts.append(text)
            #         #print(ctc_decode(text, dataset.alphabet))
            #     if texts == []:
            #         continue
            #     texts = torch.cat(texts, dim=0)
            #     if texts.size(0) < 2161:
            #         pad_size = 2161 - texts.size(0)
            #         T, B, C = texts.size()
            #         padding = torch.zeros((pad_size, B, C), dtype=texts.dtype, device=device)
            #         texts = torch.cat([texts, padding], dim=0)
            #     elif texts.size(0) > 2161:
            #         texts = texts[:2161, :, :]

            #     #print(texts.size())
            #     all_texts.append(texts)
            # if len(all_texts) != batch_size:
            #     for i in range(batch_size-len(all_texts)):
            #         all_texts.append(torch.randn(2161, 1, 172).to(device=device))
            # logits = torch.cat(all_texts, dim=1)
    #-----------------------------------------------------------------------------------------------------
            logits, _ = model(images)
            cer.append(calculate_cer(logits, labels, dataset.alphabet))
            pred_texts = ctc_decode(logits, dataset.alphabet)  # Декодируем предсказания в текст
            
            for pred, true in zip(pred_texts, labels):
                candidates.append(list(pred))
                references.append([list(true)])  # Эталонные тексты в списке списков
        cer_mean = torch.mean(torch.Tensor(cer))
        score = corpus_bleu(references, candidates)
    decoded_texts = ctc_decode(logits, dataset.alphabet)
    #decoded_fiels = model.decode_crf(crf_emissions, mask=masks)
    s = ''.join(str(x) for x in decoded_texts)
    #--------------------------------------------------------------------------------------------
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}, BLEU: {score}, CER: {cer_mean}, LR: {optimizer.param_groups[0]['lr']}", s)#, decoded_fiels[0][:100])

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'alphabet': train_loader.dataset.alphabet,
    'latent_dim': 128,
    'num_chars': len(train_loader.dataset.alphabet),
}, '/home/pavel/VSC/crnn_checkpoint.pth')

# checkpoint = torch.load('/home/pavel/VSC/crnn_checkpoint.pth', map_location=torch.device('cpu'))

# model = OCRModel(
#     latent_dim=checkpoint['latent_dim'],
#     num_chars=len(checkpoint['alphabet']),
# )
# model.load_state_dict(checkpoint['model_state_dict'])
# model = model.to("cuda")
# model.eval()

# alphabet = checkpoint['alphabet']


test_path = "/home/pavel/VSC/Torch/train/0007/Абдуллин/00001.jpg"
image = Image.open(test_path).convert('L')
image = transform_test(image).to("cuda")
alphabet = train_loader.dataset.alphabet
print(alphabet)

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


model.eval()
with torch.no_grad():
    output, _ = model(image.unsqueeze(0))
    #output.permute()
    #spatial_weights = model.cnn[20].spatial_attention(image.unsqueeze(0)).detach().cpu().numpy()
    #plt.imshow(spatial_weights[0, 0], cmap='jet')
    #plt.colorbar()
    #plt.show()
    print("OUTPUT",output)
    #recon = recon.to("cpu")
    #show_image(recon[0][0])
print(len(alphabet))
decoded_texts = ctc_decode(output, alphabet)
s = ''.join(str(x) for x in decoded_texts)
print(s)

show_image(image.to("cpu")[0])


