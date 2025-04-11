import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import gc
import numpy as np
from itertools import islice
import math
from tqdm import tqdm
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
from torch.utils.data import BatchSampler
from collections import defaultdict
from torchvision.transforms.functional import crop
from nltk.translate.bleu_score import corpus_bleu
torch.cuda.empty_cache()
gc.collect()
from torchvision.models import resnet18
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm
from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

cudnn.benchmark = True
cudnn.deterministic = False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model] -> [max_len, 1, d_model] -> needs [SeqLen, Batch, Dim] for batch_first=False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, Dim] if batch_first=False
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

class ImageToRusmarcModel(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.2, sos_token_id=1, eos_token_id=2, pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_tokens = num_tokens

        # --- Энкодер ---
        cnn_backbone = resnet18(pretrained=True)
        self.cnn_layers = nn.Sequential(*list(cnn_backbone.children())[:-2]) # [B, 512, H', W']
        
        self.cnn_channel_proj = nn.Conv2d(cnn_backbone.fc.in_features, d_model, kernel_size=1)
        self.encoder_pos_encoder = PositionalEncoding(d_model, dropout) # Позиционное кодирование для признаков энкодера

        # --- Декодер ---
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.output_layer = nn.Linear(d_model, num_tokens)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_img):
        # src_img: [B, C, H, W]
        features = self.cnn_layers(src_img) # [B, C_cnn, H', W']
        features = self.cnn_channel_proj(features) # [B, d_model, H', W']

        b, d, h_prime, w_prime = features.shape
        # Преобразование в [SeqLen, Batch, Dim] для Transformer (batch_first=False)
        # SeqLen = H' * W'
        memory = features.flatten(2) # [B, d_model, H'*W']
        memory = memory.permute(2, 0, 1) # [H'*W', B, d_model] - Это память для декодера

        # Добавляем позиционное кодирование к памяти энкодера
        memory = self.encoder_pos_encoder(memory)
        return memory # [SeqLen_enc, B, d_model]

    def decode(self, tgt_seq, memory, tgt_mask, tgt_padding_mask):
        # tgt_seq: [SeqLen_dec, B] (batch_first=False)
        # memory: [SeqLen_enc, B, d_model]
        # tgt_mask: [SeqLen_dec, SeqLen_dec]
        # tgt_padding_mask: [B, SeqLen_dec]

        # Эмбеддинг и позиционное кодирование для входа декодера
        tgt_emb = self.token_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pos_encoder(tgt_emb) # [SeqLen_dec, B, d_model]

        # Проход через декодер
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None, # Маска для энкодера обычно не нужна
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None # Можно добавить, если есть паддинг в энкодере
        ) # [SeqLen_dec, B, d_model]

        # Выходной слой
        logits = self.output_layer(output) # [SeqLen_dec, B, num_tokens]
        return logits

    def forward(self, src_img, tgt_seq):
        # src_img: [B, C, H, W]
        # tgt_seq: [B, T] - Токены целевой последовательности (batch_first=True из DataLoader)

    
        tgt_in = tgt_seq[:, :-1] # Убираем EOS
        tgt_in = tgt_in.permute(1, 0) # [T-1, B]

        # --- Подготовка масок ---
        device = src_img.device
        tgt_seq_len = tgt_in.size(0)
        # Маска, чтобы декодер не смотрел вперед
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device) # [T-1, T-1]
        # Маска для паддинг-токенов во входной последовательности декодера
        tgt_padding_mask = (tgt_in.T == self.pad_token_id) # [B, T-1]


        # --- Энкодер ---
        memory = self.encode(src_img) # [SeqLen_enc, B, d_model]

        # --- Декодер ---
        logits = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask) # [T-1, B, num_tokens]

        return logits #[SeqLen, Batch, Classes] для удобства расчета лосса

    def generate(self, image, max_len=1000): #batch_first=True
        batch_size = image.size(0)
        device = image.device

        memory = self.encode(image) # [SeqLen_enc, B, d_model]

        tgt_tokens = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device) # [B, 1]

        for _ in range(max_len - 1):
            tgt_in_step = tgt_tokens.permute(1, 0) # [current_len, B]
            tgt_mask_step = generate_square_subsequent_mask(tgt_in_step.size(0), device) # [current_len, current_len]
            tgt_padding_mask_step = (tgt_tokens == self.pad_token_id) # [B, current_len]

            # --- Декодируем один шаг ---
            output_step = self.decode(tgt_in_step, memory, tgt_mask_step, tgt_padding_mask_step) # [current_len, B, num_tokens]
            # Берем предсказание для последнего токена
            logits_last_token = output_step[-1, :, :] # [B, num_tokens]

            # --- Выбираем следующий токен (Greedy) ---
            next_token = logits_last_token.argmax(dim=-1, keepdim=True) # [B, 1]

            # --- Добавляем к последовательности ---
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1) # [B, current_len + 1]

            # --- Условие остановки (если все сгенерировали EOS) ---
            if (next_token == self.eos_token_id).all():
                 break

        return tgt_tokens # [B, final_len]
    

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
            rel_path = Path(*match.groups())
            full_path = self.images_dir / rel_path
            if len(self.cards[idx]) > self.max_seq_len or len(self.cards[idx]) == 0:
                print(len(self.cards[idx]))
                #print(self.cards[idx])
                #self.cards.pop(idx)
                #continue
            if not full_path.exists():
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

        sorted_chars = sorted(list(unique_chars))

        special_tokens = ['@', '<SOS>', '<EOS>'] # '@' - PAD
        for sp in special_tokens:
            if sp in sorted_chars:
                sorted_chars.remove(sp)

        final_alphabet = special_tokens + sorted_chars
        char_mapping = {char: idx for idx, char in enumerate(final_alphabet)}

        self.pad_token_id = char_mapping['@']
        self.sos_token_id = char_mapping['<SOS>']
        self.eos_token_id = char_mapping['<EOS>']
        print(f"PAD ID: {self.pad_token_id}, SOS ID: {self.sos_token_id}, EOS ID: {self.eos_token_id}")
        print(f"Размер словаря: {len(final_alphabet)}")
        
        print(char_mapping['@'])
        return final_alphabet, char_mapping

    def __len__(self):
        return len(self.cards)

    #@lru_cache(maxsize=2000)
    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        target_sequence_str = self.alphabet[self.sos_token_id] + self.cards[idx] + self.alphabet[self.eos_token_id]
        label = self._encode_text(target_sequence_str)
        # print(self.cards[idx])
        # plt.imshow(image[0], cmap='gray')
        # plt.axis('off')
        # plt.show()
        return image, torch.tensor(label, dtype=torch.long)

    @lru_cache(maxsize=1000)
    def _load_image(self, path):
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки изображения {path}: {str(e)}")

    def _encode_text(self, text):
        return [self.char_to_idx.get(char, self.pad_token_id) for char in text]


transform = transforms.Compose([
    #transforms.RandomRotation(4),
    #transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
images_dir = '/home/pavel/VSC/Torch/train'
txt_file = '/home/pavel/VSC/Torch/train_labels.txt'

dataset = GPNTBDataset(images_dir, txt_file, transform=transform)
test_dataset = GPNTBDataset('/home/pavel/VSC/Torch/test', '/home/pavel/VSC/Torch/test_labels.txt', transform=transform_test)

def collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images)
    labels = [item[1] for item in batch]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=dataset.pad_token_id)
    return images, padded_labels

batch_size = 4


train_loader = DataLoader(dataset, num_workers=4, collate_fn=collate_fn, pin_memory=False, batch_size=batch_size, drop_last=True, shuffle=True)

test_loader = DataLoader(test_dataset, num_workers=4, collate_fn=collate_fn, pin_memory=False,  batch_size=batch_size, drop_last=True)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)

model = ImageToRusmarcModel(num_tokens=len(dataset.alphabet), eos_token_id=dataset.eos_token_id, pad_token_id=dataset.pad_token_id, sos_token_id=dataset.sos_token_id).to(device, non_blocking=True)
#model = torch.compile(model)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00002)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

def decode_sequence(token_ids, alphabet, sos_token_id, eos_token_id, pad_token_id):
    chars = []
    for token_id in token_ids:
        token_id = token_id.item()

        if token_id in [sos_token_id, eos_token_id, pad_token_id]:
            continue

        if 0 <= token_id < len(alphabet): 
             if token_id != pad_token_id:
                 chars.append(alphabet[token_id])

    return "".join(chars)

def calculate_seq2seq_cer(model: ImageToRusmarcModel, 
                          dataloader: DataLoader, 
                          device, 
                          alphabet, 
                          sos_token_id, 
                          eos_token_id, 
                          pad_token_id,
                          subset_percentage: float = 0.1):
    
    if not (0 < subset_percentage <= 1.0):
        raise ValueError("subset_percentage must be between 0 (exclusive) and 1.0 (inclusive)")

    model.eval()
    total_cer = 0.0
    total_samples = 0
    
    try:
        num_batches_total = len(dataloader)
        if num_batches_total == 0:
            print("Warning: DataLoader is empty.")
            model.train()
            return 0.0
    except TypeError:
         print("Warning: DataLoader length not available, calculating CER on the full dataset.")
         subset_percentage = 1.0
         num_batches_total = None

    if subset_percentage < 1.0 and num_batches_total is not None:
        batches_to_take = max(1, math.ceil(num_batches_total * subset_percentage))
        print(f"Calculating CER on a subset: {batches_to_take}/{num_batches_total} batches ({subset_percentage*100:.1f}%).")
        subset_iterable = islice(dataloader, batches_to_take)
        total_for_tqdm = batches_to_take
    else:
        print("Calculating CER on the full dataset.")
        subset_iterable = dataloader
        total_for_tqdm = num_batches_total

    progress_bar = tqdm(subset_iterable, 
                        desc="Calculating CER", 
                        leave=True, 
                        ncols=100, 
                        total=total_for_tqdm)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            batch_size = images.size(0)
            
            current_max_len = labels.shape[1] + 10
            pred_token_ids = model.generate(images, max_len=current_max_len)
            pred_token_ids = pred_token_ids.cpu()
            labels = labels.cpu()

            batch_cer = 0.0
            for i in range(batch_size):
                pred_text = decode_sequence(pred_token_ids[i], alphabet, sos_token_id, eos_token_id, pad_token_id)
                true_text = decode_sequence(labels[i], dataloader.dataset.alphabet, dataloader.dataset.sos_token_id, dataloader.dataset.eos_token_id, dataloader.dataset.pad_token_id)
                
                distance = Levenshtein.distance(pred_text, true_text)
                cer = distance / max(len(true_text), 1) 
                batch_cer += cer

            total_cer += batch_cer
            total_samples += batch_size
            
            current_avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
            progress_bar.set_postfix({'Avg CER': f'{current_avg_cer:.4f}'})

    model.train()

    if total_samples == 0:
       print("Warning: No samples processed for CER calculation in the subset.")
       return 0.0
       
    final_avg_cer = total_cer / total_samples
    return final_avg_cer


#scaler = torch.amp.GradScaler(device=device)
#torch.autograd.set_detect_anomaly(True)
torch.autograd.profiler.profile(False)

num_epochs = 40
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = 10000
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)

fixed_images, fixed_labels = test_loader[10]
print(fixed_labels)
best_loss = 100
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        #torch.autograd.set_detect_anomaly(True)
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("NaN or Inf in images")
            continue
        optimizer.zero_grad()

        #with torch.amp.autocast(device_type='cuda'):
        logits = model(images, labels)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or Inf in model output")
            continue
        input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
        target_lengths = torch.tensor([len(label[label != 0]) for label in labels], dtype=torch.long, device=device)
        #print(input_lengths, target_lengths)
        if (input_lengths <= 0).any() or (target_lengths <= 0).any():
            print("Invalid input or target lengths")
            continue

        targets = labels[:, 1:].permute(1, 0).reshape(-1) # [ (T-1) * B ]
        logits_flat = logits.reshape(-1, logits.size(-1))
        loss_ce = criterion(logits_flat, targets)


        #scaler.scale(total_loss).backward()
        loss_ce.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #print(loss_ctc.item())
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("NaN or Inf in gradients")
            #print(input_lengths, target_lengths)
            print(loss_ce.item())
            optimizer.zero_grad()
            continue
        optimizer.step()
        #scaler.step(optimizer)
        #scaler.update()
        scheduler.step()
    if loss_ce.item() < best_loss:
        best_loss = loss_ce.item()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'alphabet': dataset.alphabet,
            'eos_token_id': dataset.eos_token_id,
            'sos_token_id': dataset.sos_token_id,
            'pad_token_id': dataset.pad_token_id,
            'num_chars': len(dataset.alphabet),
        }, '/home/pavel/VSC/TransformerRUSMARC_checkpoint.pth')
    candidates = []
    references = []
    pad_token_id = dataset.pad_token_id
    sos_token_id = dataset.sos_token_id
    eos_token_id = dataset.eos_token_id
    alphabet = dataset.alphabet
    cer_mean = 0
    cer_mean = calculate_seq2seq_cer(
        model,
        test_loader,
        device,
        alphabet,
        sos_token_id,
        eos_token_id,
        pad_token_id,
        subset_percentage=0.1,
    )
    model.eval()
    with torch.no_grad():
        fixed_images = fixed_images.to(device, non_blocking=True)
        pred_token_ids_example = model.generate(fixed_images, max_len=fixed_labels.shape[1] + 10)
        pred_token_ids_example = pred_token_ids_example.cpu()
        labels_example = fixed_labels.cpu()
        example_pred_text = decode_sequence(pred_token_ids_example[0], alphabet, sos_token_id, eos_token_id, pad_token_id)
        example_true_text = decode_sequence(labels_example[0], test_dataset.alphabet, test_dataset.sos_token_id, test_dataset.eos_token_id, test_dataset.pad_token_id)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_ce.item()}, CER: {cer_mean:.4f}, LR: {optimizer.param_groups[0]['lr']}")
    print("-" * 40)
    print(f"Example Pred:\n{example_pred_text}")
    print("-" * 40)
    print(f"Example True:\n{example_true_text}")
    print("-" * 40)
    model.train()

    print(logits.min(), logits.max(), logits.mean())
    print("Logits flat shape:", logits_flat.shape)
    print("Targets shape:", targets.shape)
    print("Targets min/max:", targets.min().item(), targets.max().item())
    print("Num tokens:", model.num_tokens)
    print("Pad token ID used in loss:", criterion.ignore_index)
    assert targets.min() >= 0
    assert targets.max() < model.num_tokens


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'alphabet': dataset.alphabet,
    'eos_token_id': dataset.eos_token_id,
    'sos_token_id': dataset.sos_token_id,
    'pad_token_id': dataset.pad_token_id,
    'num_chars': len(dataset.alphabet),
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


test_path = "/home/pavel/VSC/Torch/test/0011/Абрамина/00019.jpg"
image = Image.open(test_path).convert('RGB')
image = transform_test(image).to("cuda")
alphabet = train_loader.dataset.alphabet
print(alphabet)

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


model.eval()
with torch.no_grad():
    pred_token_ids_example = model.generate(image, max_len=1200)
    pred_token_ids_example = pred_token_ids_example.cpu()
    example_pred_text = decode_sequence(pred_token_ids_example[0], alphabet, sos_token_id, eos_token_id, pad_token_id)
print(len(alphabet))
s = ''.join(str(x) for x in example_pred_text)
print(s)

show_image(image.to("cpu")[0])


