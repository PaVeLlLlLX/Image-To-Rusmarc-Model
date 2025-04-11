import torch
import numpy as np
import shap
import lime
from lime import lime_image
from skimage.segmentation import slic # Для сегментации LIME
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from TransformerOCR import ImageToRusmarcModel, generate_square_subsequent_mask
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

checkpoint = torch.load('/home/pavel/VSC/crnn_checkpoint.pth', map_location=torch.device('cpu'))

model = ImageToRusmarcModel(
    num_tokens=len(checkpoint['num_chars']),
    eos_token_id=checkpoint['eos_token_id'],
    sos_token_id=checkpoint['sos_token_id'],
    pad_token_id=checkpoint['pad_token_id']
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to("cuda")

alphabet = checkpoint['alphabet']
sos_token_id = checkpoint['sos_token_id']
eos_token_id = checkpoint['eos_token_id']
pad_token_id = checkpoint['pad_token_id']

transform_test = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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


image_path = "/home/pavel/VSC/Torch/test/0011/Абрамина/00019.jpg"
pil_image = Image.open(image_path).convert('RGB')
img_tensor = transform_test(pil_image).unsqueeze(0).to(device)

# --- Определим, что мы хотим объяснить ---
# Например, вероятность генерации ПЕРВОГО СИМВОЛА ПОСЛЕ <SOS>
# Или вероятность генерации N-го символа
time_step_to_explain = 1 # Объясняем первый символ (индекс 1, т.к. 0 это <SOS>)


with torch.no_grad():
    predicted_tokens = model.generate(img_tensor, max_len=time_step_to_explain + 2) # Генерируем чуть дальше
target_token_id = predicted_tokens[0, time_step_to_explain].item()
target_char = alphabet[target_token_id] if 0 <= target_token_id < len(alphabet) else "UNKNOWN"
print(f"Будем объяснять генерацию символа '{target_char}' (ID: {target_token_id}) на шаге {time_step_to_explain}")

def get_probabilities_at_step_t(image_batch_tensor, step_t):
    """
    Принимает батч тензоров изображений и возвращает вероятности
    токенов на шаге генерации step_t.
    """
    model.eval()
    image_batch_tensor = image_batch_tensor.to(device)

    batch_size = image_batch_tensor.size(0)
    # Начинаем генерацию с <SOS>
    current_tokens = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
    all_logits = []

    with torch.no_grad():
        memory = model.encode(image_batch_tensor) # Энкодируем один раз

        for t in range(step_t + 1): # Генерируем до нужного шага включительно
            tgt_in_step = current_tokens.permute(1, 0) # [current_len, B]
            tgt_mask_step = generate_square_subsequent_mask(tgt_in_step.size(0), device)
            # Padding mask не так важна здесь, т.к. мы не используем <PAD> в генерации пока
            tgt_padding_mask_step = (current_tokens == pad_token_id) # [B, current_len]

            output_step = model.decode(tgt_in_step, memory, tgt_mask_step, tgt_padding_mask_step)
            # output_step: [current_len, B, num_tokens]
            logits_last_token = output_step[-1, :, :] # [B, num_tokens]

            if t < step_t:
                # Если еще не целевой шаг, выбираем следующий токен (жадно) и продолжаем
                next_token = logits_last_token.argmax(dim=-1, keepdim=True) # [B, 1]
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
            else:
                 # Если это целевой шаг (t == step_t), сохраняем логиты
                 # Возвращаем вероятности softmax
                 probabilities = torch.softmax(logits_last_token, dim=-1) # [B, num_tokens]
                 return probabilities.cpu().numpy() # Возвращаем NumPy массив

    # Если цикл завершился раньше (маловероятно при step_t >= 0)
    return np.zeros((batch_size, len(alphabet)))

# --- Prediction Function для LIME ---
def predict_proba_lime(numpy_images):
    """
    Обертка для LIME. Принимает NumPy массив [N, H, W, C],
    возвращает вероятности [N, num_tokens] на шаге time_step_to_explain.
    """
    # 1. Преобразование NumPy -> Tensor
    # LIME дает (N, H, W, C), модель ждет (N, C, H, W)
    # Также применяем нормализацию ImageNet, если она использовалась при обучении
    # (ToTensor() уже применен LIME неявно, если подавать PIL)
    # Нормализация важна!
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Конвертируем в тензор, если вход - numpy с float64/uint8
    if isinstance(numpy_images[0,0,0,0], np.uint8):
         tensor_images = torch.tensor(numpy_images / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
    else: # Предполагаем, что float в диапазоне 0-1
         tensor_images = torch.tensor(numpy_images, dtype=torch.float32).permute(0, 3, 1, 2)

    tensor_images = tensor_images.to(device)
    # Применяем нормализацию
    tensor_images = (tensor_images - mean) / std

    # 2. Получение вероятностей от модели
    probabilities = get_probabilities_at_step_t(tensor_images, time_step_to_explain)
    return probabilities

explainer = lime_image.LimeImageExplainer(verbose=True)
# Сегментация изображения (нужно для возмущений LIME)
# slic генерирует (H, W), нужно (H, W, C) для LIME Image
pil_image_np = np.array(pil_image) # PIL -> NumPy [H, W, C]
segments_slic = slic(pil_image_np, n_segments=100, compactness=30, sigma=3, start_label=1)

print("Запуск объяснения LIME...")
explanation = explainer.explain_instance(
    image=pil_image_np, # NumPy array [H, W, C]
    classifier_fn=predict_proba_lime,
    top_labels=5, # Объясняем топ-5 предсказанных классов (токенов) на шаге t
    hide_color=0, # Цвет для скрытия суперпикселей (0 - черный)
    num_samples=1000, # Количество возмущений (чем больше, тем точнее, но дольше)
    segmentation_fn=lambda img: segments_slic # Используем заранее вычисленную сегментацию
)

print("Объяснение LIME готово.")

# --- Визуализация LIME ---
# Показываем объяснение для токена, который мы выбрали для анализа
temp, mask = explanation.get_image_and_mask(
    label=target_token_id, # ID токена, который объясняем
    positive_only=False, # Показать и позитивные, и негативные влияния
    num_features=10, # Количество суперпикселей для подсветки
    hide_rest=False # Не скрывать остальное изображение
)

plt.figure(figsize=(10, 10))
plt.imshow(temp / 2 + 0.5) # Обратная нормализация [-1, 1] -> [0, 1] если нужно, или просто temp
plt.title(f"LIME Explanation for char '{target_char}' (ID: {target_token_id}) at step {time_step_to_explain}")
plt.axis('off')
plt.show()

# Можно показать только "позитивные" регионы
temp_pos, mask_pos = explanation.get_image_and_mask(
    label=target_token_id,
    positive_only=True,
    num_features=10,
    hide_rest=True # Скрыть остальное
)
plt.figure(figsize=(10, 10))
plt.imshow(temp_pos / 2 + 0.5)
plt.title(f"LIME Positive regions for char '{target_char}'")
plt.axis('off')
plt.show()