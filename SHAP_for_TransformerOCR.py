import torch
import numpy as np
import shap
import lime
from lime import lime_image
from skimage.segmentation import slic # Для сегментации LIME
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from TransformerOCR import ImageToRusmarcModel, generate_square_subsequent_mask, DataLoader, test_dataset
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
# --- Prediction Function для SHAP ---
# KernelSHAP ожидает функцию, которая может обрабатывать "замаскированные" данные
# и часто работает с плоскими массивами признаков.
# Мы будем маскировать пиксели, заменяя их средним цветом фона.

# 1. Подготовка фонового датасета (небольшой набор изображений)
#    Нужен для ожидания E[f(x)]
background_data = []
# Возьмем несколько изображений из тестового загрузчика (или обучающего)
num_background_samples = 50 # Обычно 50-200 достаточно
temp_loader = DataLoader(test_dataset, batch_size=num_background_samples, shuffle=True)
background_images, _ = next(iter(temp_loader)) # [N, C, H, W], тензор PyTorch
background_images_np = background_images.numpy().transpose(0, 2, 3, 1) # -> [N, H, W, C], NumPy

# 2. Обертка для SHAP
def predict_proba_shap(numpy_images_masked):
    """
    Принимает массив NumPy с маскированными изображениями [N, H*W*C].
    Возвращает массив вероятностей ТОЛЬКО ДЛЯ ЦЕЛЕВОГО ТОКЕНА [N].
    """
    num_images = numpy_images_masked.shape[0]
    # Восстанавливаем форму [N, H, W, C]
    height, width, channels = pil_image.shape # Берем размеры из оригинала
    reshaped_images = numpy_images_masked.reshape(num_images, height, width, channels)

    # Нормализация (как в LIME)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor_images = torch.tensor(reshaped_images, dtype=torch.float32).permute(0, 3, 1, 2)
    tensor_images = tensor_images.to(device)
    tensor_images = (tensor_images - mean) / std

    # Получаем ВСЕ вероятности
    all_probabilities = get_probabilities_at_step_t(tensor_images, time_step_to_explain) # [N, num_tokens]

    # Возвращаем только вероятность целевого токена
    target_token_prob = all_probabilities[:, target_token_id] # [N]
    return target_token_prob

# --- Запуск SHAP KernelExplainer ---
# SHAP ожидает фон в формате [N_bg, M_features]
background_flat = background_images_np.reshape(num_background_samples, -1)

# Создаем explainer
# Функция predict_proba_shap принимает [N, M], фон [N_bg, M]
explainer_shap = shap.KernelExplainer(predict_proba_shap, background_flat)

# Изображение для объяснения в плоском виде [1, M]
image_to_explain_flat = pil_image.reshape(1, -1)

print("Запуск расчета SHAP values (может быть долго)...")
# nsamples - количество сгенерированных возмущений для КАЖДОГО фонового примера
# l1_reg='aic' - автоматический подбор регуляризации
shap_values = explainer_shap.shap_values(image_to_explain_flat, nsamples=500, l1_reg="aic")
print("Расчет SHAP values завершен.")


# --- Визуализация SHAP ---
# shap_values - это массив [1, M] или список массивов, если было несколько выходов
# Нам нужно преобразовать его обратно в форму изображения [1, H, W, C]
shap_values_img = shap_values.reshape(1, pil_image.shape[0], pil_image.shape[1], pil_image.shape[2])

# Исходное изображение для plot [1, H, W, C]
image_to_explain_plot = pil_image.reshape(1, pil_image.shape[0], pil_image.shape[1], pil_image.shape[2])

shap.image_plot(shap_values_img, -image_to_explain_plot) # -image для инвертирования цветов фона при плоте

plt.suptitle(f"SHAP Explanation for char '{target_char}' (ID: {target_token_id}) at step {time_step_to_explain}")
plt.show()