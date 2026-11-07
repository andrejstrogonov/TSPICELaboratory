import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

# Структура описания моделей, строго соответствующая main.py
MODEL_SPECS = {
    "poisson": {
        "label": "Электростатика · уравнение Пуассона",
        "in_channels": 3,
        "out_channels": 1,
        "output_names": ("potential_V",),
    },
    "magnetostatic": {
        "label": "Магнитостатика · потенциал Az",
        "in_channels": 3,
        "out_channels": 1,
        "output_names": ("vector_potential_Az_normalized",),
    },
    "carriers": {
        "label": "Носители · равновесные n/p",
        "in_channels": 4,
        "out_channels": 2,
        "output_names": ("log10_electron_density_norm", "log10_hole_density_norm"),
    },
}


class BaseUNet(nn.Module):
    """Параметризуемый каркас U-Net для узких физических задач."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.enc1 = self._block(in_channels, 16)
        self.enc2 = self._block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._block(32, 16)
        self.final_conv = nn.Conv2d(16, out_channels, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Проверка кратности 4 для корректной работы пулинга и ConvTranspose2d
        if inputs.shape[-2] % 4 or inputs.shape[-1] % 4:
            h, w = inputs.shape[-2], inputs.shape[-1]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            inputs = nn.functional.pad(inputs, (0, pad_w, 0, pad_h), mode='replicate')

        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.dec2(torch.cat((self.up2(bottleneck), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))
        out = self.final_conv(dec1)

        # Возвращаем исходный размер, если было дополнение (padding)
        return out[..., :out.shape[-2] - pad_h if 'pad_h' in locals() else None, :out.shape[
                                                                                      -1] - pad_w if 'pad_w' in locals() else None]


# =====================================================================
# ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ В СИМУЛЯТОР (Связующий мост)
# =====================================================================
def predict_field_warm_start(model_name: str, checkpoint_path: str, mesh_axes: tuple, device_params, gate_v: float,
                             drain_v: float) -> np.ndarray:
    """
    Трансформирует геометрию и сетку из C++/UI-слоя в тензор,
    делает ИИ-предикт и разворачивает его обратно под физический шаг сетки.
    """
    x_m, y_m, z_m = mesh_axes
    nx, ny, nz = len(x_m), len(y_m), len(z_m)

    # 1. Загрузка специализированной модели
    spec = MODEL_SPECS[model_name]
    model = BaseUNet(spec["in_channels"], spec["out_channels"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Формируем входной 2D-тензор-срез по осям X-Z (так как физика транзистора планарна вглубь)
    input_tensor = np.zeros((1, spec["in_channels"], nx, nz), dtype=np.float32)

    if model_name == "poisson":
        # Канал 0: Проницаемость (нормализованная по Кремнию)
        input_tensor[0, 0, :, :] = device_params.epsilon_r / 11.7
        # Канал 1: Граничные условия потенциала
        input_tensor[0, 1, :, 0] = gate_v  # На затворе сверху
        input_tensor[0, 1, -1, :] = drain_v  # На стоке справа
        # Канал 2: Маска контактов
        input_tensor[0, 2, :, 0] = 1.0
        input_tensor[0, 2, -1, :] = 1.0

    elif model_name == "carriers":
        # Логарифмическое сжатие легирования (чтобы сеть не сходила с ума от масштабов 10^20)
        log_doping = math.log10(device_params.doping) / 20.0
        input_tensor[0, 0, :, :] = log_doping  # Nd profile
        input_tensor[0, 1, :, :] = 0.0  # Na profile
        input_tensor[0, 2, :, :] = gate_v / 1.2  # Электростатический потенциал (нормализация)
        input_tensor[0, 3, :, :] = 1.0  # Полупроводниковая зона

    # 3. Инференс
    with torch.no_grad():
        torch_in = torch.from_numpy(input_tensor)
        torch_out = model(torch_in).numpy()[0]  # Выход: [out_channels, nx, nz]

    # 4. Обратная денормализация и экструзия по третьей оси (Y - ширина транзистора)
    # Поля потенциалов однородны по ширине Y, а концентрации денормализуются из логарифма
    final_fields = np.zeros((spec["out_channels"], nx, ny, nz), dtype=np.float64)

    for c in range(spec["out_channels"]):
        grid_2d = torch_out[c]  # [nx, nz]
        if model_name == "carriers":
            # Восстанавливаем реальные значения концентрации носителей: 10^(cut * 8)
            grid_2d = 10 ** (grid_2d * 8.0)
            grid_2d = np.clip(grid_2d, 1e8, 1e21)

        # Растягиваем 2D срез по всей ширине транзистора (ось Y)
        for j in range(ny):
            final_fields[c, :, j, :] = grid_2d

    return final_fields
