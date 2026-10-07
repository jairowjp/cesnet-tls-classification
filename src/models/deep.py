"""
Modelos profundos de la rama B: reciben la secuencia de los primeros 30 paquetes de cada flujo.

Entrada común (ver src/data/loader.py → sequence_features)
-----------------------------------------------------------
  x    : tensor (lote, 30, 3) con [tamaño/1500, dirección (+1, −1, 0), log10(1 + tiempo entre paquetes)]
  mask : tensor booleano (lote, 30), True en las posiciones de relleno (sin paquete real)

El EDA mostró que el 51,6 % de las posiciones son relleno (mediana de 11 paquetes por flujo).
Por eso todos los modelos promedian su salida SOLO sobre las posiciones reales, y el
Transformer además usa la máscara para que la atención ignore el relleno.

Los tres modelos tienen la misma firma forward(x, mask) → logits (lote, n_clases), así el
script de entrenamiento los trata a todos igual.
"""
import torch
from torch import nn


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Promedio sobre la dimensión de la secuencia ignorando el relleno.

    h: (lote, 30, d) · mask: (lote, 30) con True en el relleno → devuelve (lote, d).
    Todo flujo del dataset tiene al menos 3 paquetes, así que el denominador nunca es cero;
    clamp(min=1) es solo una protección adicional.
    """
    keep = (~mask).unsqueeze(-1).to(h.dtype)
    return (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


class TrafficCNN1D(nn.Module):
    """Red convolucional 1D: filtros que recorren la secuencia y detectan patrones locales
    entre paquetes vecinos (por ejemplo, el intercambio típico de un saludo TLS).

    Dos bloques convolución → normalización → ReLU amplían el campo de visión a 5 paquetes;
    luego se promedia sobre las posiciones reales y una capa lineal produce las 23 salidas.
    """

    def __init__(self, n_classes: int, in_feats: int = 3, canales=(64, 128), kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        c1, c2 = canales
        pad = kernel // 2  # mantiene la longitud de 30 posiciones
        self.features = nn.Sequential(
            nn.Conv1d(in_feats, c1, kernel, padding=pad), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel, padding=pad), nn.BatchNorm1d(c2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c2, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # En el borde entre paquetes reales y relleno, el filtro también "ve" posiciones vacías.
        # Se fuerzan a cero para que la salida nunca dependa del contenido del relleno.
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        h = self.features(x.transpose(1, 2)).transpose(1, 2)   # Conv1d espera (lote, canales, posiciones)
        return self.head(masked_mean(h, mask))


class TrafficTransformer(nn.Module):
    """Transformer encoder ligero, sin preentrenamiento.

    1. Cada paquete (3 valores) se proyecta a un vector de d_model dimensiones.
    2. Se suma una codificación de posición aprendida (el orden de los paquetes importa).
    3. Las capas de atención relacionan cualquier par de paquetes, estén cerca o lejos.
       src_key_padding_mask impide que la atención mire posiciones de relleno.
    4. Se promedian las posiciones reales y una capa lineal produce las 23 salidas.
    """

    def __init__(self, n_classes: int, in_feats: int = 3, seq_len: int = 30, d_model: int = 64,
                 cabezas: int = 4, capas: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_feats, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)                 # inicialización pequeña, práctica estándar
        layer = nn.TransformerEncoderLayer(d_model, cabezas, ff, dropout, batch_first=True, norm_first=True)
        # enable_nested_tensor=False: evita una ruta interna de PyTorch que emite avisos con norm_first
        self.encoder = nn.TransformerEncoder(layer, capas, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x) + self.pos, src_key_padding_mask=mask)
        return self.head(masked_mean(h, mask))


class TrafficLSTM(nn.Module):
    """LSTM (opcional): lee la secuencia paquete a paquete conservando una memoria del contexto.

    Se incluye como referencia del Análisis comparativo; no forma parte del compromiso del proyecto.
    """

    def __init__(self, n_classes: int, in_feats: int = 3, unidades: int = 128, capas: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, unidades, num_layers=capas, batch_first=True,
                            dropout=dropout if capas > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(unidades, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(masked_mean(h, mask))


# Claves de configs/experiment.yaml que controlan el ENTRENAMIENTO, no la arquitectura de la red
CLAVES_ENTRENAMIENTO = ("epocas", "lr", "programador", "paciencia")


def build_deep_model(name: str, params: dict, n_classes: int) -> nn.Module:
    """Construye el modelo a partir de su sección en configs/experiment.yaml.

    Las claves de entrenamiento (CLAVES_ENTRENAMIENTO) se separan de las de arquitectura.
    """
    arch = {k: v for k, v in params.items() if k not in CLAVES_ENTRENAMIENTO}
    if name == "cnn1d":
        return TrafficCNN1D(n_classes, canales=tuple(arch["canales"]), kernel=arch["kernel"], dropout=arch["dropout"])
    if name == "transformer":
        return TrafficTransformer(n_classes, **arch)
    if name == "lstm":
        return TrafficLSTM(n_classes, **arch)
    raise ValueError(f"Modelo profundo desconocido: {name!r}")
