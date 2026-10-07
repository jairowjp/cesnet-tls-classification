"""Pruebas de los modelos profundos: forma de la salida y respeto de la máscara de relleno."""
import pytest
import torch

from src.config import load_config
from src.models.deep import build_deep_model, masked_mean

N_CLASES = 23


def _lote(n=4, reales=5):
    """n flujos de 30 posiciones con `reales` paquetes y el resto de relleno."""
    torch.manual_seed(0)
    x = torch.randn(n, 30, 3)
    mask = torch.zeros(n, 30, dtype=torch.bool)
    mask[:, reales:] = True
    x[mask] = 0.0                      # el relleno llega en ceros, como en sequence_features
    return x, mask


@pytest.mark.parametrize("nombre", ["cnn1d", "transformer", "lstm"])
def test_salida_tiene_una_puntuacion_por_clase(nombre):
    model = build_deep_model(nombre, load_config()["modelos"][nombre], N_CLASES).eval()
    x, mask = _lote()
    assert model(x, mask).shape == (4, N_CLASES)


@pytest.mark.parametrize("nombre", ["cnn1d", "transformer", "lstm"])
def test_modelo_ignora_el_contenido_del_relleno(nombre):
    """Si la máscara funciona, alterar las posiciones de relleno no debe cambiar la salida."""
    model = build_deep_model(nombre, load_config()["modelos"][nombre], N_CLASES).eval()
    x, mask = _lote()
    ruido = x.clone()
    ruido[mask] = torch.randn(int(mask.sum()), 3)   # basura (3 valores por posición) solo en el relleno
    with torch.no_grad():
        assert torch.allclose(model(x, mask), model(ruido, mask), atol=1e-5)


def test_transformer_respeta_limite_de_parametros():
    model = build_deep_model("transformer", load_config()["modelos"]["transformer"], N_CLASES)
    assert sum(p.numel() for p in model.parameters()) <= 2_000_000


def test_promedio_con_mascara_ignora_relleno():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])      # la tercera posición es relleno
    mask = torch.tensor([[False, False, True]])
    assert masked_mean(h, mask).item() == pytest.approx(2.0)


def test_claves_de_entrenamiento_no_llegan_a_la_arquitectura():
    """La configuración v2 agrega programador y paciencia: el constructor de la red no debe recibirlas."""
    from src.config import load_config
    from src.models.deep import build_deep_model

    for nombre in ("cnn1d", "transformer"):
        assert build_deep_model(nombre, load_config()["modelos"][nombre], 23) is not None
