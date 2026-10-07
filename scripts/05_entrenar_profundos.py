"""
Entrena y evalúa una red profunda (CNN 1D, Transformer o LSTM) con el protocolo del proyecto.

Mismo protocolo que los modelos clásicos (scripts/03_entrenar_clasicos.py)
--------------------------------------------------------------------------
  * Entrenamiento con septiembre de 2022; 20 % para validación (estratificado, semilla fija).
  * Pesos por clase "balanced", la misma fórmula del EDA y de los modelos clásicos.
  * Prueba con octubre; F1 sin secuencias repetidas; deriva en noviembre y diciembre.
  * Latencia medida SIEMPRE en CPU, para compararla en igualdad con Random Forest y XGBoost,
    aunque el entrenamiento se haga en GPU.
  * Mismos archivos de salida en results/modelos/<modelo>/, que scripts/04_comparar_modelos.py
    incorpora sin cambios.

Específico de las redes
-----------------------
  * Entrada: secuencia de 30 paquetes (30 × 3) con máscara de relleno.
  * Parada temprana: se guarda la época con mejor F1 macro de validación y se detiene tras
    `paciencia` épocas sin mejorar (evita el sobreajuste y ahorra tiempo de GPU).

Uso:
    python scripts/05_entrenar_profundos.py --modelo cnn1d
    python scripts/05_entrenar_profundos.py --modelo transformer
    python scripts/05_entrenar_profundos.py --modelo cnn1d --epocas 1 --limite-filas 50000   # prueba rápida
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar src/ al ejecutar el script

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.data.loader import labels, load_months, load_split, sequence_features  # noqa: E402
from src.evaluation.metrics import count_parameters, f1_without_repeats, latency_ms_per_flow, performance  # noqa: E402
from src.evaluation.report import plot_confusion  # noqa: E402
from src.models.deep import build_deep_model  # noqa: E402


def set_seed(seed: int) -> None:
    """Fija la semilla de todas las fuentes de azar (NumPy y PyTorch en CPU y GPU)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_loader(X, mask, y=None, batch=1024, shuffle=False, seed=42) -> DataLoader:
    """Empaqueta arreglos de NumPy en un DataLoader de PyTorch.

    El generador con semilla hace que el orden aleatorio de los lotes sea reproducible.
    """
    tensors = [torch.from_numpy(X), torch.from_numpy(mask)]
    if y is not None:
        tensors.append(torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch, shuffle=shuffle, generator=g)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, mask: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    """Clase predicha para cada flujo. no_grad desactiva el cálculo de gradientes (más rápido, menos memoria)."""
    model.eval()
    out = []
    for xb, mb in to_loader(X, mask, batch=batch):
        out.append(model(xb.to(device), mb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def memoria_pico_gb() -> float:
    """Memoria RAM máxima usada por el proceso hasta ahora, en GB (para dimensionar el equipo necesario)."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", choices=["cnn1d", "transformer", "lstm"], default="cnn1d")
    ap.add_argument(
        "--epocas", type=int, default=0, help="sobrescribe las épocas de la configuración (0 = usar config)"
    )
    ap.add_argument("--limite-filas", type=int, default=0, help="prueba rápida con N flujos de entrenamiento")
    ap.add_argument(
        "--dispositivo",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto usa la GPU si está disponible (Colab/Kaggle) y la CPU si no",
    )
    a = ap.parse_args()

    cfg = load_config()
    seed, umbrales = cfg["semilla"], cfg["umbrales"]
    params = cfg["modelos"][a.modelo]
    tr_cfg = cfg["entrenamiento_profundo"]
    epocas = a.epocas or params["epocas"]
    if a.dispositivo == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = a.dispositivo
    set_seed(seed)
    out = ROOT / "results" / "modelos" / a.modelo
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 1. datos
    t0 = time.time()
    train_df = load_split("entrenamiento")
    if a.limite_filas:
        train_df = train_df.sample(n=min(a.limite_filas, len(train_df)), random_state=seed)
        print(f"MODO PRUEBA: {len(train_df):,} flujos de entrenamiento", flush=True)
    y_all, classes = labels(train_df)
    X_all, M_all = sequence_features(train_df)
    train_hashes = train_df["SEQ_HASH"].to_numpy()
    del train_df
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_all)), test_size=cfg["datos"]["fraccion_validacion"], stratify=y_all, random_state=seed
    )
    X_tr, M_tr, y_tr = X_all[idx_tr], M_all[idx_tr], y_all[idx_tr]
    X_val, M_val, y_val = X_all[idx_val], M_all[idx_val], y_all[idx_val]
    del X_all, M_all
    print(
        f"Entrenamiento: {len(y_tr):,} · validación: {len(y_val):,} · {len(classes)} clases · "
        f"dispositivo: {device} (carga en {time.time() - t0:.0f} s)",
        flush=True,
    )

    # ------------------------------------------------------------------ 2. modelo, pérdida y optimizador
    model = build_deep_model(a.modelo, params, len(classes)).to(device)
    n_params = count_parameters(model)
    # Pesos "balanced": n / (k · n_clase), la misma fórmula que Random Forest y XGBoost
    w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    train_loader = to_loader(X_tr, M_tr, y_tr, batch=tr_cfg["lote"], shuffle=True, seed=seed)
    # Versión 2: programador OneCycle opcional (validado en el diagnóstico de la semana 3) y paciencia por modelo
    paciencia = params.get("paciencia", tr_cfg["paciencia"])
    sched = None
    if params.get("programador") == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=params["lr"] * 5, total_steps=epocas * len(train_loader)
        )
    print(f"Parámetros entrenables: {n_params:,}", flush=True)

    # ------------------------------------------------------------------ 3. entrenamiento con parada temprana
    best_f1, best_state, sin_mejora, historial = -1.0, None, 0, []
    t_train = time.time()
    for ep in range(1, epocas + 1):
        model.train()
        t_ep, perdida = time.time(), 0.0
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            perdida += loss.item() * len(yb)
        val_f1 = performance(y_val, predict(model, X_val, M_val, device))["f1_macro"]
        historial.append(
            {
                "epoca": ep,
                "perdida": perdida / len(y_tr),
                "f1_macro_validacion": val_f1,
                "tasa_aprendizaje": opt.param_groups[0]["lr"],
            }
        )
        marca = ""
        if val_f1 > best_f1:
            best_f1, best_state, sin_mejora, marca = val_f1, copy.deepcopy(model.state_dict()), 0, "  ← mejor"
        else:
            sin_mejora += 1
        print(
            f"Época {ep:2d}/{epocas} · pérdida {perdida / len(y_tr):.4f} · F1 macro validación {val_f1:.4f} · "
            f"{time.time() - t_ep:.0f} s{marca}",
            flush=True,
        )
        if sin_mejora >= paciencia:
            print(f"Parada temprana: {paciencia} épocas sin mejorar", flush=True)
            break
    train_s = time.time() - t_train
    model.load_state_dict(best_state)  # se evalúa la mejor época, no la última
    val_perf = performance(y_val, predict(model, X_val, M_val, device))

    # ------------------------------------------------------------------ 4. prueba (octubre)
    test_df = load_split("prueba")
    y_te, _ = labels(test_df, classes)
    X_te, M_te = sequence_features(test_df)
    pred = predict(model, X_te, M_te, device)
    test_perf = performance(y_te, pred)
    sin_rep = f1_without_repeats(y_te, pred, test_df["SEQ_HASH"].to_numpy(), train_hashes)
    del test_df
    np.savez_compressed(out / "predicciones_prueba.npz", y_true=y_te, y_pred=pred)
    rep = pd.DataFrame(
        classification_report(
            y_te, pred, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0
        )
    ).T
    rep.to_csv(out / "reporte_por_clase.csv")
    cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "matriz_confusion.csv")
    plot_confusion(
        cm, classes, out / "matriz_confusion.png", f"{a.modelo} · octubre 2022 · F1 macro {test_perf['f1_macro']:.3f}"
    )

    # ------------------------------------------------------------------ 5. deriva
    deriva = {}
    for mes in cfg["datos"]["meses"]["deriva"]:
        d = load_months([mes])
        y_d, _ = labels(d, classes)
        X_d, M_d = sequence_features(d)
        deriva[mes] = performance(y_d, predict(model, X_d, M_d, device))["f1_macro"]
        del d, X_d, M_d

    # ------------------------------------------------------------------ 6. costo (latencia en CPU, como los clásicos)
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    with torch.no_grad():
        latency = latency_ms_per_flow(
            lambda i: model_cpu(torch.from_numpy(X_te[i]), torch.from_numpy(M_te[i])).argmax(1), np.arange(len(y_te))
        )
    model_path = ROOT / "models" / f"{a.modelo}.pt"
    torch.save(
        {"estado": model.to("cpu").state_dict(), "modelo": a.modelo, "parametros": params, "clases": classes},
        model_path,
    )
    size_mb = model_path.stat().st_size / 1e6

    # ------------------------------------------------------------------ 7. resultados (mismo formato que los clásicos)
    res = {
        "modelo": a.modelo,
        "modo_prueba": bool(a.limite_filas or a.epocas),
        "parametros": params,
        "entrenamiento": {
            "flujos": int(len(y_tr)),
            "segundos": round(train_s, 1),
            "dispositivo": device,
            "epocas_ejecutadas": len(historial),
            "historial": historial,
        },
        "validacion": val_perf,
        "prueba_octubre": test_perf,
        "prueba_sin_repetidos": sin_rep,
        "deriva_f1_macro": deriva,
        "caida_f1_octubre_a_diciembre_puntos": round(100 * (test_perf["f1_macro"] - deriva.get("2022-12", np.nan)), 2),
        "costo": {
            "memoria_pico_GB": memoria_pico_gb(),
            "latencia_ms_por_flujo": round(latency, 5),
            "tamano_modelo_MB": round(size_mb, 2),
            "parametros_entrenables": n_params,
        },
        "umbrales": {
            "f1_macro_minimo_cumplido": test_perf["f1_macro"] >= umbrales["f1_macro_minimo"],
            "f1_macro_excelencia_cumplido": test_perf["f1_macro"] >= umbrales["f1_macro_excelencia"],
            "latencia_cumplida": latency <= umbrales["latencia_ms_maxima"],
            "transformer_menos_de_2M_parametros": n_params <= 2_000_000 if a.modelo == "transformer" else None,
        },
    }
    (out / "metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("\n================ RESULTADOS ================")
    print(
        f"F1 macro octubre ............ {test_perf['f1_macro']:.4f}  (mínimo {umbrales['f1_macro_minimo']}, "
        f"excelencia {umbrales['f1_macro_excelencia']})"
    )
    print(
        f"F1 macro sin repetidos ...... {sin_rep['f1_macro_sin_repetidos']:.4f}  "
        f"({sin_rep['fraccion_excluida']:.1%} de la prueba excluida)"
    )
    print(f"F1 ponderado / exactitud .... {test_perf['f1_weighted']:.4f} / {test_perf['accuracy']:.4f}")
    print("Deriva (F1 macro) ........... " + " · ".join(f"{m}: {v:.4f}" for m, v in deriva.items()))
    print(f"Latencia por flujo (CPU) .... {latency:.4f} ms (máximo {umbrales['latencia_ms_maxima']} ms)")
    print(f"Parámetros / tamaño ......... {n_params:,} / {size_mb:.2f} MB")
    print(f"Entrenamiento ............... {train_s / 60:.1f} min en {device}, {len(historial)} épocas")
    print(f"Umbrales .................... {res['umbrales']}")
    print(f"Resultados en ............... {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
