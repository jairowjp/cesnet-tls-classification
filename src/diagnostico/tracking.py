"""
Tracking de métricas durante el entrenamiento (componente 1 de la actividad, 25 %).

Tres piezas, una por tipo de modelo del proyecto:

  * RegistroMetricas: guarda cada paso (época o ronda) en memoria, en CSV/JSON y en MLflow.
  * CallbackRegistroXGB: callback personalizado de XGBoost (equivalente a los callbacks de Keras)
    que registra la pérdida y el F1 macro de entrenamiento y validación durante el boosting.
  * entrenar_red: bucle de entrenamiento de PyTorch con registro manual por época, programador
    de tasa de aprendizaje opcional (OneCycle) y parada temprana opcional.

Para scikit-learn (Random Forest) se usan learning_curve y validation_curve directamente en
scripts/09_diagnostico.py, como pide la actividad.

MLflow se usa en modo local con almacenamiento SQLite (results/diagnostico/mlflow.db): ningún dato
sale del equipo. Para ver los experimentos:  mlflow ui --backend-store-uri sqlite:///results/diagnostico/mlflow.db
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _carpeta_diagnostico(carpeta: Path) -> Path:
    """Carpeta raíz del diagnóstico (results/diagnostico): ahí va la ÚNICA base de MLflow de todas las corridas.

    Se busca hacia arriba la carpeta "diagnostico"; si no existe, se usa la carpeta de la corrida.
    """
    for padre in [carpeta, *carpeta.parents]:
        if padre.name == "diagnostico":
            return padre
    return carpeta


class RegistroMetricas:
    """Registra métricas por paso y las envía a MLflow (si está disponible).

    Uso:
        with RegistroMetricas("diagnostico", "xgboost-base", carpeta, parametros) as reg:
            reg.registrar(1, perdida_entrenamiento=0.9, perdida_validacion=1.0)
    Al salir del bloque se guardan historial.csv e historial.json y se cierra la corrida de MLflow.
    """

    def __init__(
        self,
        experimento: str,
        corrida: str,
        carpeta: Path,
        parametros: dict | None = None,
        usar_mlflow: bool = True,
        db: Path | None = None,
    ):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.corrida, self.parametros, self.filas = corrida, parametros or {}, []
        self.mlflow = None
        if usar_mlflow:
            try:
                import mlflow
            except ImportError as e:
                raise ImportError("Falta MLflow. Instálalo con: uv pip install mlflow") from e
            db = Path(db or _carpeta_diagnostico(self.carpeta) / "mlflow.db").resolve()
            mlflow.set_tracking_uri(f"sqlite:///{db}")
            # Los artefactos van junto a la base (results/diagnostico/mlruns), no a la carpeta desde donde se ejecuta
            if mlflow.get_experiment_by_name(experimento) is None:
                mlflow.create_experiment(experimento, artifact_location=(db.parent / "mlruns").resolve().as_uri())
            mlflow.set_experiment(experimento)
            mlflow.start_run(run_name=corrida)
            # MLflow acepta parámetros como texto; las listas y None se convierten para que no falle
            mlflow.log_params({k: str(v) for k, v in self.parametros.items()})
            self.mlflow = mlflow

    def registrar(self, paso: int, **metricas) -> None:
        """Agrega una fila (paso + métricas). Los valores None se guardan pero no se envían a MLflow."""
        self.filas.append({"paso": int(paso), **metricas})
        if self.mlflow:
            numericas = {k: float(v) for k, v in metricas.items() if isinstance(v, (int, float, np.floating))}
            self.mlflow.log_metrics(numericas, step=int(paso))

    def historial(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)

    def cerrar(self, resumen: dict | None = None, estado: str = "FINISHED") -> None:
        """Guarda el historial y el resumen; los registra como artefactos y cierra la corrida."""
        hist = self.historial()
        hist.to_csv(self.carpeta / "historial.csv", index=False)
        (self.carpeta / "historial.json").write_text(json.dumps(self.filas, indent=1, ensure_ascii=False))
        if resumen is not None:
            (self.carpeta / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False, default=str))
        if self.mlflow:
            if resumen:
                numericas = {
                    f"final_{k}": float(v)
                    for k, v in resumen.items()
                    if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
                }
                self.mlflow.log_metrics(numericas)
            self.mlflow.log_artifact(str(self.carpeta / "historial.csv"))
            self.mlflow.end_run(status=estado)
            self.mlflow = None

    def __enter__(self):
        return self

    def __exit__(self, tipo, valor, traza):
        if self.mlflow:  # si hubo un error, la corrida queda marcada como fallida en MLflow
            self.cerrar(estado="FAILED" if tipo else "FINISHED")
        return False


def _callback_base():
    """Clase base de callbacks de XGBoost; se importa aquí para no exigir XGBoost a quien no lo use."""
    from xgboost.callback import TrainingCallback

    return TrainingCallback


class CallbackRegistroXGB(_callback_base()):
    """Callback personalizado de XGBoost: registra métricas en cada ronda de boosting.

    La pérdida (mlogloss) de entrenamiento y validación la calcula XGBoost en cada ronda a partir
    de eval_set = [(X_entrenamiento, y), (X_validacion, y)]. El F1 macro es más costoso (exige
    predecir), así que se calcula cada `cada` rondas.
    """

    def __init__(self, registro: RegistroMetricas, X_tr, y_tr, X_val, y_val, cada: int = 10):
        super().__init__()
        self.registro, self.cada = registro, cada
        self.datos = [(X_tr, np.asarray(y_tr)), (X_val, np.asarray(y_val))]

    def after_iteration(self, model, epoch, evals_log) -> bool:
        from xgboost import DMatrix

        metricas = {
            "perdida_entrenamiento": evals_log["validation_0"]["mlogloss"][-1],
            "perdida_validacion": evals_log["validation_1"]["mlogloss"][-1],
            "f1_entrenamiento": None,
            "f1_validacion": None,
        }
        if epoch % self.cada == 0:
            for nombre, (X, y) in zip(("f1_entrenamiento", "f1_validacion"), self.datos, strict=True):
                prob = model.predict(DMatrix(X))
                metricas[nombre] = float(f1_score(y, prob.argmax(axis=1), average="macro"))
        self.registro.registrar(epoch + 1, **metricas)
        return False  # False = continuar entrenando (la parada temprana la decide XGBoost)


def entrenar_red(
    modelo,
    X_tr,
    M_tr,
    y_tr,
    X_val,
    M_val,
    y_val,
    registro: RegistroMetricas,
    *,
    epocas: int,
    lr: float,
    pesos_clase: np.ndarray,
    lote: int = 1024,
    programador: str = "constante",
    paciencia: int | None = None,
    muestra_f1: int = 50_000,
    semilla: int = 42,
    dispositivo: str = "cpu",
):
    """Entrena una red de PyTorch registrando, en cada época, pérdida y F1 de entrenamiento y validación.

    Las cuatro métricas se miden en modo evaluación (sin dropout) y con la misma función de pérdida
    ponderada que se optimiza, para que las curvas de entrenamiento y validación sean comparables.
    El F1 de entrenamiento se mide sobre una submuestra fija (`muestra_f1`) para no duplicar el costo.

    programador: "constante" (tasa fija) u "onecycle" (sube y luego baja la tasa: ayuda a converger).
    paciencia: épocas sin mejorar el F1 de validación antes de detener (None = sin parada temprana).
    Devuelve el modelo con los pesos de la MEJOR época y el número de épocas ejecutadas.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    modelo = modelo.to(dispositivo)
    perdida_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos_clase, dtype=torch.float32, device=dispositivo))
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr)
    g = torch.Generator().manual_seed(semilla)
    cargador = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(M_tr), torch.from_numpy(y_tr)),
        batch_size=lote,
        shuffle=True,
        generator=g,
    )
    sched = None
    if programador == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 5, total_steps=epocas * len(cargador))
    idx_f1 = rng.choice(len(y_tr), size=min(muestra_f1, len(y_tr)), replace=False)

    @torch.no_grad()
    def evaluar(X, M, y):
        modelo.eval()
        perdidas, preds = [], []
        for i in range(0, len(y), 8192):
            xb = torch.from_numpy(X[i : i + 8192]).to(dispositivo)
            mb = torch.from_numpy(M[i : i + 8192]).to(dispositivo)
            yb = torch.from_numpy(y[i : i + 8192]).to(dispositivo)
            salida = modelo(xb, mb)
            perdidas.append(perdida_fn(salida, yb).item() * len(yb))
            preds.append(salida.argmax(1).cpu().numpy())
        return sum(perdidas) / len(y), float(f1_score(y, np.concatenate(preds), average="macro"))

    mejor_f1, mejor_estado, sin_mejora, ep = -1.0, None, 0, 0
    for ep in range(1, epocas + 1):
        modelo.train()
        t0 = time.time()
        for xb, mb, yb in cargador:
            xb, mb, yb = xb.to(dispositivo), mb.to(dispositivo), yb.to(dispositivo)
            opt.zero_grad(set_to_none=True)
            perdida_fn(modelo(xb, mb), yb).backward()
            opt.step()
            if sched:
                sched.step()
        p_tr, f1_tr = evaluar(X_tr[idx_f1], M_tr[idx_f1], y_tr[idx_f1])
        p_val, f1_val = evaluar(X_val, M_val, y_val)
        registro.registrar(
            ep,
            perdida_entrenamiento=p_tr,
            perdida_validacion=p_val,
            f1_entrenamiento=f1_tr,
            f1_validacion=f1_val,
            tasa_aprendizaje=opt.param_groups[0]["lr"],
            segundos=round(time.time() - t0, 1),
        )
        marca = ""
        if f1_val > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora, marca = f1_val, copy.deepcopy(modelo.state_dict()), 0, "  ← mejor"
        else:
            sin_mejora += 1
        print(
            f"Época {ep:2d}/{epocas} · pérdida ent. {p_tr:.4f} · pérdida val. {p_val:.4f} · "
            f"F1 ent. {f1_tr:.4f} · F1 val. {f1_val:.4f}{marca}",
            flush=True,
        )
        if paciencia and sin_mejora >= paciencia:
            print(f"Parada temprana: {paciencia} épocas sin mejorar el F1 de validación", flush=True)
            break
    modelo.load_state_dict(mejor_estado)
    return modelo, ep
