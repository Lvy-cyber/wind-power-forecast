"""模型与评估口径。

四个模型，从最"笨"到最强：
  1. 持续法    P(t+h) = P(t)
  2. 日变化气候态  用训练期同一时刻（小时）的平均功率
  3. 岭回归       滞后风速/功率 + 风向 + 时刻的正弦余弦
  4. 梯度提升树    同一套特征，非线性

评估口径（行业惯用）：
  RMSE    = sqrt(mean((P_pred - P_obs)^2))，单位 MW
  准确率  = (1 - RMSE / 装机容量) × 100%
  合格率  = |误差| <= 25% 装机容量的样本占比
  Skill   = 1 - RMSE_model / RMSE_persistence（相对持续法的误差下降）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg


# ---------------------------------------------------------------- 指标

def score(y: np.ndarray, yhat: np.ndarray, cap: float = cfg.CAP_MW) -> dict:
    err = np.asarray(yhat) - np.asarray(y)
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "n": int(len(y)),
        "RMSE_MW": round(rmse, 3),
        "RMSE_pct": round(rmse / cap * 100, 2),
        "MAE_MW": round(float(np.mean(np.abs(err))), 3),
        "准确率_pct": round((1.0 - rmse / cap) * 100, 2),
        "合格率_pct": round(float(np.mean(np.abs(err) <= cfg.PASS_RATIO * cap) * 100), 2),
        "偏差_MW": round(float(np.mean(err)), 3),
    }


# ---------------------------------------------------------------- 模型

def fit_predict(model: str, Xtr, ytr, Xte, extra: dict):
    if model == "persistence":
        return np.clip(extra["p_now"], 0.0, cfg.CAP_MW)
    if model == "climatology":
        return np.clip(extra["clim"], 0.0, cfg.CAP_MW)
    if model == "ridge":
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(Xtr, ytr)
        return np.clip(pipe.predict(Xte), 0.0, cfg.CAP_MW)
    if model == "gbdt":
        gbr = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        gbr.fit(Xtr, ytr)
        return np.clip(gbr.predict(Xte), 0.0, cfg.CAP_MW)
    raise ValueError(model)


def climatology_table(ytr: np.ndarray, target_time_tr: pd.DatetimeIndex) -> pd.Series:
    """训练期按目标时刻的小时取平均功率；这就是最朴素的"日内气候态"。"""
    s = pd.Series(np.asarray(ytr, dtype=float), index=target_time_tr)
    return s.groupby(s.index.hour).mean()


MODELS = ("persistence", "climatology", "ridge", "gbdt")
