"""数据层：读 ERA5 → 轮毂高度风速 → 功率序列 → 特征矩阵。

关键说明：本项目没有 SCADA 实测功率，功率序列由 ERA5 风速经功率曲线换算得到。
这是"模拟功率"，平滑程度高于真实机组出力，因此本项目的误差水平偏乐观。
报告与 README 都会把这一点写在最显眼的位置。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------- 读数据

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(cfg.DATA_RAW, parse_dates=["time"]).set_index("time").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_meta() -> dict:
    if os.path.exists(cfg.DATA_META):
        with open(cfg.DATA_META, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------- 功率换算

def power_curve(v: np.ndarray) -> np.ndarray:
    xs = np.array([p[0] for p in cfg.TURBINE_CURVE], dtype=float)
    ys = np.array([p[1] for p in cfg.TURBINE_CURVE], dtype=float)
    v = np.asarray(v, dtype=float)
    p = np.interp(v, xs, ys, left=0.0, right=0.0)
    p[(v < xs[1]) | (v > 25.0)] = 0.0
    return p


def shear_alpha(ws10: pd.Series, ws100: pd.Series) -> float:
    """逐时风切变指数 α = ln(v100/v10)/ln(10)，取中位数；只用两层都 >=2 m/s 的时次。"""
    df = pd.DataFrame({"lo": ws10, "hi": ws100}).dropna()
    df = df[(df.lo >= 2.0) & (df.hi >= 2.0)]
    a = np.log(df.hi / df.lo) / np.log(10.0)
    return float(a.median())


def build_power(df: pd.DataFrame) -> pd.DataFrame:
    """把 100 m 风速外推到轮毂高度，做空气密度修正，再按功率曲线换成功率 (MW)。

    与项目一的口径保持一致：IEC 等功率法 v_eq = v·(ρ/ρ_ref)^(1/3)。
    """
    rho = df["pressure_hpa"] * 100.0 / (287.05 * (df["temp_c"] + 273.15))
    alpha = shear_alpha(df["ws10"], df["ws100"])

    ws_hub = df["ws100"] * (cfg.SITE["hub_height_m"] / cfg.SITE["ref_height_m"]) ** alpha
    v_eq = ws_hub * (rho / cfg.RHO_REF) ** (1.0 / 3.0)

    out = pd.DataFrame(
        {
            "ws_hub": ws_hub,
            "ws10": df["ws10"],
            "rho": rho,
            "power": pd.Series(power_curve(v_eq.values), index=df.index),
            "wd100": df["wd100"],
            "ws100": df["ws100"],
        }
    )
    out.attrs["alpha"] = alpha
    return out


def prepare() -> tuple[pd.DataFrame, dict]:
    raw = load_raw()
    pw = build_power(raw)
    os.makedirs(os.path.dirname(cfg.DATA_POWER), exist_ok=True)
    pw.to_csv(cfg.DATA_POWER, encoding="utf-8-sig")

    meta = {
        "alpha": pw.attrs["alpha"],
        "rows": int(len(pw)),
        "start": str(pw.index.min()),
        "end": str(pw.index.max()),
        "mean_ws_hub": float(pw.ws_hub.mean()),
        "mean_power_mw": float(pw.power.mean()),
        "capacity_factor": float(pw.power.mean() / cfg.CAP_MW),
        "full_load_hours": float(pw.power.mean() * len(pw) / cfg.CAP_MW),
        "calm_rate": float((pw.power < 0.01 * cfg.CAP_MW).mean()),
        "rated_rate": float((pw.power > 0.99 * cfg.CAP_MW).mean()),
    }
    return pw, meta


# ---------------------------------------------------------------- 特征

LAGS = (0, 1, 2, 3, 6, 12, 24)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """只用 t 时刻及之前的信息构造特征，避免未来信息泄漏。"""
    ws = df["ws_hub"]
    p = df["power"]
    idx = df.index

    f = pd.DataFrame(index=idx)
    for lag in LAGS:
        f[f"ws_lag{lag}"] = ws.shift(lag)
    for win in (3, 6, 24):
        f[f"ws_mean{win}"] = ws.shift(1).rolling(win).mean()
    f["ws_std6"] = ws.shift(1).rolling(6).std()

    f["p_now"] = p
    f["p_lag1"] = p.shift(1)
    f["p_mean6"] = p.shift(1).rolling(6).mean()

    f["ws10"] = df["ws10"]
    wd = np.deg2rad(df["wd100"])
    f["wd_sin"] = np.sin(wd)
    f["wd_cos"] = np.cos(wd)

    hour = idx.hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    doy = idx.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def build_xy(df: pd.DataFrame, feats: pd.DataFrame, h: int):
    """返回 (X, y, issue_time, target_time)，已剔除缺测与边界样本。"""
    y = df["power"].shift(-h)
    target_time = df.index + pd.Timedelta(hours=h)
    mask = y.notna() & feats.notna().all(axis=1)
    return feats[mask], y[mask], df.index[mask], target_time[mask]


def split_masks(issue_time: pd.DatetimeIndex, target_time: pd.DatetimeIndex, h: int):
    """训练/测试按时间切分，并在切分点留出 h 小时的隔离带（purge）。"""
    test_start = pd.Timestamp(cfg.TEST_START, tz="UTC")
    train = (issue_time < test_start) & (target_time < test_start)
    test = issue_time >= test_start
    return np.asarray(train), np.asarray(test)
