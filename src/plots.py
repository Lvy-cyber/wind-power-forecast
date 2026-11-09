"""出图：误差随预见期、典型过程、散点、分月、分功率段。"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from . import config as cfg

for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
        break
plt.rcParams.update(
    {
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.edgecolor": "#9aa9b4",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#24333f",
        "text.color": "#1a1a1a",
        "xtick.color": "#4a5b6a",
        "ytick.color": "#4a5b6a",
        "grid.color": "#dfe6ea",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)

NAVY = "#0F3D5C"
GOLD = "#C8A45C"
RED = "#B0392E"
GREY = "#8AA2B2"


def _save(fig, name: str, tight: bool = True) -> str:
    os.makedirs(cfg.OUT_FIG, exist_ok=True)
    path = os.path.join(cfg.OUT_FIG, name)
    if tight:
        fig.savefig(path, facecolor="white", bbox_inches="tight")
    else:
        fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 图1

def plot_rmse_horizon(res: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8.6, 4.3))

    piv = res.pivot_table(index="h", columns="model", values="RMSE_pct")
    for m in ("persistence", "climatology", "ridge", "gbdt"):
        if m not in piv:
            continue
        label = cfg.LABELS[m]
        if m == "climatology":
            label = "日变化气候态（无实时信息，误差≈35%，不随预见期变化）"
        ax.plot(
            piv.index,
            piv[m],
            marker="o" if m in ("ridge", "gbdt") else None,
            ms=3.2,
            lw=2.0 if m in ("persistence", "gbdt") else 1.4,
            ls="--" if m == "climatology" else "-",
            color=cfg.COLORS[m],
            label=label,
            zorder=3 if m == "gbdt" else 2,
        )

    for h in cfg.MAIN_HORIZONS:
        ax.axvline(h, color="#DCE3E8", lw=0.8, zorder=0)

    ax.set_xlabel("预见期（小时）")
    ax.set_ylabel("RMSE（占装机容量 %）")
    ax.set_xlim(1, max(cfg.HORIZONS))
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax2 = ax.twinx()
    ax2.set_ylim(100 - ax.get_ylim()[1], 100 - ax.get_ylim()[0])
    ax2.set_ylabel("对应准确率 (%)", color="#6b7c8a")
    ax2.grid(False)

    ax.annotate("超短期\n0–4 h", xy=(2.6, ax.get_ylim()[1] * 0.88), ha="center", fontsize=7.5,
                color="#7a8b99")
    ax.annotate("日前\n24 h", xy=(22.4, ax.get_ylim()[1] * 0.86), ha="center", fontsize=7.5,
                color="#7a8b99")

    ax.set_title("图1　预测误差随预见期增长（测试集 10–12 月）", loc="left",
                 fontsize=11, fontweight="bold")
    return _save(fig, "fig1_rmse_horizon.png")


# ---------------------------------------------------------------- 图2

def _pick_window(tt: pd.DatetimeIndex, y: np.ndarray, days: int = 7):
    s = pd.Series(np.asarray(y), index=pd.DatetimeIndex(tt))
    sd = s.rolling(days * 24, min_periods=days * 20).std()
    end = sd.idxmax()
    return end - pd.Timedelta(days=days), end


def plot_example(store: dict, df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.4), sharex=False)

    for ax, h in zip(axes, (4, 24)):
        st = store[h]
        tt = pd.DatetimeIndex(st["target_time"])
        start, end = _pick_window(tt, st["y"])
        m = (tt >= start) & (tt <= end)

        ax.plot(tt[m], st["y"][m], color="#1a1a1a", lw=1.5, label="实际功率（ERA5 换算）")
        ax.plot(tt[m], st["preds"]["persistence"][m], color=GREY, lw=1.2, ls="--",
                label="持续法")
        ax.plot(tt[m], st["preds"]["gbdt"][m], color=NAVY, lw=1.4, label="GBDT")

        ax.set_ylim(-0.1, cfg.CAP_MW * 1.12)
        ax.set_ylabel("功率 (MW)")
        ax.set_title(f"{'（a）' if h == 4 else '（b）'}预见期 {h} h　"
                     f"{start:%m-%d} ~ {end:%m-%d}",
                     loc="left", fontsize=9.5)
        ax.grid(True, axis="y")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        if h == 4:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle("图2　典型周过程：持续法在 4 h 内可用，24 h 已明显滞后",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, "fig2_example.png", tight=False)


# ---------------------------------------------------------------- 图3

def plot_scatter(store: dict) -> str:
    st = store[24]
    y = st["y"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)

    for ax, key, title in zip(
        axes, ("persistence", "gbdt"), ("持续法", "GBDT")
    ):
        ax.scatter(y, st["preds"][key], s=5, alpha=0.25,
                   color=GREY if key == "persistence" else NAVY, linewidths=0)
        ax.plot([0, cfg.CAP_MW], [0, cfg.CAP_MW], color=RED, lw=1.0, ls="--")
        rmse = np.sqrt(np.mean((st["preds"][key] - y) ** 2))
        ax.set_title(f"{title}　RMSE {rmse:.2f} MW（{rmse / cfg.CAP_MW * 100:.1f}%）",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("实际功率 (MW)")
        ax.set_xlim(0, cfg.CAP_MW * 1.02)
        ax.set_ylim(0, cfg.CAP_MW * 1.02)
        ax.grid(True)
        ax.set_aspect("equal")

    axes[0].set_ylabel("预测功率 (MW)")
    fig.suptitle("图3　24 h 预见期：散点越贴近 1:1 线越好", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "fig3_scatter.png", tight=False)


# ---------------------------------------------------------------- 图4

def plot_monthly(monthly: pd.DataFrame) -> str:
    sub = monthly[monthly.h == 24]
    months = sorted(sub.month.unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45))
    w = 0.26
    x = np.arange(len(months))
    series = (("Persistence", GREY), ("Ridge", GOLD), ("GBDT", NAVY))

    for ax, col, title in zip(
        axes, ("准确率_pct", "合格率_pct"), ("准确率 (%)", "合格率 (%)")
    ):
        for i, (name, color) in enumerate(series):
            vals = [
                sub[(sub.model == name) & (sub.month == m)][col].iloc[0] if len(
                    sub[(sub.model == name) & (sub.month == m)]) else np.nan
                for m in months
            ]
            ax.bar(x + (i - 1) * w, vals, width=w, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m} 月" for m in months])
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, axis="y")
        ax.set_ylim(0, 100)
        if col == "准确率_pct":
            ax.axhline(cfg.BENCHMARK_ACC, color=RED, ls="--", lw=1.0)
            ax.text(len(months) - 0.55, cfg.BENCHMARK_ACC + 1.5, "80% 基准",
                    fontsize=7, color=RED, ha="right")

    axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("图4　24 h 分月表现：三个月都没有模型破 70%，GBDT 合格率每月最高",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig4_monthly.png", tight=False)


# ---------------------------------------------------------------- 图5

def plot_error_bins(bins: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    order = ["0–10%", "10–30%", "30–60%", "60–88%", "88–100%"]
    cats = [c for c in order if c in set(bins["功率区间(占装机)"])]
    x = np.arange(len(cats))
    w = 0.26

    for i, (name, color) in enumerate((("Persistence", GREY), ("Ridge", GOLD), ("GBDT", NAVY))):
        vals = []
        for c in cats:
            row = bins[(bins.model == name) & (bins["功率区间(占装机)"] == c)]
            vals.append(float(row["MAE_pct"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 1) * w, vals, width=w, color=color, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_xlabel("实际出力区间（占装机容量）")
    ax.set_ylabel("平均绝对误差（占装机 %）")
    ax.set_ylim(0, 66)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax.set_title("图5　24 h 误差按出力区间：中低出力段回归模型减半，满发段反而不如持续法",
                 loc="left", fontsize=11, fontweight="bold")
    return _save(fig, "fig5_error_bins.png")
