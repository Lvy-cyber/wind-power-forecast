"""一键跑通：ERA5 → 功率序列 → 四个模型 × 24 个预见期 → 评估 → 出图 → 出报告。

用法：
    python run.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src import config as cfg
from src import data as D
from src import models as M
from src import plots
from src.report import build_report


def _ensure_dirs() -> None:
    for d in (cfg.OUT_FIG, cfg.OUT_TAB, cfg.OUT_REPORT, "data/processed"):
        os.makedirs(d, exist_ok=True)


def run() -> None:
    _ensure_dirs()

    print("[1/6] 构造功率序列")
    df, meta = D.prepare()
    print(
        f"      {len(df)} 小时 | 轮毂风速 {meta['mean_ws_hub']:.2f} m/s | "
        f"平均功率 {meta['mean_power_mw']:.2f} MW | CF {meta['capacity_factor'] * 100:.1f}%"
    )

    feats = D.feature_frame(df)

    print("[2/6] 逐预见期训练与预测")
    rows: list[dict] = []
    store: dict[int, dict] = {}
    for h in cfg.HORIZONS:
        X, y, issue_time, target_time = D.build_xy(df, feats, h)
        tr, te = D.split_masks(issue_time, target_time, h)
        if te.sum() < 100:
            continue

        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]

        p_now = df["power"].reindex(issue_time[te]).values
        clim_tab = M.climatology_table(ytr.values, target_time[tr])
        clim = clim_tab.reindex(target_time[te].hour).values

        preds = {
            m: M.fit_predict(m, Xtr, ytr, Xte, {"p_now": p_now, "clim": clim})
            for m in M.MODELS
        }
        for m, p in preds.items():
            rows.append({"h": h, "model": m, **M.score(yte.values, p)})

        store[h] = {
            "y": yte.values,
            "preds": preds,
            "target_time": target_time[te],
            "issue_time": issue_time[te],
        }
        print(f"      h={h:>2} h | 样本 {int(te.sum()):>5} | RMSE(persist) {rows[-4]['RMSE_pct']:>5.1f}% | "
              f"RMSE(gbdt) {rows[-1]['RMSE_pct']:>5.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(f"{cfg.OUT_TAB}/metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 3. 基准表
    print("[3/6] 汇总主表")
    main = res[res.h.isin(cfg.MAIN_HORIZONS)].copy()
    main["Skill_vs_persist"] = 0.0
    for h in cfg.MAIN_HORIZONS:
        base = res[(res.h == h) & (res.model == "persistence")]["RMSE_MW"].iloc[0]
        m = main.h == h
        main.loc[m, "Skill_vs_persist"] = 1.0 - main.loc[m, "RMSE_MW"] / base
    main["Skill_vs_persist"] = (main["Skill_vs_persist"] * 100).round(1)
    main.to_csv(f"{cfg.OUT_TAB}/metrics_main.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 4. 分月与分功率段
    print("[4/6] 分月 / 分功率段")
    monthly_rows = []
    for h in cfg.MAIN_HORIZONS:
        st = store[h]
        tt = pd.DatetimeIndex(st["target_time"])
        for key, name in (("persistence", "Persistence"), ("ridge", "Ridge"), ("gbdt", "GBDT")):
            err = st["preds"][key] - st["y"]
            tmp = pd.DataFrame({"m": tt.month, "se": err**2, "ae": np.abs(err)})
            g = tmp.groupby("m")
            for month, sub in g:
                rmse = float(np.sqrt(sub["se"].mean()))
                monthly_rows.append(
                    {
                        "h": h,
                        "model": name,
                        "month": int(month),
                        "RMSE_MW": round(rmse, 3),
                        "准确率_pct": round((1 - rmse / cfg.CAP_MW) * 100, 2),
                        "合格率_pct": round(float((sub["ae"] <= cfg.PASS_RATIO * cfg.CAP_MW).mean() * 100), 2),
                    }
                )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(f"{cfg.OUT_TAB}/monthly_metrics.csv", index=False, encoding="utf-8-sig")

    h24 = store.get(24)
    bin_rows = []
    if h24 is not None:
        y = h24["y"]
        edges = [0, 0.5, 1.5, 3.0, 4.4, 5.01]
        labels = ["0–10%", "10–30%", "30–60%", "60–88%", "88–100%"]
        b = pd.cut(y, bins=edges, labels=labels, right=False)
        for key, name in (("persistence", "Persistence"), ("ridge", "Ridge"), ("gbdt", "GBDT")):
            err = np.abs(h24["preds"][key] - y)
            tmp = pd.DataFrame({"b": b, "ae": err, "se": err**2})
            g = tmp.groupby("b", observed=False)
            for lab, sub in g:
                if len(sub) < 20:
                    continue
                bin_rows.append(
                    {
                        "功率区间(占装机)": lab,
                        "model": name,
                        "样本数": int(len(sub)),
                        "MAE_MW": round(float(sub["ae"].mean()), 3),
                        "MAE_pct": round(float(sub["ae"].mean() / cfg.CAP_MW * 100), 2),
                    }
                )
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(f"{cfg.OUT_TAB}/error_by_power_bin.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 5. 出图
    print("[5/6] 出图")
    figs = {
        "rmse": plots.plot_rmse_horizon(res),
        "example": plots.plot_example(store, df),
        "scatter": plots.plot_scatter(store),
        "monthly": plots.plot_monthly(monthly),
        "bins": plots.plot_error_bins(bins),
    }

    # ---------------------------------------------------------- 6. 报告
    print("[6/6] 生成报告")
    summary = _summary(res, main, monthly, meta)

    # 24 h 诊断：预测值离散度 + 满发段的均值收缩，报告第 2/5 节引用
    diag: dict = {}
    if h24 is not None:
        y24 = h24["y"]
        for key in ("persistence", "ridge", "gbdt"):
            pp = h24["preds"][key]
            diag[key] = {
                "pred_mean": round(float(pp.mean()), 2),
                "pred_std": round(float(pp.std()), 2),
                "pred_max": round(float(pp.max()), 2),
            }
        full = y24 >= 0.88 * cfg.CAP_MW
        diag["full_obs"] = round(float(y24[full].mean()), 2)
        diag["full_n"] = int(full.sum())
        for key in ("persistence", "ridge", "gbdt"):
            diag[f"full_{key}"] = round(float(h24["preds"][key][full].mean()), 2)
    summary["diag24"] = diag
    with open(f"{cfg.OUT_TAB}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    html = build_report(summary=summary, res=res, main=main, monthly=monthly,
                        bins=bins, figs=figs, meta=meta, diag=diag)
    out_html = f"{cfg.OUT_REPORT}/功率预测baseline报告.html"
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"      报告：{out_html}")

    print("\n===== 关键结果（测试集 10–12 月）=====")
    for h in cfg.MAIN_HORIZONS:
        p = res[(res.h == h) & (res.model == "persistence")].iloc[0]
        g = res[(res.h == h) & (res.model == "gbdt")].iloc[0]
        r = res[(res.h == h) & (res.model == "ridge")].iloc[0]
        print(
            f"{h:>2} h | 持续法 准确率 {p['准确率_pct']:5.1f}% 合格率 {p['合格率_pct']:5.1f}% | "
            f"回归 准确率 {r['准确率_pct']:5.1f}% | GBDT 准确率 {g['准确率_pct']:5.1f}% 合格率 {g['合格率_pct']:5.1f}%"
        )


def _summary(res: pd.DataFrame, main: pd.DataFrame, monthly: pd.DataFrame, meta: dict) -> dict:
    out: dict = {
        "站点": f"{cfg.SITE['name']}（{cfg.SITE['province']}）",
        "坐标": f"{cfg.SITE['lat']}°N, {cfg.SITE['lon']}°E",
        "样本小时": meta["rows"],
        "数据期间": f"{meta['start'][:10]} ~ {meta['end'][:10]}",
        "切变指数": round(meta["alpha"], 3),
        "平均轮毂风速_ms": round(meta["mean_ws_hub"], 2),
        "平均功率_MW": round(meta["mean_power_mw"], 2),
        "容量系数": round(meta["capacity_factor"], 4),
        "满发小时_h": round(meta["full_load_hours"], 0),
        "零出力占比": round(meta["calm_rate"], 4),
        "满发占比": round(meta["rated_rate"], 4),
        "训练期": "2024-01-01 ~ 2024-09-30",
        "测试期": f"{cfg.TEST_START} ~ 2024-12-31",
    }
    for h in cfg.MAIN_HORIZONS:
        sub = main[main.h == h].set_index("model")
        for m in M.MODELS:
            if m not in sub.index:
                continue
            s = sub.loc[m]
            out[f"h{h}_{m}_准确率"] = float(s["准确率_pct"])
            out[f"h{h}_{m}_合格率"] = float(s["合格率_pct"])
            out[f"h{h}_{m}_RMSE"] = float(s["RMSE_MW"])
    out["GBDT_skill_h1"] = float(main[(main.h == 1) & (main.model == "gbdt")]["Skill_vs_persist"].iloc[0])
    out["GBDT_skill_h4"] = float(main[(main.h == 4) & (main.model == "gbdt")]["Skill_vs_persist"].iloc[0])
    out["GBDT_skill_h24"] = float(main[(main.h == 24) & (main.model == "gbdt")]["Skill_vs_persist"].iloc[0])
    out["Ridge_skill_h24"] = float(main[(main.h == 24) & (main.model == "ridge")]["Skill_vs_persist"].iloc[0])
    return out


if __name__ == "__main__":
    run()
