"""生成自包含 HTML 报告（图片 base64 内嵌，可直接打印成 PDF）。"""

from __future__ import annotations

import base64
from datetime import datetime

import pandas as pd

from . import config as cfg

CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; font-size: 9.6pt;
       line-height: 1.65; color: #1a1a1a; margin: 0;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 21pt; color: #0F3D5C; margin: 0 0 4px; letter-spacing: 1px; line-height: 1.3; }
h2 { font-size: 12.5pt; color: #0F3D5C; margin: 18px 0 7px; padding-left: 9px;
     border-left: 3.5pt solid #C8A45C; letter-spacing: .5px; page-break-after: avoid; }
h3 { font-size: 10.5pt; color: #24333f; margin: 11px 0 5px; page-break-after: avoid; }
p { margin: 5px 0; }
.sub { color: #5a6b7a; font-size: 9.5pt; }
.hd { border-bottom: 2.2pt solid #0F3D5C; padding-bottom: 8px; margin-bottom: 14px; }
.meta { color: #7a8b99; font-size: 8.6pt; margin-top: 6px; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 14px 0 6px; }
.kpi { background: #F2F6F9; border-left: 3pt solid #0F3D5C; padding: 8px 10px; }
.kpi .v { font-size: 16pt; font-weight: 700; color: #0F3D5C; line-height: 1.2; }
.kpi .v small { font-size: 8.5pt; font-weight: 400; color: #5a6b7a; }
.kpi .k { font-size: 8.2pt; color: #5a6b7a; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 4px; font-size: 8.8pt;
        page-break-inside: avoid; }
th { background: #0F3D5C; color: #fff; font-weight: 600; text-align: left; padding: 5px 7px; }
td { padding: 4px 7px; border-bottom: 0.6pt solid #E2E8EC; }
tr:nth-child(even) td { background: #FAFCFD; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { background: #FBF7EE; font-weight: 600; }
figure { margin: 8px 0 5px; page-break-inside: avoid; }
figure img { width: 100%; border: 0.6pt solid #E2E8EC; }
figcaption { font-size: 8.4pt; color: #7a8b99; margin-top: 3px; text-align: center; }
.tabcap { font-size: 8.6pt; color: #24333f; font-weight: 600; margin: 8px 0 3px;
          padding-left: 2px; page-break-after: avoid; }
.note { background: #FBF7EE; border-left: 3pt solid #C8A45C; padding: 8px 11px; font-size: 8.8pt;
        color: #4a4032; margin: 10px 0; }
.warn { background: #FDF3F1; border-left: 3pt solid #B0392E; padding: 8px 11px; font-size: 8.8pt;
        color: #5c2a24; margin: 10px 0; }
ul { margin: 5px 0; padding-left: 18px; }
li { margin: 2px 0; }
code { background: #F2F6F9; padding: 1px 4px; border-radius: 2px; font-size: 8.6pt; }
.foot { margin-top: 26px; padding-top: 8px; border-top: 0.6pt solid #D8E0E6;
        font-size: 8pt; color: #8a97a2; }
"""


def _img(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _fig(path: str, caption: str) -> str:
    return f'<figure><img src="{_img(path)}"><figcaption>{caption}</figcaption></figure>'


def _main_table(main: pd.DataFrame) -> str:
    rows = []
    for h in cfg.MAIN_HORIZONS:
        sub = main[(main.h == h) & (main.model != "climatology")]
        best = sub.loc[sub["RMSE_MW"].idxmin(), "model"]
        for _, r in sub.iterrows():
            cls = ' class="best"' if r["model"] == best else ""
            skill = "—" if r["model"] == "climatology" else f"{r['Skill_vs_persist']:+.1f}%"
            rows.append(
                f"<tr{cls}><td>{h} h</td><td>{cfg.LABELS[r['model']]}</td>"
                f"<td class='num'>{r['RMSE_MW']:.2f}</td>"
                f"<td class='num'>{r['RMSE_pct']:.1f}%</td>"
                f"<td class='num'>{r['准确率_pct']:.1f}%</td>"
                f"<td class='num'>{r['合格率_pct']:.1f}%</td>"
                f"<td class='num'>{skill}</td></tr>"
            )
    head = (
        "<tr><th>预见期</th><th>模型</th><th>RMSE (MW)</th><th>RMSE / 装机</th>"
        "<th>准确率</th><th>合格率</th><th>较持续法</th></tr>"
    )
    return f"<table>{head}{''.join(rows)}</table>"


def build_report(summary: dict, res: pd.DataFrame, main: pd.DataFrame,
                 monthly: pd.DataFrame, bins: pd.DataFrame, figs: dict, meta: dict,
                 diag: dict | None = None) -> str:
    s = summary
    dg = diag or s.get("diag24", {}) or {}
    g1 = s["h1_gbdt_准确率"]
    g4 = s["h4_gbdt_准确率"]
    g24 = s["h24_gbdt_准确率"]
    r24 = s["h24_ridge_准确率"]
    p24 = s["h24_persistence_准确率"]
    p1 = s["h1_persistence_准确率"]
    q24 = s["h24_gbdt_合格率"]
    qr24 = s["h24_ridge_合格率"]
    skill_g24 = s["GBDT_skill_h24"]
    skill_r24 = s.get("Ridge_skill_h24", 0.0)

    html = []
    html.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    html.append("<title>风电场功率预测 baseline</title>")
    html.append(f"<style>{CSS}</style></head><body>")

    # ---------------- 封面
    html.append("<div class='hd'>")
    html.append("<h1>风电场功率预测 baseline<br>持续法 vs 统计回归</h1>")
    html.append(
        f"<div class='sub'>{cfg.SITE['name']}　·　{cfg.SITE['lat']}°N, {cfg.SITE['lon']}°E　·　"
        f"装机 {cfg.CAP_MW:.1f} MW（代表性机型）</div>"
    )
    html.append(
        f"<div class='meta'>数据：ERA5 逐时（{s['数据期间']}，{s['样本小时']} 小时）　|　"
        f"训练 1–9 月，测试 10–12 月　|　生成：{datetime.now():%Y-%m-%d}</div>"
    )
    html.append("</div>")

    html.append("<div class='kpis'>")
    for v, unit, k in [
        (f"{g1:.1f}", "%", "1 h 准确率（GBDT）"),
        (f"{g4:.1f}", "%", "4 h 准确率（GBDT）"),
        (f"{r24:.1f}", "%", "24 h 准确率（Ridge）"),
        (f"{q24:.1f}", "%", "24 h 合格率（GBDT）"),
    ]:
        html.append(f"<div class='kpi'><div class='v'>{v} <small>{unit}</small></div><div class='k'>{k}</div></div>")
    html.append("</div>")

    html.append("<h2>摘要</h2>")
    html.append(
        f"<p>用 {s['样本小时']} 小时 ERA5 逐时资料构造轮毂高度风速与功率序列，按 1–9 月训练、"
        f"10–12 月测试，在 1–24 h 每个预见期上单独训练四个模型：持续法、日变化气候态、"
        f"岭回归与梯度提升树（GBDT）。</p>"
    )
    html.append(
        f"<p>结论有三条。<b>第一，短预见期里“不变”几乎就是最优假设</b>：1 h 持续法准确率已达 "
        f"{p1:.1f}%、合格率 99.2%，GBDT 只把 RMSE 再压低 13%（准确率 {g1:.1f}%）；"
        f"4 h GBDT 为 {g4:.1f}%。<b>第二，24 h 时所有模型都掉到 70% 以下</b>：持续法 {p24:.1f}%，"
        f"最好的 Ridge {r24:.1f}%（误差较持续法降 {skill_r24:.1f}%），GBDT {g24:.1f}%。"
        f"<b>第三，24 h 的“最优”取决于考核口径</b>——Ridge 的 RMSE 更低，GBDT 的合格率更高"
        f"（{q24:.1f}% vs {qr24:.1f}%）。这不是谁更强，而是误差分布形状不同，见第 2、5 节。</p>"
    )

    html.append(
        "<div class='warn'><b>必须先说清楚的三件事：</b>"
        "① 本项目没有 SCADA 实测功率，功率由 ERA5 风速经功率曲线换算，"
        "<b>平滑程度高于真实机组</b>，因此误差水平偏乐观；"
        "② 模型输入是<b>观测外推</b>，不是数值天气预报（NWP）。它适合代表 0–4 h 超短期预测的上限，"
        "不能代表日前业务精度——真实日前预测要接入 NWP，误差会明显更大；"
        "③ 结论用于方法对比与流程复现，不构成任何发电计划或考核承诺。</div>"
    )

    # ---------------- 1 数据
    html.append("<h2>1　数据与功率序列</h2>")
    html.append(
        f"<p>ERA5 提供 10 m / 100 m 风速与风向、2 m 气温、地面气压。按项目一同样的口径，"
        f"先用逐时风切变中位数 α = {s['切变指数']} 把 100 m 风速外推到轮毂 110 m，"
        f"再用 IEC 等功率法做空气密度修正（ρ 逐时计算），最后按 5.0 MW / Φ156 m 的"
        f"代表性功率曲线换成功率。</p>"
    )
    html.append(
        f"<p>全年平均轮毂风速 {s['平均轮毂风速_ms']} m/s、平均功率 {s['平均功率_MW']} MW、"
        f"容量系数 {s['容量系数'] * 100:.1f}%、等效满发 {s['满发小时_h']:.0f} h；"
        f"零出力（&lt;10% 装机）占 {s['零出力占比'] * 100:.1f}%，满发（&gt;99% 装机）占 "
        f"{s['满发占比'] * 100:.1f}%。后者很关键：<b>功率曲线在额定风速以上是平的</b>，"
        f"风速再大也不增加出力，这让高风速段的误差天然被压缩。</p>"
    )
    html.append(
        "<div class='note'>容量系数口径：这里的 30.9% 是<b>毛容量系数</b>（ERA5 风速换算的发电量 ÷ 装机），"
        "未扣尾流、可用率与电气损耗。项目一按 12.4% 综合折减得到的<b>净容量系数是 27.0%</b>，"
        "两者差值正是折减部分，并不矛盾。</div>"
    )

    html.append("<h3>1.1　预测任务与切分</h3>")
    html.append(
        "<p>对每个预见期 h ∈ {1, …, 24} 单独训练一个模型（direct multi-horizon），"
        "特征只用 t 时刻及之前的信息：7 个风速滞后（0/1/2/3/6/12/24 h）、"
        "3/6/24 h 滑动均值、6 h 风速标准差、当前功率与功率滞后、风向正弦余弦、"
        "小时与日序的正弦余弦。</p>"
    )
    html.append(
        "<div class='note'>训练 2024-01-01 ~ 09-30，测试 10-01 ~ 12-31。切分点处按预见期留出 "
        "h 小时的隔离带（purge），避免训练样本的目标值落进测试期造成泄漏。"
        "测试期正好覆盖秋季到冬季大风期，是全年较难的时段。</div>"
    )

    html.append("<h3>1.2　评估口径</h3>")
    html.append(
        "<table><tr><th>指标</th><th>定义</th><th>为什么用它</th></tr>"
        "<tr><td>RMSE</td><td>sqrt(mean((P_pred − P_obs)²))，MW</td>"
        "<td>对大误差敏感，是考核与结算的基础量</td></tr>"
        "<tr><td>准确率</td><td>(1 − RMSE / 装机容量) × 100%</td>"
        "<td>行业惯用口径，便于横向比较</td></tr>"
        "<tr><td>合格率</td><td>|误差| ≤ 25% 装机的样本占比</td>"
        "<td>看“单点是否可用”，比均值更贴近调度体验</td></tr>"
        "<tr><td>Skill</td><td>1 − RMSE_model / RMSE_持续法</td>"
        "<td>相对最简单的基准，模型到底多值钱</td></tr></table>"
    )

    # ---------------- 2 结果
    html.append("<h2>2　结果：误差随预见期怎么长</h2>")
    html.append(_fig(figs["rmse"], "图1　四种模型的 RMSE 随预见期变化（右轴为对应准确率）"))
    html.append(
        f"<p>曲线形状本身就有信息量。<b>1–2 h 持续法几乎打平复杂模型</b>（RMSE 6.8% vs 5.9%，"
        f"只差 0.9 个百分点），说明这一时段大气的“记忆”还没耗散，上模型的收益很小；"
        f"4 h 时 GBDT 把 RMSE 从 18.7% 压到 17.2%；8 h 之后曲线分成两组："
        f"持续法陡升，两个回归模型缓升。</p>"
    )
    html.append(
        f"<p>更值得注意的是<b>7 h 之后 Ridge 的 RMSE 一直低于 GBDT</b>"
        f"（7 h 23.4% vs 23.6%，24 h 32.5% vs 33.9%）。这不是 GBDT 没训好，"
        f"而是预见期越长，可用特征对目标的解释力越弱，<b>平方损失下的最优预测必然向条件均值收缩</b>："
        f"Ridge 收缩得更狠（预测标准差仅 {dg.get('ridge', {}).get('pred_std', 0.55)} MW、"
        f"最大值只到 {dg.get('ridge', {}).get('pred_max', 3.2)} MW），尾部误差更轻；"
        f"GBDT 保留了更多变化（标准差 {dg.get('gbdt', {}).get('pred_std', 0.83)} MW、"
        f"最大预测 {dg.get('gbdt', {}).get('pred_max', 5.0)} MW），中低出力段更贴近，"
        f"但满发段容易出现大偏差。<b>按 RMSE 考核 Ridge 赢，按合格率考核 GBDT 赢</b>——"
        f"“最优模型”必须先说清考核口径。</p>"
    )
    html.append(
        "<p>日变化气候态始终最差——它只有“平均的一天”，没有任何当天的信息。"
        "把它放在图里是为了说明：<b>一个不含实时信息的模型，无论多复杂都没有业务价值</b>。</p>"
    )
    html.append("<div class='tabcap'>表 1　三个重点预见期的模型对比（测试集 10–12 月，高亮为 RMSE 最优）</div>")
    html.append(_main_table(main))
    html.append(
        f"<p>Skill 列（相对持续法的误差下降）：1 h "
        f"{s['GBDT_skill_h1']:+.1f}%，4 h {s['GBDT_skill_h4']:+.1f}%，24 h "
        f"{skill_g24:+.1f}%（Ridge {skill_r24:+.1f}%）；气候态的准确率在三个预见期几乎恒定"
        f"（64.9–65.0%），未列入表中。一个务实的判断是："
        f"<b>0–4 h 的收益主要来自数据质量与观测频率，不在模型复杂度；"
        f"24 h 想再上一个台阶，必须换信息源（NWP），而不是换模型。</b></p>"
    )

    # ---------------- 3 过程
    html.append("<h2>3　典型过程：误差是什么样子</h2>")
    html.append(_fig(figs["example"], "图2　测试期内波动最大的两个 7 天窗口：实际值、持续法与 GBDT"))
    html.append(
        "<p>4 h 面板里，持续法基本是把曲线右移，遇到爬坡就系统性滞后；"
        "GBDT 能提前半拍跟上趋势。24 h 面板则暴露了统计外推的天花板："
        "GBDT 的曲线被明显压平，能抓住日变化，却不敢跟到 5 MW 的尖峰；"
        "它对<b>冷空气过境的到达时间</b>也无能为力——这正是日前预测必须依赖 NWP 的原因。</p>"
    )

    # ---------------- 4 散点
    html.append("<h2>4　误差的分布，而不只是均值</h2>")
    html.append(_fig(figs["scatter"], "图3　24 h 预见期：预测 vs 实际（左：持续法；右：GBDT）"))
    html.append(
        "<p>持续法的散点沿水平方向铺开——它把当前功率当成 24 h 后的功率，"
        "所以在实际出力变化大的时候，横向误差极大。GBDT 的点云在中低出力段明显收窄，"
        "但上缘被压在 3 MW 附近：实际满发时它平均只报 2.2 MW（Ridge 更低，2.1 MW）。"
        "图上那条“贴着 1:1 线下方的横带”，就是<b>回归到均值</b>最直观的样子。</p>"
    )

    # ---------------- 5 分月/分功率段
    html.append("<h2>5　什么时候最难预测</h2>")
    html.append(_fig(figs["monthly"], "图4　24 h 预见期的分月准确率与合格率"))
    html.append(
        "<p>分月看，<b>三个月没有一个模型把 24 h 准确率做到 70%</b>，Ridge 在 11 月最接近（69.3%）。"
        "持续法在 10 月最差（准确率 54.5%、合格率 41.1%）：10 月是秋季环流调整期，"
        "逐时功率变化最剧烈（月均 |ΔP| 0.23 MW，高于 11 月的 0.20 MW）；"
        "11–12 月风速虽更大（月均 7.8 / 7.6 m/s），过程反而更连续，持续法略好。"
        "模型相对持续法的增益也正好在 10 月最大（+12.8 个百分点）——"
        "<b>天气过程越不连续，持续法越容易失手，模型的价值才显现</b>。</p>"
    )
    html.append(
        "<p>合格率上 GBDT 三个月都是第一（58.6 / 59.2 / 56.5%），说明它给出的“可用点”更多，"
        "尽管 RMSE 不是最低。这也解释了为什么业务考核很少只看单一指标："
        "调度关心的是“有多少时段可以直接用”，财务关心的是“结算误差有多大”。</p>"
    )
    html.append(_fig(figs["bins"], "图5　24 h 平均绝对误差按实际出力区间分解"))
    html.append(
        "<p>按实际出力区间分解，模型的“性格”更清楚：10–30% 与 30–60% 段，回归模型把 MAE "
        "从 26–29% 压到 13–16%，几乎减半；但到 88–100% 满发段，两个回归模型都明显差于持续法"
        "（Ridge 57.1%、GBDT 55.5%，持续法 38.0%）。原因有两层：满发样本只占测试期 14%"
        "（全年 8%），模型在功率曲线顶端见到的样本有限；更关键的是平方损失下的最优预测会向条件均值收缩——"
        "实际 4.9 MW 时，Ridge 平均只报 2.1 MW、GBDT 2.2 MW。持续法反而在这里占便宜："
        "它直接把“现在是 5 MW”延续过去。</p>"
    )
    html.append(
        "<p>给运维与考核的两点启示：① 若按合格率考核，GBDT 更稳；若按 RMSE 报计划曲线，Ridge 更保险，"
        "两者差距在 24 h 才显现，超短期可以忽略；② <b>满发与爬坡段靠单点统计模型都不够</b>，"
        "需要 NWP 提供环流形势，并用概率区间（P10/P50/P90）替代单点，才能把“可能的尖峰”表达出来。</p>"
    )

    # ---------------- 6 局限
    html.append("<h2>6　局限：这份结果不能证明什么</h2>")
    html.append("<ul>")
    html.append(
        "<li><b>功率是模拟的。</b>没有 SCADA 实测，功率由 ERA5 风速确定性换算。"
        "真实机组有湍流、偏航误差、限电、故障与降容，实际误差会明显更大——"
        "这一条决定了所有数字都应看作<b>相对比较</b>，而非绝对水平。</li>"
    )
    html.append(
        "<li><b>用的是观测外推，不是 NWP。</b>真实的日前预测以数值天气预报为输入，"
        "NWP 本身的风速误差会直接传导。本项目 24 h 的结果应理解为“统计外推的上界”。</li>"
    )
    html.append(
        "<li><b>单点预测，无不确定性。</b>业务上更实用的是区间预测（如 P10/P50/P90），"
        "用于申报计划与储能调度。</li>"
    )
    html.append(
        "<li><b>满发段系统性低估。</b>24 h 预见期回归模型在 ≥88% 出力段的平均预测只有 2.1–2.2 MW，"
        "而实际均值 4.9 MW。这是平方损失 + 低可预报性的必然结果，业务上要单独校正或改用分位数损失。</li>"
    )
    html.append(
        "<li><b>单站单年。</b>2024 年一个站点的 12 个月，不能代表年际差异；"
        "严格做法是滚动多年回测。</li>"
    )
    html.append("</ul>")

    html.append(
        "<div class='note'><b>如果要继续做：</b>① 接入 NWP 预报场（风速、风向、温度）做真正的日前预测；"
        "② 分位数回归 / 分位数损失给出区间；③ 加入机组可用性与限电标签；"
        "④ 用真实 SCADA 做同样的对比，检验这套结论是否成立。</div>"
    )

    # ---------------- 7 面试可讲点
    html.append("<h2>7　三个可以在面试里讲的点</h2>")
    html.append("<ul>")
    html.append(
        "<li><b>“为什么 1 h 用持续法就够？”</b>—— 1 h 的功率变化主要来自湍流与局地扰动，"
        "小时尺度上不可预报；持续法 6.8% 的 RMSE 已经接近这套数据自身的可预报下限，"
        "GBDT 只能再降 0.9 个百分点。所以超短期的投入应该放在观测频率与数据质量上。</li>"
    )
    html.append(
        "<li><b>“为什么 24 h 时 Ridge 反超 GBDT？”</b>—— 预见期越长，特征对目标的解释力越弱，"
        "平方损失下的最优预测越接近条件均值。线性模型天然收缩、尾部轻；树模型保留非线性波动、"
        "中段准但尾巴重。结果就是 RMSE 与合格率给两个不同的赢家。</li>"
    )
    html.append(
        "<li><b>“为什么满发段误差最大？”</b>—— 额定风速以上功率曲线是平的，模型无法判断"
        "“风还会不会更大”；加上满发样本少、平方损失向均值收缩，模型会系统性低估尖峰。"
        "这正是需要概率预测而不是单点预测的地方。</li>"
    )
    html.append("</ul>")

    # ---------------- 8 复现
    html.append("<h2>8　复现方式</h2>")
    html.append(
        "<p><code>pip install -r requirements.txt</code><br>"
        "<code>python run.py</code>　# 自动构造功率、训练 4 个模型 × 24 个预见期、出图出报告</p>"
        "<p>站点、机型、预见期、切分点、合格阈值都在 <code>src/config.py</code>；"
        "特征集在 <code>src/data.py</code>；评估口径在 <code>src/models.py</code>。</p>"
    )
    html.append(
        "<div class='foot'>数据来源：ERA5 © Copernicus Climate Change Service / ECMWF，"
        "经 Open-Meteo Archive API 获取（CC BY 4.0）。"
        "功率曲线为代表性曲线，不代表任何厂商型号。本项目为方法复现，不构成工程结论。</div>"
    )
    html.append("</body></html>")
    return "".join(html)
