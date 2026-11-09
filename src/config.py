"""配置：站点、机型、预测任务定义、评估口径。

所有"可被追问"的假设都在这里，报告与 README 直接引用。
"""

from __future__ import annotations

# ---------------------------------------------------------------- 站点
SITE = {
    "name": "辉腾锡勒",
    "province": "内蒙古自治区乌兰察布市",
    "lat": 41.25,
    "lon": 112.75,
    "elevation_m": 2002,
    "hub_height_m": 110.0,
    "ref_height_m": 100.0,
}

# ---------------------------------------------------------------- 机型（与项目一一致）
CAP_MW = 5.0
TURBINE_CURVE = (
    (0.0, 0.0),
    (3.0, 0.0),
    (4.0, 0.25),
    (5.0, 0.55),
    (6.0, 1.00),
    (7.0, 1.60),
    (8.0, 2.40),
    (9.0, 3.30),
    (10.0, 4.30),
    (10.5, 5.00),
    (25.0, 5.00),
    (25.1, 0.0),
)

RHO_REF = 1.225  # kg/m3，功率曲线标定工况

# ---------------------------------------------------------------- 任务
HORIZONS = tuple(range(1, 25))       # 1~24 h 全曲线
MAIN_HORIZONS = (1, 4, 24)           # 表内重点：超短期 / 短中期 / 日前
TEST_START = "2024-10-01"            # 训练 1~9 月，测试 10~12 月
PASS_RATIO = 0.25                    # 单点误差 <= 25% 装机容量记为合格
BENCHMARK_ACC = 80.0                 # 行业常见考核基准线（各地细则不同）

# ---------------------------------------------------------------- 路径
DATA_RAW = "data/raw/era5_huitengxile_2024.csv"
DATA_META = "data/raw/era5_huitengxile_2024.meta.json"
DATA_POWER = "data/processed/power_hourly.csv"
OUT_FIG = "output/figures"
OUT_TAB = "output/tables"
OUT_REPORT = "report"

COLORS = {
    "persistence": "#8AA2B2",
    "climatology": "#B9C6D0",
    "ridge": "#C8A45C",
    "gbdt": "#0F3D5C",
}
LABELS = {
    "persistence": "持续法 (Persistence)",
    "climatology": "日变化气候态",
    "ridge": "线性回归 (Ridge)",
    "gbdt": "梯度提升树 (GBDT)",
}
