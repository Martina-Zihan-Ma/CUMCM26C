#!/usr/bin/env python3
"""问题二预测前置分析：仅用清洗后的附件2长表检验一月的可预测性。

运行：python scripts/analyze_january_forecastability.py
所有预测均在每日 00:00 生成，严格不读取预测日的 Load/PV 实测值。
"""
from __future__ import annotations

from pathlib import Path
import os
import warnings

ROOT = Path(__file__).resolve().parents[1]
# 缓存留在仓库可写范围，避免依赖用户主目录权限。
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats
from scipy.signal import periodogram


INPUT = ROOT / "data" / "processed" / "attachment2_actual_long.csv"
OUT = ROOT / "results" / "question2_forecast_analysis"
FIG = OUT / "figures"
N_PER_DAY, FIRST_TEST_DAY = 144, 14  # 0-based: 2025-01-15
VARIABLES = {"Load": "load_kw", "PV": "pv_actual_kw", "NetLoad": "net_load_kw"}


def set_font() -> None:
    font = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    if not INPUT.exists():
        found = list(ROOT.rglob("attachment2_actual_long.csv"))
        if not found:
            raise FileNotFoundError("未找到 attachment2_actual_long.csv")
        path = found[0]
    else:
        path = INPUT
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date", "datetime"])
    needed = ["date", "time_label", "time_index", "datetime", "load_kw", "pv_actual_kw", "load_kwh", "pv_actual_kwh"]
    assert set(needed).issubset(df), "清洗长表缺少必要字段"
    assert len(df) == 365 * N_PER_DAY and not df[needed].isna().any().any(), "行数或关键字段缺失异常"
    assert not df.datetime.duplicated().any() and df.time_index.between(0, 143).all(), "datetime/time_index异常"
    assert np.allclose(df.load_kwh, df.load_kw / 6) and np.allclose(df.pv_actual_kwh, df.pv_actual_kw / 6), "kWh换算异常"
    df = df.sort_values("datetime").copy()
    df["net_load_kw"] = df.load_kw - df.pv_actual_kw
    df["pv_ratio"] = np.divide(df.pv_actual_kw, df.load_kw, out=np.zeros(len(df)), where=df.load_kw.to_numpy() != 0)
    return df


def jan_matrix(jan: pd.DataFrame, column: str) -> np.ndarray:
    return jan.pivot(index="date", columns="time_index", values=column).to_numpy(float)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-lag], x[lag:]) / denom if lag else 1.0 for lag in range(max_lag + 1)])


def spectrum(x: np.ndarray) -> list[dict[str, float]]:
    f, p = periodogram(x, detrend="linear")
    usable = np.where(f > 0)[0]
    top = usable[np.argsort(p[usable])[-5:]][::-1]
    return [{"period_hours": float(1 / f[i] / 6), "power": float(p[i])} for i in top]


def plot_time_series(jan: pd.DataFrame, label: str, col: str) -> None:
    for window, suffix in [(jan, "full_month"), (jan.iloc[:7 * N_PER_DAY], "first_7_days")]:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(window.datetime, window[col], lw=.65, color="#2878b5")
        ax.set(title=f"2025年1月{label}时间序列 / {suffix}", xlabel="日期时间", ylabel="功率 / kW")
        ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(FIG / f"{label.lower()}_{suffix}.png", dpi=180); plt.close(fig)


def plot_heatmap(matrix: np.ndarray, label: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set(title=f"2025年1月{label}：日期-时刻热力图", xlabel="一天内时段（10分钟）", ylabel="日期（1月日序）")
    ax.set_xticks(range(0, 145, 24), [str(x // 6) for x in range(0, 145, 24)])
    ax.set_yticks(range(0, 31, 5), [str(x + 1) for x in range(0, 31, 5)])
    fig.colorbar(im, ax=ax, label="功率 / kW"); fig.tight_layout()
    fig.savefig(FIG / f"{label.lower()}_heatmap.png", dpi=180); plt.close(fig)


def plot_profiles(jan: pd.DataFrame, matrix: np.ndarray, label: str) -> None:
    x = np.arange(144) / 6
    q25, med, q75 = np.quantile(matrix, [.25, .5, .75], axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, matrix.mean(0), label="均值", lw=1.8); ax.plot(x, med, label="中位数", lw=1.2)
    ax.fill_between(x, q25, q75, alpha=.25, label="25%–75%")
    ax.set(title=f"2025年1月{label}平均日内曲线", xlabel="时刻 / h", ylabel="功率 / kW", xlim=(0, 24)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / f"{label.lower()}_intraday_profile.png", dpi=180); plt.close(fig)
    weekend = jan.date.dt.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(12, 4))
    for mask, name, color in [(~weekend, "工作日", "#2878b5"), (weekend, "周末", "#d95f02")]:
        ax.plot(x, jan.loc[mask].pivot(index="date", columns="time_index", values=label_to_col(label)).mean(), label=name, color=color)
    ax.set(title=f"2025年1月{label}：工作日与周末平均曲线", xlabel="时刻 / h", ylabel="功率 / kW", xlim=(0, 24)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / f"{label.lower()}_weekday_weekend.png", dpi=180); plt.close(fig)


def label_to_col(label: str) -> str:
    return VARIABLES[label]


def daily_features(jan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g in jan.groupby("date", sort=True):
        def peak_time(column: str) -> str:
            return str(g.loc[g[column].idxmax(), "time_label"])
        nonzero = g.loc[g.pv_actual_kw > 0]
        rows.append({"date": date, "is_weekend": date.dayofweek >= 5,
            "load_total_kwh": g.load_kwh.sum(), "load_mean_kw": g.load_kw.mean(), "load_max_kw": g.load_kw.max(), "load_peak_time": peak_time("load_kw"),
            "pv_total_kwh": g.pv_actual_kwh.sum(), "pv_max_kw": g.pv_actual_kw.max(), "pv_peak_time": peak_time("pv_actual_kw"),
            "pv_nonzero_intervals": len(nonzero), "pv_nonzero_duration_hours": len(nonzero) / 6,
            "netload_total_kwh": g.net_load_kw.sum() / 6, "netload_max_kw": g.net_load_kw.max(), "netload_min_kw": g.net_load_kw.min(), "netload_negative_intervals": int((g.net_load_kw < 0).sum())})
    return pd.DataFrame(rows)


def trend_stats(features: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in features if c not in {"date", "is_weekend"} and pd.api.types.is_numeric_dtype(features[c])]
    rows = []
    x = np.arange(len(features))
    for c in numeric:
        r = stats.linregress(x, features[c])
        rows.append({"metric": c, "slope_per_day": r.slope, "p_value": r.pvalue, "r_squared": r.rvalue ** 2})
    return pd.DataFrame(rows)


def plot_daily_features(features: pd.DataFrame) -> None:
    groups = {"load": ["load_total_kwh", "load_mean_kw", "load_max_kw"], "pv": ["pv_total_kwh", "pv_max_kw", "pv_nonzero_duration_hours"], "netload": ["netload_total_kwh", "netload_max_kw", "netload_min_kw", "netload_negative_intervals"]}
    for label, cols in groups.items():
        fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.5 * len(cols)), sharex=True)
        for ax, col in zip(np.atleast_1d(axes), cols):
            ax.plot(features.date, features[col], marker="o", ms=3); ax.set(ylabel=col); ax.grid(alpha=.25)
        axes[0].set_title(f"2025年1月每日{label}特征变化")
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / f"daily_features_{label}.png", dpi=180); plt.close(fig)


def periodicity_and_similarity(matrices: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, list[dict[str, float]]]]:
    rows, spectral = [], {}
    for label, mat in matrices.items():
        series = mat.ravel()
        values = acf(series, 1008)
        for lag in [144, 288, 1008]: rows.append({"analysis": "acf_all", "variable": label, "lag_steps": lag, "lag_days": lag / 144, "value": values[lag], "detail": "95%% CI approx ±%.4f" % (1.96 / np.sqrt(len(series)))})
        spectral[label] = spectrum(series)
        for rank, item in enumerate(spectral[label], 1): rows.append({"analysis": "periodogram", "variable": label, "rank": rank, "period_hours": item["period_hours"], "value": item["power"], "detail": "linear detrend"})
        adjacent = [safe_corr(mat[i - 1], mat[i]) for i in range(1, len(mat))]
        weekly = [safe_corr(mat[i - 7], mat[i]) for i in range(7, len(mat))]
        for name, vals in [("adjacent_day_correlation", adjacent), ("seven_day_correlation", weekly)]:
            rows.append({"analysis": name, "variable": label, "value": np.nanmean(vals), "std": np.nanstd(vals, ddof=1), "n_pairs": len(vals), "detail": "daily 144-point Pearson correlation"})
        corr = np.corrcoef(mat)
        fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(title=f"2025年1月{label}日曲线两两相关性", xlabel="日期（1月日序）", ylabel="日期（1月日序）")
        fig.colorbar(im, ax=ax, label="Pearson r"); fig.tight_layout(); fig.savefig(FIG / f"{label.lower()}_daily_correlation.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4)); ax.stem(np.arange(1009) / 144, values, basefmt=" ", markerfmt=" ")
        ax.axhline(1.96 / np.sqrt(len(series)), color="r", ls="--"); ax.axhline(-1.96 / np.sqrt(len(series)), color="r", ls="--", label="95%近似界")
        ax.set(title=f"2025年1月{label}自相关函数 ACF", xlabel="滞后 / 天", ylabel="ACF", xlim=(0, 7)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG / f"{label.lower()}_acf.png", dpi=180); plt.close(fig)
        f, p = periodogram(series, detrend="linear"); mask = f > 0
        fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(1 / f[mask] / 6, p[mask], lw=.8); ax.set(xlim=(0, 200), title=f"2025年1月{label}频谱（已去线性趋势）", xlabel="周期 / 小时", ylabel="谱功率"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG / f"{label.lower()}_periodogram.png", dpi=180); plt.close(fig)
    # PV按每日峰值归一化：评估形状，排除日总强度尺度影响。
    pv = matrices["PV"]; norm = np.divide(pv, pv.max(axis=1, keepdims=True), out=np.zeros_like(pv), where=pv.max(axis=1, keepdims=True) > 0)
    # 每日峰值归一化后再计算ACF，主要反映日内形状的重复性而非发电量尺度。
    norm_acf = acf(norm.ravel(), 1008)
    for lag in [144, 288, 1008]:
        rows.append({"analysis": "acf_normalized_daily_shape", "variable": "PV_normalized", "lag_steps": lag, "lag_days": lag / 144, "value": norm_acf[lag], "detail": "PV divided by each day peak; 95%% CI approx ±%.4f" % (1.96 / np.sqrt(norm.size))})
    adj = [safe_corr(norm[i-1], norm[i]) for i in range(1, len(norm))]; week = [safe_corr(norm[i-7], norm[i]) for i in range(7, len(norm))]
    for name, vals in [("adjacent_day_normalized_shape_correlation", adj), ("seven_day_normalized_shape_correlation", week)]: rows.append({"analysis": name, "variable": "PV_normalized", "value": np.nanmean(vals), "std": np.nanstd(vals, ddof=1), "n_pairs": len(vals), "detail": "PV divided by each day peak"})
    corr = np.corrcoef(norm); fig, ax = plt.subplots(figsize=(7,6)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm"); ax.set(title="2025年1月PV归一化日曲线相关性",xlabel="日期（1月日序）",ylabel="日期（1月日序）"); fig.colorbar(im,ax=ax,label="Pearson r"); fig.tight_layout(); fig.savefig(FIG / "pv_normalized_daily_correlation.png",dpi=180); plt.close(fig)
    return pd.DataFrame(rows), spectral


def predict_for_day(history: np.ndarray, target_idx: int, model: str) -> np.ndarray:
    if model == "前一日同期": return history[target_idx - 1]
    if model == "前一周同期": return history[target_idx - 7]
    if model == "过去3日同期均值": return history[target_idx - 3:target_idx].mean(0)
    if model == "过去7日同期均值": return history[target_idx - 7:target_idx].mean(0)
    raise ValueError(model)


def causal_weight(history: np.ndarray, end: int) -> float:
    """只对 end 日之前的可用历史做内部一步预测验证，确定前日的权重。"""
    errors, pairs = [], []
    for d in range(7, end):
        pairs.append((history[d - 1], history[d - 7], history[d]))
    if not pairs: return .5
    grid = np.arange(0, 1.01, .05)
    return float(grid[np.argmin([np.mean([(w*a + (1-w)*b - y) ** 2 for a,b,y in pairs]) for w in grid])])


def metrics(y: np.ndarray, pred: np.ndarray, pv: bool = False) -> dict[str, float]:
    e = pred - y; mae = np.mean(abs(e)); rmse = np.sqrt(np.mean(e**2)); scale = np.mean(abs(y))
    ans = {"MAE": mae, "RMSE": rmse, "nMAE": mae / scale if scale else np.nan, "sMAPE": np.mean(2 * abs(e) / (abs(y) + abs(pred) + 1e-9)), "R2": 1 - np.sum(e**2) / np.sum((y-y.mean())**2)}
    if pv:
        day = y > 0; ans["daylight_MAE"] = np.mean(abs(e[day])) if day.any() else np.nan; ans["daylight_RMSE"] = np.sqrt(np.mean(e[day]**2)) if day.any() else np.nan
    return ans


def backtest(matrices: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_models = ["前一日同期", "前一周同期", "过去3日同期均值", "过去7日同期均值", "加权同期平均"]
    records = []
    for model in base_models:
        pred_by_target: dict[str, list[np.ndarray]] = {"Load": [], "PV": []}; actual_by_target = {"Load": [], "PV": []}
        weights = []
        for d in range(FIRST_TEST_DAY, 31):
            for target in ["Load", "PV"]:
                h = matrices[target]
                if model == "加权同期平均":
                    w = causal_weight(h, d); prediction = w*h[d-1] + (1-w)*h[d-7]
                    if target == "Load": weights.append((d, w))
                else: prediction = predict_for_day(h, d, model)
                pred_by_target[target].append(prediction); actual_by_target[target].append(h[d])
        for target in ["Load", "PV"]:
            records.append({"model": model, "target": target, **metrics(np.concatenate(actual_by_target[target]), np.concatenate(pred_by_target[target]), target == "PV")})
        actual_net = np.concatenate(actual_by_target["Load"]) - np.concatenate(actual_by_target["PV"])
        pred_net = np.concatenate(pred_by_target["Load"]) - np.concatenate(pred_by_target["PV"])
        records.append({"model": model, "target": "NetLoad", **metrics(actual_net, pred_net)})
    weight_df = pd.DataFrame(weights, columns=["test_day_index", "load_weight_previous_day"]) if weights else pd.DataFrame()
    return pd.DataFrame(records), weight_df


def report(df: pd.DataFrame, periodic: pd.DataFrame, spectra: dict[str, list[dict[str, float]]], features: pd.DataFrame, trends: pd.DataFrame, back: pd.DataFrame, weights: pd.DataFrame) -> None:
    def ac(label: str, lag: int) -> float: return periodic.query("analysis == 'acf_all' and variable == @label and lag_steps == @lag").iloc[0].value
    def normalized_ac(lag: int) -> float: return periodic.query("analysis == 'acf_normalized_daily_shape' and lag_steps == @lag").iloc[0].value
    best_load = back[back.target == "Load"].sort_values("RMSE").iloc[0]; best_pv = back[back.target == "PV"].sort_values("RMSE").iloc[0]
    def periods(label: str) -> str: return "、".join(f"{x['period_hours']:.2f}h" for x in spectra[label])
    def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
        cols = list(frame.columns)
        def cell(v: object) -> str:
            return f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v)
        return "\n".join(["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"] + ["|" + "|".join(cell(v) for v in row) + "|" for row in frame.itertuples(index=False, name=None)])
    week_gap = periodic.query("analysis == 'seven_day_correlation'").set_index("variable")["value"].to_dict()
    lines = ["# 问题二：2025年1月预测前置分析", "", "## 数据说明与信息边界", f"", f"读取清洗成品表 `{INPUT.relative_to(ROOT)}`，共 {len(df):,} 行。仅提取 2025-01-01 至 2025-01-31 的 4,464 行（31天×144时段）；未读取原始附件2，也未重新清洗。每个预测日 d 在 00:00 只可使用 d 之前完成的日数据：2月1日只用1月，2月2日可额外用2月1日实测，依此 expanding window。", "", "## 一月日内、周内与短期趋势", "", "Load、PV与NetLoad均绘制了全月/连续7日曲线、热力图、四分位日内曲线和工作日-周末曲线（见 figures）。每日特征在 `daily_features_january.csv`；下表为一月内部线性趋势，31日样本不能外推为年度趋势。", "", markdown_table(trends), "", "## 周期性、频谱与相似度", "", "|变量|ACF(144, 1日)|ACF(288, 2日)|ACF(1008, 7日)|频谱前5主周期|", "|---|---:|---:|---:|---|", *[f"|{v}|{ac(v,144):.4f}|{ac(v,288):.4f}|{ac(v,1008):.4f}|{periods(v)}|" for v in VARIABLES], "", f"归一化PV形状的 ACF(144)={normalized_ac(144):.4f}，ACF(1008)={normalized_ac(1008):.4f}。7日间日曲线平均相关性：Load={week_gap['Load']:.3f}，PV={week_gap['PV']:.3f}，NetLoad={week_gap['NetLoad']:.3f}。PV的归一化日曲线相关性另行统计，以把‘曲线形状’和‘发电强度’分开；全PV ACF会被夜间共同为零抬高，故不能单独作为晴雨日强度稳定的证据。", "", "## 严格滚动回测", "", "初始历史为1月1日至14日；对1月15日至31日逐日一次性预测144点。没有随机划分或shuffle；A-D仅取已发生的同一时刻历史。加权模型的前一日权重在每个预测日用该日之前的内部历史回测网格选择，未使用预测日真实值。NetLoad预测由独立Load预测减独立PV预测获得。", "", markdown_table(back), "", f"Load最佳基线（按RMSE）为 **{best_load.model}**（RMSE={best_load.RMSE:.2f} kW）；PV最佳基线为 **{best_pv.model}**（RMSE={best_pv.RMSE:.2f} kW）。", "", "## 对问题二的建议", "", "存在足够证据将日周期作为2月初预测的主基线；是否增加周周期以各目标的严格回测表现决定，而非只根据ACF。建议分别预测Load与PV并相减得到NetLoad：这保留了PV不确定性，便于后续采用Load较高分位、PV较低分位或NetLoad上分位的保守情景。因紧急购电价为常规价5倍，优化阶段应评估这种偏保守（特别是净负荷上分位）方案的成本-缺电风险权衡。", "", "2月1日只能基于1月历史采用本报告最佳基线；从2月2日起每日纳入前一天实测并扩展训练。严禁以全年实测数据训练后反向预测2月至12月。", "", "## 局限性", "", "仅31天、17个测试日的回测不足以刻画季节变化和极端天气；PV夜间零值会夸大某些统计周期指标。后续应继续执行逐日、预报日前可得的数据回测，并在有可用气象预报时纳入气象特征。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning); OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); set_font()
    data = load_data(); jan = data.loc[(data.date >= "2025-01-01") & (data.date <= "2025-01-31")].copy()
    assert len(jan) == 31*N_PER_DAY and jan.groupby("date").size().eq(N_PER_DAY).all(), "一月日期/时段不完整"
    matrices = {name: jan_matrix(jan, col) for name, col in VARIABLES.items()}
    for label, col in VARIABLES.items(): plot_time_series(jan, label, col); plot_heatmap(matrices[label], label); plot_profiles(jan, matrices[label], label)
    features = daily_features(jan); trends = trend_stats(features); features.to_csv(OUT / "daily_features_january.csv", index=False, encoding="utf-8-sig"); trends.to_csv(OUT / "daily_feature_trends.csv", index=False, encoding="utf-8-sig"); plot_daily_features(features)
    periodic, spectra = periodicity_and_similarity(matrices); periodic.to_csv(OUT / "periodicity_statistics.csv", index=False, encoding="utf-8-sig")
    back, weights = backtest(matrices); back.to_csv(OUT / "backtest_metrics.csv", index=False, encoding="utf-8-sig"); weights.to_csv(OUT / "causal_weight_history.csv", index=False, encoding="utf-8-sig")
    report(data, periodic, spectra, features, trends, back, weights)
    print(f"读取: {INPUT}"); print(f"输出: {OUT}"); print(back.to_string(index=False))


if __name__ == "__main__": main()
