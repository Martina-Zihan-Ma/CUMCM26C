# Question 2 dynamic-battery run report

## Reproducibility

- Git branch and commit: `main`, `2fb809b` (`origin/main` matched at execution).
- Historical commits verified: `ad678ad` (dynamic battery), `c43cb98` (Excel revision), and `2fb809b` (merge).
- Executed model programs: `python3 src/q2_forecast.py`; `Q2_MAX_DAYS=3 python3 src/Q2.py`; and `python3 src/Q2.py`.
- The forecast run covers 364 days (2025-01-02--2025-12-31; 52,416 rows). The formal dynamic optimization covers 334 days (2025-02-01--2025-12-31; 48,096 10-minute intervals).
- The output set present before this run was preserved under `outputs/question2_archive_before_dynamic_battery_20260911_192326/`.

## Model and information boundary

- The sole optimization model was `src/Q2.py`; no legacy final/optimize script was executed.
- `SAFETY_QUANTILE = 0.80`, 20 residual-bootstrap scenarios, and no terminal reserve/CVaR are used.
- January 1 is recorded as 6,000 kWh at both endpoints. The February 1 realized initial SOC is 1,851.843470 kWh after causal January warm-up.
- Forecast `history_end_date` is strictly earlier than every target date. In `execute_causal_dispatch`, only current interval load, PV, fixed grid plan, and current SOC are used; actual observations enter residual history only after the day ends. No future-information leakage was detected.

## Cost and operating results

| Metric | Value |
|---|---:|
| Planned purchase | 20,803,919.119 kWh |
| Planned cost | 12,865,023.718 yuan |
| Emergency purchase | 197,058.168 kWh |
| Emergency cost | 1,145,438.228 yuan |
| Dynamic-battery total cost | 14,010,461.945 yuan |
| Emergency days / intervals | 158 / 1,406 |
| Spill | 1,884,625.048 kWh |
| No-storage total cost | 18,940,040.160 yuan |
| Savings | 4,929,578.214 yuan (26.0273%) |

## Independent checks

- Max actual energy-balance error: `4.547e-13` kWh.
- Max SOC-transition error: `4.547e-12` kWh.
- Max interday SOC discontinuity: `0` kWh.
- Actual SOC range: 1,200--10,800 kWh; simultaneous charge/discharge intervals: 0.
- Plan, dispatch, and emergency outputs each have 48,096 rows; daily summary has 334 rows.

## Excel mapping

The official plan sheet uses cyclic headers (`t=1,...,143,0`), while the raw `result2_final.xlsx` wrote direct `t=0,...,143` order. This is an Excel-display-only issue; it does not change any optimization result. The raw workbook is retained, and `result2_final_checked.xlsx` corrects the plan-column order plus compressed emergency-period labels.

## Figures

The January periodicity and forecast-vs-actual figures were rebuilt from the fresh forecast output; their metrics are unchanged, so they remain suitable. The former SOC-reserve figure is obsolete and should not be used. New dynamic-dispatch, cumulative-cost, and daily-cost PNG/PDF figures are in `outputs/question2/figures/`; all five current figures passed PDF glyph and collision audits.
