# Q2 dynamic battery validation

## Result

- PASS — forecast_364_days
- PASS — forecast_no_future_information
- PASS — formal_result_shapes
- PASS — energy_balance: `4.547e-13`
- PASS — soc_transition: `4.547e-12`
- PASS — soc_continuity: `0.000e+00`
- PASS — soc_bounds
- PASS — charge_discharge_limits
- PASS — no_simultaneous_charge_discharge: `0.000e+00`
- PASS — causal_surplus_deficit_rule
- PASS — cost_recalculation
- PASS — excel_cyclic_headers_detected
- PASS — excel_checked_mapping

## Excel time mapping

The original `result2_final.xlsx` used direct t=0..143 placement: **detected**.
The official headers are cyclic (t=1..143,0); `result2_final_checked.xlsx` preserves the raw workbook and corrects only Sheet 1 placement and Sheet 3 emergency-period labels.

No terminal reserve or CVaR metric is evaluated. Causality is evidenced by strictly prior forecast history and by the recorded one-step dispatch balance.
