"""问题一：附件右端点建模与官方模板输出映射的回归测试。"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solve_question1 import INPUT, make_periods, template_period_key


class TimeMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = make_periods(pd.read_csv(INPUT))

    def test_attachment_right_endpoint_mapping_preserves_source_order(self):
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "00:10:00"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("00:00", "00:10"))
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "00:20:00"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("00:10", "00:20"))
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "23:50"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("23:40", "23:50"))
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "0:00+1"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("23:50", "24:00"))
        self.assertEqual(self.schedule["source_time_index"].tolist(), list(range(144)))

    def test_official_template_keys_cycle_to_same_calendar_solution(self):
        # 模板0:10--0:20取附件标签00:20的数据；最后一行循环取标签00:10的数据。
        expected = {
            "0:10-0:20": ("00:10-00:20", "00:20:00"),
            "23:50-0:00+1": ("23:50-24:00", "0:00+1"),
            "0:00+1-0:10+1": ("00:00-00:10", "00:10:00"),
        }
        for template_label, (key, source_label) in expected.items():
            self.assertEqual(template_period_key(template_label), key)
            row = self.schedule.loc[self.schedule["period_key"] == key].iloc[0]
            self.assertEqual(str(row.time_label), source_label)
        self.assertEqual(self.schedule["period_key"].nunique(), 144)


if __name__ == "__main__":
    unittest.main()
