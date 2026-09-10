"""问题一时间映射回归测试。

先以官方模板的左端点规则约束旧实现；修复前首项会被错误映射为00:00–00:10，
因此本测试必须失败。修复后同一测试应通过。
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solve_question1 import INPUT, make_periods


class TimeMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = make_periods(pd.read_csv(INPUT))

    def test_template_left_endpoint_mapping(self):
        # 官方模板规定00:10为区间左端点，故并非00:00–00:10。
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "00:10:00"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("00:10", "00:20"))
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "00:20:00"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("00:20", "00:30"))
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "23:50"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("23:50", "24:00"))

    def test_next_day_label_is_normalized_to_calendar_start(self):
        # 0:00+1必须仅出现一次，并循环至代表日00:00–00:10。
        row = self.schedule.loc[self.schedule["time_label"].astype(str) == "0:00+1"].iloc[0]
        self.assertEqual((row.period_start_text, row.period_end_text), ("00:00", "00:10"))
        self.assertEqual(self.schedule["period_key"].nunique(), 144)


if __name__ == "__main__":
    unittest.main()
