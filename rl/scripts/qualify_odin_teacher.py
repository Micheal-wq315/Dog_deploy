"""检查多种子首次尝试的教师通过率，阻止将明显失败的教师继续蒸馏。"""

import argparse
import json
from pathlib import Path


CORE_SCENARIOS = ("flat_reference", "step_up_06cm", "step_down_08cm", "low_box_12cm")


def qualify(directories, min_completion=0.8, min_low_contact=0.6):
    results = {}
    for directory in directories:
        if not json.loads((Path(directory) / "config.json").read_text())["completed"]:
            raise ValueError(f"评测未完整结束: {directory}")
    reports = [json.loads((Path(directory) / "terrain_summary.json").read_text()) for directory in directories]
    for scenario in CORE_SCENARIOS:
        total = completed = low_contact = 0
        for report in reports:
            row = report[scenario]
            total += row["first_attempts"]
            completed += row["first_completed_attempts"]
            low_contact += row["first_low_contact_completed_attempts"]
        rate = completed / max(total, 1)
        low_rate = low_contact / max(total, 1)
        results[scenario] = dict(attempts=total, completed=completed, low_contact_completed=low_contact,
                                 completion_fraction=rate, low_contact_fraction=low_rate,
                                 passed=total > 0 and rate >= min_completion and low_rate >= min_low_contact)
    return {"passed": all(row["passed"] for row in results.values()), "scenarios": results,
            "min_completion": min_completion, "min_low_contact": min_low_contact}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", help="含terrain_summary.json的评测目录。")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.directories)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit("教师专项通过率未达标，停止学生训练；请先检查教师回放和课程曲线。")
