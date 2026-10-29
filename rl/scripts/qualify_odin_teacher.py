"""检查多种子首次尝试的教师通过率，阻止将明显失败的教师继续蒸馏。"""

import argparse
import json
from pathlib import Path


CORE_SCENARIOS = ("flat_reference", "step_up_06cm", "step_down_08cm", "low_box_12cm")


def qualify(directories, min_completion=0.8, min_low_contact=0.6, scenarios=CORE_SCENARIOS,
            min_flat_vx=None):
    results = {}
    for directory in directories:
        if not json.loads((Path(directory) / "config.json").read_text())["completed"]:
            raise ValueError(f"评测未完整结束: {directory}")
    reports = [json.loads((Path(directory) / "terrain_summary.json").read_text()) for directory in directories]
    if not scenarios or any(name not in CORE_SCENARIOS for name in scenarios):
        raise ValueError("验收场景必须是非空的核心场景子集。")
    for scenario in scenarios:
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
        if scenario == "flat_reference" and min_flat_vx is not None:
            # 固定基准指令为0.35m/s；仅缓慢挪到3m不代表速度跟踪合格。
            speeds = []
            for directory in directories:
                summary = json.loads((Path(directory) / "summary.json").read_text())
                speeds.append(next(row["vx"] for row in summary if row["case"] == scenario))
            results[scenario]["mean_vx"] = sum(speeds) / len(speeds)
            results[scenario]["min_vx"] = min_flat_vx
            results[scenario]["passed"] &= results[scenario]["mean_vx"] >= min_flat_vx
    return {"passed": all(row["passed"] for row in results.values()), "scenarios": results,
            "min_completion": min_completion, "min_low_contact": min_low_contact}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", help="含terrain_summary.json的评测目录。")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=CORE_SCENARIOS, default=CORE_SCENARIOS,
                        help="阶段检查场景；最终验收默认包含全部核心场景。")
    parser.add_argument("--min-flat-vx", type=float, default=0.25,
                        help="平地0.35m/s指令下的最低平均实测前进速度。")
    args = parser.parse_args()
    result = qualify(args.directories, scenarios=args.scenarios, min_flat_vx=args.min_flat_vx)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit("教师专项通过率未达标，停止学生训练；请先检查教师回放和课程曲线。")
