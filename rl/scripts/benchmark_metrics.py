"""固定场景逐回合统计；独立于仿真和GPU，供评测与回归检查共用。"""

import numpy as np


def benchmark_attempt_metrics(rows):
    """逐回合统计完整越障；阈值包含机器人越过最后障碍后的落地点。"""
    required = {
        "flat_reference": 3.0, "step_up_06cm": 3.2, "step_down_08cm": 2.0,
        "low_box_12cm": 2.3, "bypass_wall_45cm": 2.8, "slalom_45cm": 4.0,
    }
    attempts = {}
    for row in rows:
        attempts.setdefault((row["env_id"], row["episode_id"]), []).append(row)
    completed = clean = censored = 0
    first_attempts = {}
    for key, trajectory in attempts.items():
        prefix = []
        success = False
        for row in trajectory:
            prefix.append(row)
            if row["terminated"]:
                break
            distance = required[row["terrain_name"]]
            stable = row["root_clearance_m"] >= 0.30 and row["tilt_deg"] <= 25.0
            if row["forward_progress_m"] >= distance and abs(row["lateral_offset_m"]) < 0.7 and stable:
                success = True
                break
        completed += int(success)
        low_contact = success and max(r["nonfoot_force_peak_n"] for r in prefix) < 20.0
        clean += int(low_contact)
        first_attempts.setdefault(key[0], (success, low_contact))
        censored += int(not success and not (trajectory[-1]["terminated"] or trajectory[-1]["timeout"]))
    count = len(attempts)
    return {
        "attempts": count, "completed_attempts": completed,
        "completion_fraction": completed / max(count, 1),
        "low_contact_completed_attempts": clean,
        "low_contact_completion_fraction": clean / max(count, 1),
        "unfinished_at_test_horizon": censored,
        "nonfoot_contact_frame_fraction": float(np.mean([r["nonfoot_force_peak_n"] > 20 for r in rows])),
        "wheel_force_p95_n": float(np.percentile([r["wheel_force_peak_n"] for r in rows], 95)),
        "first_attempts": len(first_attempts),
        "first_completed_attempts": sum(int(value[0]) for value in first_attempts.values()),
        "first_low_contact_completed_attempts": sum(int(value[1]) for value in first_attempts.values()),
    }
