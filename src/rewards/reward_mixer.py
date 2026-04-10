from typing import Dict


def mix_rewards(reward_dict: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    for name, value in reward_dict.items():
        score += value * weights.get(name, 1.0)
    return score

