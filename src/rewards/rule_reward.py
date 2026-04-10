from collections import Counter
from typing import Iterable


def repetition_penalty(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return -float(repeated) / max(len(tokens), 1)


def length_penalty(text: str, max_length: int = 256) -> float:
    overflow = max(0, len(text) - max_length)
    return -overflow / max(max_length, 1)


def keyword_coverage_reward(text: str, keywords: Iterable[str]) -> float:
    keywords = [keyword for keyword in keywords if keyword]
    if not keywords:
        return 0.0
    covered = sum(1 for keyword in keywords if keyword in text)
    return covered / len(keywords)

