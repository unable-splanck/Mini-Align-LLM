from typing import Iterable


def format_accuracy(texts: Iterable[str], required_prefix: str = "###") -> float:
    texts = list(texts)
    if not texts:
        return 0.0
    matched = sum(text.strip().startswith(required_prefix) for text in texts)
    return matched / len(texts)


def repetition_rate(text: str) -> float:
    tokens = text.split()
    if len(tokens) <= 1:
        return 0.0
    unique_tokens = len(set(tokens))
    return 1.0 - (unique_tokens / len(tokens))


def keyword_coverage(text: str, keywords: Iterable[str]) -> float:
    keywords = [keyword for keyword in keywords if keyword]
    if not keywords:
        return 0.0
    covered = sum(1 for keyword in keywords if keyword in text)
    return covered / len(keywords)

