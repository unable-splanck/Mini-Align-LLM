from typing import Iterable


def lexical_overlap_score(prediction: str, references: Iterable[str]) -> float:
    pred_tokens = set(prediction.split())
    if not pred_tokens:
        return 0.0
    ref_tokens = set()
    for reference in references:
        ref_tokens.update(reference.split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)

