from typing import Iterable, List, Sequence

import torch


def generate_responses(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    device = next(model.parameters()).device
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    generations: List[str] = []
    for index, output in enumerate(outputs):
        new_tokens = output[int(prompt_lengths[index]) :]
        generations.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return generations


def batch_prompts(records: Iterable[dict], prompt_template: str) -> List[str]:
    return [
        prompt_template.format(
            instruction=record.get("instruction", "").strip(),
            input=record.get("input", "").strip(),
        )
        for record in records
    ]

