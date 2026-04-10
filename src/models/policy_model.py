from typing import Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


def resolve_dtype(use_fp16: bool) -> Optional[torch.dtype]:
    if use_fp16:
        return torch.float16
    return None


def load_policy_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    use_fp16: bool = False,
    device_map: Optional[str] = None,
) -> PreTrainedModel:
    dtype = resolve_dtype(use_fp16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map=device_map,
    )
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id
    return model
