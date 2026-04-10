import argparse

import torch

from src.eval.generate import generate_responses
from src.models.policy_model import load_policy_model
from src.models.tokenizer import load_tokenizer
from src.utils.logger import get_logger


DEFAULT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a base or fine-tuned causal LM.")
    parser.add_argument("--model-path", default="distilgpt2")
    parser.add_argument("--prompt", required=True, help="Instruction text for the model.")
    parser.add_argument("--input-text", default="", help="Optional extra input section.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    logger = get_logger("infer")
    tokenizer = load_tokenizer(args.model_path, trust_remote_code=args.trust_remote_code)
    model = load_policy_model(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fp16=args.use_fp16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompt = DEFAULT_TEMPLATE.format(instruction=args.prompt.strip(), input=args.input_text.strip())
    output = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )[0]
    logger.info("Inference complete.")
    print(output)


if __name__ == "__main__":
    main()
