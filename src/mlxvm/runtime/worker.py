"""Isolated MLX generation worker.

MLX initializes Metal during import, so running it in a child process keeps the
manager and its registry recoverable even if Metal initialization fails.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--settings", default="{}")
    parser.add_argument("--prompt-cache")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.settings)

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import load_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    if settings.get("seed") is not None:
        mx.random.seed(settings["seed"])
    prompt_cache = None
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    cache_metadata = None
    if args.prompt_cache:
        prompt_cache, cache_metadata = load_prompt_cache(args.prompt_cache, return_metadata=True)
        tokenizer_config.update(json.loads(cache_metadata["tokenizer_config"]))
        tokenizer_config["trust_remote_code"] = True if args.trust_remote_code else None
        if cache_metadata.get("model") != args.model:
            raise ValueError("prompt cache was created for a different model")
    model, tokenizer = load(
        args.model,
        tokenizer_config=tokenizer_config,
    )
    if cache_metadata is not None:
        tokenizer.chat_template = json.loads(cache_metadata["chat_template"])
    messages = []
    if settings.get("system_prompt"):
        messages.append({"role": "system", "content": settings["system_prompt"]})
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    messages.append({"role": "user", "content": prompt})
    if getattr(tokenizer, "chat_template", None):
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt_tokens = prompt
    sampler = make_sampler(
        temp=settings.get("temperature", 0.0),
        top_p=settings.get("top_p", 1.0),
        min_p=settings.get("min_p", 0.0),
        top_k=settings.get("top_k", 0),
    )
    kwargs = {"sampler": sampler, "prompt_cache": prompt_cache}
    if settings.get("max_kv_size") is not None:
        kwargs["max_kv_size"] = settings["max_kv_size"]
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=settings.get("max_tokens", 256),
        **kwargs,
    ):
        print(response.text, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
