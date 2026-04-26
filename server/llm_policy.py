# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Policy inference for video demo: same prompt contract as the Colab notebook
# (SYSTEM_PROMPT + observation_to_prompt) → parse_action.

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Tuple

from models import FinancialAction, FinancialObservation
from inference import SYSTEM_PROMPT, observation_to_prompt, parse_action, format_action

_local_lock = threading.Lock()
_local_model: Any = None
_local_tokenizer: Any = None


def _llm_backends() -> List[str]:
    b: List[str] = []
    if os.environ.get("LLM_LOCAL_ADAPTER", "").strip():
        b.append("local")
    if os.environ.get("LLM_BASE_URL", "").strip():
        b.append("api")
    return b


def llm_status() -> dict[str, Any]:
    backs = _llm_backends()
    return {
        "ready": bool(backs),
        "backends": backs,
        "local_adapter": bool(os.environ.get("LLM_LOCAL_ADAPTER", "").strip()),
        "api": bool(os.environ.get("LLM_BASE_URL", "").strip()),
    }


def _generate_text_api(*, system: str, user: str) -> str:
    from openai import OpenAI  # type: ignore

    base = os.environ["LLM_BASE_URL"].strip()
    key = os.environ.get("LLM_API_KEY", "").strip() or "unused"
    model = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct:hf-inference")
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "128"))
    temp = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
    client = OpenAI(base_url=base, api_key=key)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temp,
    )
    if not r.choices:
        return ""
    c = r.choices[0].message.content
    return (c or "").strip()


def _load_local() -> None:
    global _local_model, _local_tokenizer
    with _local_lock:
        if _local_model is not None:
            return
        path = os.environ["LLM_LOCAL_ADAPTER"].strip()
        if not path:
            raise RuntimeError("LLM_LOCAL_ADAPTER not set")
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Local adapter path set but 'unsloth' not installed. "
                "pip install unsloth (GPU environment)."
            ) from e
        base = os.environ.get(
            "LLM_BASE_ID",
            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        )
        # Prefer loading the Colab output folder if it is a full Unsloth save
        try:
            _local_model, _local_tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=int(os.environ.get("LLM_MAX_SEQ", "2048")),
                load_in_4bit=True,
                dtype=None,
            )
        except Exception:
            m, t = FastLanguageModel.from_pretrained(
                model_name=base,
                max_seq_length=int(os.environ.get("LLM_MAX_SEQ", "2048")),
                load_in_4bit=True,
                dtype=None,
            )
            from peft import PeftModel  # type: ignore

            _local_model = PeftModel.from_pretrained(m, path)
            _local_tokenizer = t
        FastLanguageModel.for_inference(_local_model)  # type: ignore


def _generate_text_local(*, system: str, user: str) -> str:
    import torch
    from unsloth import FastLanguageModel  # type: ignore  # noqa: F401

    _load_local()
    assert _local_tokenizer is not None
    assert _local_model is not None
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    toks = _local_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_local_model.device)
    with torch.no_grad():
        out = _local_model.generate(
            input_ids=toks,
            max_new_tokens=int(os.environ.get("LLM_MAX_TOKENS", "128")),
            do_sample=True,
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
            pad_token_id=_local_tokenizer.eos_token_id,
        )
    gen = out[0][toks.shape[-1] :]
    return _local_tokenizer.decode(gen, skip_special_tokens=True).strip()


def _choose_backend() -> str:
    has_local = bool(os.environ.get("LLM_LOCAL_ADAPTER", "").strip())
    has_api = bool(os.environ.get("LLM_BASE_URL", "").strip())
    prefer_api = os.environ.get("LLM_PREFER_API", "").lower() in ("1", "true", "yes")
    if not has_local and not has_api:
        raise RuntimeError(
            "No LLM configured. Set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL for an API "
            "(Hugging Face router, vLLM, OpenAI, …), and/or LLM_LOCAL_ADAPTER=path to "
            "Unsloth/PEFT output from Colab. Optional: LLM_PREFER_API=1 to prefer API "
            "when both are set."
        )
    if has_local and not has_api:
        return "local"
    if has_api and not has_local:
        return "api"
    return "api" if prefer_api else "local"


def _generate_text(*, system: str, user: str) -> str:
    b = _choose_backend()
    if b == "api":
        return _generate_text_api(system=system, user=user)
    return _generate_text_local(system=system, user=user)


def generate_trained_action(obs: FinancialObservation) -> Tuple[FinancialAction, str, str]:
    """
    Returns: (action, raw_model_text, formatted_action_str)
    """
    if getattr(obs, "done", False):
        raise ValueError("Episode is done; cannot select an action")
    user = observation_to_prompt(obs)
    raw = _generate_text(system=SYSTEM_PROMPT, user=user)
    if not raw:
        raise ValueError("Empty model response")
    action = parse_action(raw, obs)
    return action, raw, format_action(action)
