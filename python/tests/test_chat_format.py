import datetime
from pathlib import Path

import pytest

from m2w.frontends.hf_llm import chat_format as C

FIXTURES = Path(__file__).parent / "fixtures"


def test_aion_format():
    fmt = C.derive_chat_format((FIXTURES / "aion_chat_template.jinja").read_text())
    assert fmt["first_prefix"].startswith("<|system|>You are Aion Instruct")
    assert "<<date:%Y-%m-%d>>" in fmt["first_prefix"] and fmt["first_prefix"].endswith("<|end|><|user|>")
    assert fmt["first_prefix_with_system"].endswith("\n\n<<system>><|end|><|user|>")
    assert fmt["user_suffix_generation_prompt"] == "<|end|><|assistant|>" == fmt["user_suffix_assistant_prefix"]
    assert fmt["assistant_suffix"] == "<|end|>" and fmt["next_user_prefix"] == "<|user|>"
    text = C.fill_placeholders(C.compose(fmt, [("Hi", None)]), datetime.date(2026, 11, 24))
    assert text.endswith("Current date: 2026-11-24<|end|><|user|>Hi<|end|><|assistant|>")


def test_qwen3_format_with_thinking_disabled_matches_the_hard_coded_prompt_of_the_page():
    fmt = C.derive_chat_format((FIXTURES / "qwen3_chat_template.jinja").read_text(), {"enable_thinking": False})
    prompt = C.compose(fmt, [("Hello", None)])
    assert prompt == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # exactly what chat.js builds today
    assert fmt["user_suffix_assistant_prefix"] == "<|im_end|>\n<|im_start|>assistant\n"                  # history turns carry no think block
    assert C.compose(fmt, [("Hello", "Hi"), ("Bye", None)]).endswith("Hi<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_gemma4_format_starts_with_the_bos_token():
    fmt = C.derive_chat_format((FIXTURES / "gemma4_chat_template.jinja").read_text(), {}, "<bos>")
    assert C.compose(fmt, [("Hello", None)]) == "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
    assert C.compose(fmt, [("Hello", "Hi"), ("Bye", None)]) == "<bos><|turn>user\nHello<turn|>\n<|turn>model\nHi<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n"
    assert fmt["first_prefix_with_system"] == "<bos><|turn>system\n<<system>><turn|>\n<|turn>user\n"


def test_unrenderable_template_is_reported():
    with pytest.raises(C.ChatTemplateError):
        C.derive_chat_format("{{ messages | length }}")


def test_end_of_turn_token_is_the_special_token_the_assistant_suffix_starts_with():
    tokenizer = {"added_tokens": [{"id": 7, "content": "<|im_end|>", "special": True}, {"id": 8, "content": "<|im", "special": True}, {"id": 9, "content": "<|im_end|>\n", "special": False},
                                  {"id": 10, "content": "<|endoftext|>", "special": True}]}
    assert C.end_of_turn_token_id({"assistant_suffix": "<|im_end|>\n"}, tokenizer) == 7   # the longest special token: a plain (non-special) token and a prefix of it do not count
    assert C.end_of_turn_token_id({"assistant_suffix": "\n"}, tokenizer) is None
