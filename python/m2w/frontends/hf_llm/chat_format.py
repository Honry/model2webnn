"""HF chat template (jinja) -> chat_format.json: the few string pieces the browser runtime needs to build prompts, so the runtime never has to run jinja.

The template is rendered with sentinel messages and the pieces are cut out of the results:
    turn 1 :  first_prefix [+ system text] + <user text> + user_suffix_generation_prompt          (everything before the model starts answering)
    turn n :  <previous answer> + assistant_suffix + next_user_prefix + <user text> + user_suffix_generation_prompt
    (rendering a FINISHED turn again uses user_suffix_assistant_prefix instead of the generation prompt: they differ when the template adds e.g. an empty <think> block)
The composition is verified against the real jinja rendering of a multi-turn conversation; if the template has another shape, an error is raised (the caller
may continue without chat format). Dynamic parts: strftime_now(fmt) becomes the placeholder "<<date:fmt>>", the system text "<<system>>".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

FORMAT = "m2w-chat"
VERSION = 0
USER, USER2, ASSISTANT, SYSTEM = "@@USER1@@", "@@USER2@@", "@@ASSISTANT@@", "@@SYSTEM@@"


class ChatTemplateError(ValueError):
    pass


class _GenerationTag(Extension):
    """HF's {% generation %} ... {% endgeneration %} (training masks): renders its body."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def load_template(model_dir) -> str:
    d = Path(model_dir)
    if (d / "chat_template.jinja").exists():
        return (d / "chat_template.jinja").read_text(encoding="utf-8")
    if (d / "chat_template.json").exists():
        value = json.loads((d / "chat_template.json").read_text(encoding="utf-8")).get("chat_template")
    else:
        value = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template") if (d / "tokenizer_config.json").exists() else None
    if isinstance(value, list):
        value = next((t["template"] for t in value if t.get("name") == "default"), value[0]["template"])
    if not value:
        raise ChatTemplateError(f"no chat template found in {d}")
    return value


def render(template: str, messages: list, add_generation_prompt: bool, bos_token: str = "", **kwargs) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_GenerationTag, "jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)

    def raise_exception(message):
        raise ChatTemplateError(message)

    env.globals.update(raise_exception=raise_exception, strftime_now=lambda fmt: f"<<date:{fmt}>>")
    return env.from_string(template).render(messages=messages, add_generation_prompt=add_generation_prompt, bos_token=bos_token, eos_token="", **kwargs)


def _between(text: str, before: str, after: Optional[str]) -> str:
    start = text.index(before) + len(before)
    return text[start:] if after is None else text[start:text.index(after, start)]


def derive_chat_format(template: str, template_kwargs: Optional[dict] = None, bos_token: str = "") -> dict:
    """bos_token: the text of the BOS token for templates that start with it (Gemma): it becomes part of first_prefix and the runtime tokenizes it as the special token."""
    kw = dict(template_kwargs or {}, bos_token=bos_token)
    user = lambda c: {"role": "user", "content": c}  # noqa: E731
    assistant = lambda c: {"role": "assistant", "content": c}  # noqa: E731
    r1 = render(template, [user(USER)], True, **kw)
    for marker in (USER,):
        if marker not in r1:
            raise ChatTemplateError("the template does not render the user message")
    first_prefix = r1[: r1.index(USER)]
    suffix_generation = r1[r1.index(USER) + len(USER):]
    system_text = render(template, [{"role": "system", "content": SYSTEM}, user(USER)], True, **kw)
    first_prefix_with_system = system_text[: system_text.index(USER)].replace(SYSTEM, "<<system>>")
    if "<<system>>" not in first_prefix_with_system:
        raise ChatTemplateError("the template does not render a system message")
    after_assistant = render(template, [user(USER), assistant(ASSISTANT)], False, **kw)
    assistant_suffix = after_assistant[after_assistant.index(ASSISTANT) + len(ASSISTANT):]
    r2 = render(template, [user(USER), assistant(ASSISTANT), user(USER2)], True, **kw)
    between_user_and_assistant = _between(r2, USER, ASSISTANT)
    between_assistant_and_user = _between(r2, ASSISTANT, USER2)
    if not between_assistant_and_user.startswith(assistant_suffix):
        raise ChatTemplateError("unsupported template shape: text after an assistant message differs between the last and an earlier turn")
    next_user_prefix = between_assistant_and_user[len(assistant_suffix):]
    fmt = {
        "format": FORMAT, "version": VERSION, "template_kwargs": {k: v for k, v in kw.items() if k != "bos_token"},
        "first_prefix": first_prefix, "first_prefix_with_system": first_prefix_with_system, "user_suffix_generation_prompt": suffix_generation,
        "assistant_suffix": assistant_suffix, "next_user_prefix": next_user_prefix,
        # history form of the text between a user message and the assistant's answer (differs from the generation prompt when the template adds e.g. an empty <think> block)
        "user_suffix_assistant_prefix": between_user_and_assistant,
    }
    verify_chat_format(template, fmt, bos_token)
    return fmt


def compose(fmt: dict, turns: list, system: Optional[str] = None) -> str:
    """turns: [(user, assistant_or_None), ...]; the last turn has assistant None (the model answers next)."""
    out = fmt["first_prefix_with_system"].replace("<<system>>", system) if system else fmt["first_prefix"]
    for i, (u, a) in enumerate(turns):
        out += (fmt["next_user_prefix"] if i else "") + u
        if a is None:
            out += fmt["user_suffix_generation_prompt"]          # the model answers next
        else:
            out += fmt["user_suffix_assistant_prefix"] + a + fmt["assistant_suffix"]
    return out


def verify_chat_format(template: str, fmt: dict, bos_token: str = "") -> None:
    kw = dict(fmt["template_kwargs"], bos_token=bos_token)
    convo = [("Hello there", "Hi! How can I help?"), ("Tell me a joke", None)]
    for system in (None, "Be brief."):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "Tell me a joke"}]
        expected = render(template, messages, True, **kw)
        got = compose(fmt, convo, system)
        if got != expected:
            raise ChatTemplateError(f"the derived chat format does not reproduce the template (system={system!r}):\n  expected {expected!r}\n  got      {got!r}")


def fill_placeholders(text: str, now) -> str:
    """Replace <<date:fmt>> by strftime(fmt) of ``now`` (what the runtime does)."""
    import re
    return re.sub(r"<<date:([^>]*)>>", lambda m: now.strftime(m.group(1)), text)
