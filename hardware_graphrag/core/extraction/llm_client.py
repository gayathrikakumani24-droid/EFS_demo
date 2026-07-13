"""
Thin, pluggable LLM client abstraction.

Supports OpenAI, Groq (OpenAI-compatible API), and Google Gemini, chosen
via config.CONFIG.llm.provider. If no API key is configured, or the
provider's SDK isn't installed, `LLMClient.complete_json()` returns None
and callers (entity/relationship extractors) fall back to deterministic
regex/keyword-based heuristics — so the whole pipeline still runs end
to end without any API key, just with lower extraction recall.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from config import CONFIG
from utils.logger import get_logger

logger = get_logger("extraction.llm_client")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    return text


def _extract_json_span(text: str) -> str:
    """Trim any leading/trailing prose the model added around the JSON object,
    keeping everything from the first '{' to the last '}'."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort repair for JSON truncated mid-generation (e.g. the response
    hit the max_tokens limit while writing a string value). Walks the text
    tracking bracket/string nesting, rolls back to the last point that was
    safely outside an open string or bracket, strips any trailing dangling
    comma, then closes any still-open braces/brackets and retries parsing.
    This intentionally sacrifices the last (incomplete) entry rather than
    losing the whole batch of entities/relationships already emitted.
    """
    stack: list = []
    in_string = False
    escape = False
    last_safe_pos = 0
    n = len(text)

    def _next_meaningful_char(pos: int) -> Optional[str]:
        j = pos
        while j < n and text[j] in " \t\r\n":
            j += 1
        return text[j] if j < n else None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                # Only treat this as a safe cut point if the string just closed was a
                # *value* (next meaningful char is ',' '}' ']' or end-of-text), not a
                # *key* (next meaningful char is ':') — cutting after a bare key would
                # leave a dangling "key" with no value, which is still invalid JSON.
                nxt = _next_meaningful_char(i + 1)
                if nxt in (",", "}", "]", None):
                    last_safe_pos = i + 1
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                last_safe_pos = i + 1

    truncated = text[:last_safe_pos] if last_safe_pos else text
    truncated = re.sub(r",\s*$", "", truncated.rstrip())

    # recompute the bracket stack up to the truncation point to know what to close
    stack = []
    in_string = False
    escape = False
    for ch in truncated:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

    closer = {"{": "}", "[": "]"}
    candidate = truncated + "".join(closer[b] for b in reversed(stack))

    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        return None


def _parse_json_response(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse (possibly messy) LLM output into a dict. Returns (result, error_message)."""
    cleaned = _extract_json_span(_strip_code_fences(raw))
    try:
        return json.loads(cleaned, strict=False), None
    except json.JSONDecodeError as e:
        repaired = _repair_truncated_json(cleaned)
        if repaired is not None:
            return repaired, f"recovered via truncation-repair (original error: {e})"
        return None, str(e)


class LLMClient:
    """Wraps whichever provider is configured behind one simple call."""

    def __init__(self):
        self.cfg = CONFIG.llm
        self._client = None
        self._provider = self.cfg.provider.lower()
        self._init_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    def _init_client(self) -> None:
        if not self.cfg.api_key or self._provider == "none":
            logger.info("No LLM API key configured - extraction will use heuristic fallback.")
            return

        try:
            if self._provider in ("openai", "groq"):
                from openai import OpenAI
                base_url = self.cfg.base_url or (
                    "https://api.groq.com/openai/v1" if self._provider == "groq" else None
                )
                self._client = OpenAI(api_key=self.cfg.api_key, base_url=base_url)
            elif self._provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.cfg.api_key)
                self._client = genai.GenerativeModel(self.cfg.model)
            else:
                logger.warning(f"Unknown LLM provider '{self._provider}'.")
        except ImportError as e:
            logger.warning(f"SDK for provider '{self._provider}' not installed ({e}); using heuristic fallback.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client for '{self._provider}': {e}")

    def complete_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call the LLM and parse a JSON object from its response. Returns None on any failure."""
        if not self.available:
            return None
        try:
            finish_reason = None
            if self._provider in ("openai", "groq"):
                kwargs = dict(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                try:
                    # ask for strict JSON mode where the model/provider supports it;
                    # silently retry without it if the model rejects the parameter.
                    resp = self._client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except Exception:
                    resp = self._client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
            elif self._provider == "gemini":
                resp = self._client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                raw = resp.text
            else:
                return None

            if finish_reason == "length":
                logger.warning(
                    "LLM response was truncated (hit max_tokens). Attempting to recover partial "
                    f"JSON; consider raising LLM_MAX_TOKENS (currently {self.cfg.max_tokens}) for more complete extractions."
                )

            result, err = _parse_json_response(raw)
            if result is None:
                logger.warning(f"LLM returned non-JSON output, discarding. Error: {err}")
                return None
            if err:
                logger.info(f"LLM JSON output {err}")
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def complete_text(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Free-form text completion, used for final answer generation."""
        if not self.available:
            return None
        try:
            if self._provider in ("openai", "groq"):
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                return resp.choices[0].message.content
            elif self._provider == "gemini":
                resp = self._client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                return resp.text
        except Exception as e:
            logger.error(f"LLM text completion failed: {e}")
        return None


_singleton: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton
