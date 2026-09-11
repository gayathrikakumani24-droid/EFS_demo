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
import time
from typing import Any, Dict, Optional, Tuple

from config import CONFIG
from utils.logger import get_logger

logger = get_logger("extraction.llm_client")


def _strip_code_fences(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _extract_json_span(text: Optional[str]) -> str:
    """Trim any leading/trailing prose the model added around the JSON object,
    keeping everything from the first '{' to the last '}'."""
    if not text:
        return ""
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
    if not raw or not raw.strip():
        return None, "empty response"
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
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        import os
        from config import LLMConfig, _resolve_llm_api_key
        prov = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        mdl = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip()
        ext_mdl = os.getenv("LLM_EXTRACTION_MODEL", mdl).strip()
        code_mdl = os.getenv("LLM_CODE_MODEL", mdl).strip()
        key = _resolve_llm_api_key(prov)

        self.cfg = LLMConfig(
            provider=prov,
            model=mdl,
            extraction_model=ext_mdl,
            code_model=code_mdl,
            api_key=key
        )
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
            if self._provider in ("openai", "groq", "openrouter"):
                from openai import OpenAI
                base_url = self.cfg.base_url or (
                    "https://api.groq.com/openai/v1" if self._provider == "groq"
                    else "https://openrouter.ai/api/v1" if self._provider == "openrouter"
                    else None
                )
                headers = None
                if self._provider == "openrouter":
                    headers = {
                        "HTTP-Referer": "https://github.com/hardware-graphrag",
                        "X-Title": "Hardware GraphRAG",
                    }
                self._client = OpenAI(
                    api_key=self.cfg.api_key,
                    base_url=base_url,
                    timeout=self.cfg.timeout_s,
                    default_headers=headers
                )
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

    def _call_openai_with_retry(self, **kwargs):
        """Helper to invoke OpenAI/Groq/OpenRouter chat completions with rate limit & token budget detection."""
        if self._provider == "groq":
            fallback_models = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
                "groq/compound",
                "groq/compound-mini"
            ]
        elif self._provider == "openrouter":
            fallback_models = [
                "google/gemini-2.0-flash-001",
                "meta-llama/llama-3.3-70b-instruct",
                "qwen/qwen-2.5-coder-32b-instruct",
                "deepseek/deepseek-chat",
                "openai/gpt-oss-20b"
            ]
        else:
            fallback_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

        # Dynamic prompt budget check for Groq
        if self._provider == "groq":
            prompt_text = "".join(m.get("content", "") for m in kwargs.get("messages", []))
            est_prompt_tokens = len(prompt_text) // 3.5
            current_max = kwargs.get("max_tokens", 3072)
            total_budget = 14000
            if est_prompt_tokens + current_max > total_budget:
                adjusted_max = max(1024, int(total_budget - est_prompt_tokens))
                logger.info(f"Adjusting max_tokens from {current_max} to {adjusted_max} to fit Groq token budget.")
                kwargs["max_tokens"] = adjusted_max
            else:
                kwargs["max_tokens"] = min(current_max, 3072)

        kwargs.setdefault("timeout", 15.0)
        for attempt in range(2):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as ex:
                err_str = str(ex)

                # 1. Immediate credit/payment check - disable LLM API calls instantly
                if any(k in err_str.lower() for k in ["402", "insufficient credits", "credits", "payment_required", "openrouter_credits"]):
                    logger.warning(f"LLM API key has insufficient credits ({err_str[:120]}). Falling back to specification-agnostic heuristic engine.")
                    self.available = False
                    raise ex

                # 2. Token limit / payload size check
                if "413" in err_str or "request too large" in err_str.lower() or "context_length_exceeded" in err_str.lower() or "tpm limit" in err_str.lower():
                    logger.warning(f"Request too large (413/TPM limit). Retrying with fallback model... Error: {err_str[:120]}")
                    kwargs["max_tokens"] = min(kwargs.get("max_tokens", 3072), 2048)
                    kwargs.pop("response_format", None)
                    for alt_model in fallback_models:
                        if kwargs.get("model") != alt_model:
                            kwargs["model"] = alt_model
                            try:
                                return self._client.chat.completions.create(**kwargs)
                            except Exception:
                                continue
                    raise ex

                # 3. Model rate limit / decommissioned error
                elif any(k in err_str.lower() for k in ["429", "rate limit", "404", "400", "model_not_found", "model_decommissioned", "decommissioned"]):
                    kwargs.pop("response_format", None)
                    logger.warning(f"Model error/limit hit on '{kwargs.get('model')}': {err_str[:120]}. Retrying provider fallback models...")
                    for alt_model in fallback_models:
                        if kwargs.get("model") != alt_model:
                            kwargs["model"] = alt_model
                            time.sleep(0.5)
                            try:
                                return self._client.chat.completions.create(**kwargs)
                            except Exception:
                                continue
                    raise ex
                elif any(kw in err_str.lower() for kw in ["unreachable", "connection", "timeout", "503", "502", "500", "overloaded"]):
                    logger.warning(f"Model unreachable / connection error ({err_str[:100]}). Retrying fallback models...")
                    time.sleep(1.0)
                    for alt_model in fallback_models:
                        if kwargs.get("model") != alt_model:
                            kwargs["model"] = alt_model
                            try:
                                return self._client.chat.completions.create(**kwargs)
                            except Exception:
                                continue
                    raise ex
                else:
                    raise ex
        return self._client.chat.completions.create(**kwargs)

    def complete_json(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Call the LLM and parse a JSON object from its response. Returns None on any failure."""
        if not self.available:
            return None
        target_model = model or self.cfg.model
        if "json" not in system_prompt.lower():
            system_prompt = f"{system_prompt}\n\nRespond strictly in valid JSON format."
        try:
            finish_reason = None
            if self._provider in ("openai", "groq", "openrouter"):
                kwargs = dict(
                    model=target_model,
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
                    resp = self._call_openai_with_retry(response_format={"type": "json_object"}, **kwargs)
                except Exception:
                    kwargs.pop("response_format", None)
                    resp = self._call_openai_with_retry(**kwargs)

                choices = getattr(resp, "choices", None) if resp else None
                if not choices:
                    logger.warning("LLM call returned no choices/response.")
                    return None
                choice = choices[0]
                if not choice or not getattr(choice, "message", None):
                    logger.warning("LLM call choice contained no message.")
                    return None
                raw = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
            elif self._provider == "gemini":
                try:
                    resp = self._client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                    raw = resp.text if resp else ""
                except Exception as ex:
                    if "404" in str(ex) or "not found" in str(ex).lower():
                        import google.generativeai as genai
                        raw = None
                        for alt_model in ("gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-1.5-pro"):
                            try:
                                logger.info(f"Retrying Gemini JSON completion with model '{alt_model}'...")
                                fallback_client = genai.GenerativeModel(alt_model)
                                resp = fallback_client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                                raw = resp.text if resp else ""
                                self._client = fallback_client
                                break
                            except Exception:
                                continue
                        if raw is None:
                            raise ex
                    else:
                        raise ex

            else:
                return None

            if finish_reason == "length":
                logger.warning(
                    "LLM response was truncated (hit max_tokens). Attempting to recover partial "
                    f"JSON; consider raising LLM_MAX_TOKENS (currently {self.cfg.max_tokens}) for more complete extractions."
                )

            if not raw or not raw.strip():
                return None

            result, err = _parse_json_response(raw)
            if result is None:
                if err != "empty response":
                    logger.info(f"LLM returned non-JSON output ({err}). Falling back to generic specification-agnostic extraction.")
                return None
            if err:
                logger.info(f"LLM JSON output {err}")
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def complete_text(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Free-form text completion, used for final answer generation."""
        if not self.available:
            return None
        target_model = model or self.cfg.model
        try:
            if self._provider in ("openai", "groq", "openrouter"):
                resp = self._call_openai_with_retry(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                choices = getattr(resp, "choices", None) if resp else None
                if not choices:
                    logger.warning("LLM text completion returned response with no choices.")
                    return None
                choice = choices[0]
                if not choice or not getattr(choice, "message", None):
                    return None
                return choice.message.content or ""
            elif self._provider == "gemini":
                try:
                    resp = self._client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                    return resp.text
                except Exception as ex:
                    if "404" in str(ex) or "not found" in str(ex).lower():
                        import google.generativeai as genai
                        for alt_model in ("gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-1.5-pro"):
                            try:
                                logger.info(f"Retrying Gemini completion with model '{alt_model}'...")
                                fallback_client = genai.GenerativeModel(alt_model)
                                resp = fallback_client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                                self._client = fallback_client
                                return resp.text
                            except Exception:
                                continue
                    raise ex

        except Exception as e:
            logger.error(f"LLM text completion failed: {e}")
        return None


_singleton: Optional[LLMClient] = None


def get_llm_client(force_reload: bool = False) -> LLMClient:
    global _singleton
    if _singleton is None or force_reload:
        _singleton = LLMClient()
    return _singleton
