"""Shared Groq chat helper with ordered model failover.

Every LLM call in the app goes through here, so the failover order lives in one
place and a failing model is logged instead of silently skipped. A model the
API reports as missing (404) is skipped for the rest of the process, and one
rate-limited for longer than the SDK waits (a daily cap) is skipped until the
limit resets.
"""
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_MODEL = "openai/gpt-oss-20b"
DEFAULT_FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# The Groq SDK already retries a 429 after the server's retry-after when that is
# at most 60 s; a longer wait means a per-day limit, not a busy minute.
SDK_MAX_RETRY_AFTER = 60

_unavailable_models = set()
_rate_limited_until = {}


def _is_available(model):
    return model not in _unavailable_models and _rate_limited_until.get(model, 0) <= time.monotonic()


def candidate_models(primary_env=None):
    """Primary model (from `primary_env` if set) followed by de-duplicated fallbacks.

    Fallbacks can be overridden with a comma-separated GROQ_FALLBACK_MODELS.
    Models already known to be missing or rate-limited for the day are left out.
    """
    primary = os.environ.get(primary_env, "") if primary_env else ""
    fallbacks_env = os.environ.get("GROQ_FALLBACK_MODELS", "")
    fallbacks = [m.strip() for m in fallbacks_env.split(",") if m.strip()] or DEFAULT_FALLBACK_MODELS

    models = []
    for m in [primary or DEFAULT_PRIMARY_MODEL, *fallbacks]:
        if m and m not in models and _is_available(m):
            models.append(m)
    return models


def _is_missing_model(error):
    return getattr(error, "status_code", None) == 404 or "model_not_found" in str(error)


def _retry_after(error):
    """Seconds a 429 asked us to wait, or None."""
    if getattr(error, "status_code", None) != 429:
        return None
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _record_failure(model, error):
    wait = _retry_after(error)
    if _is_missing_model(error):
        _unavailable_models.add(model)
        logger.warning("Model %s is not available on this account; skipping it from now on", model)
    elif wait and wait > SDK_MAX_RETRY_AFTER:
        _rate_limited_until[model] = time.monotonic() + wait
        logger.warning("Model %s is rate-limited for %.0f s (likely a daily limit); skipping it until then",
                       model, wait)
    else:
        logger.warning("Model %s failed: %s", model, error)


def _create(client, model, prompt, temperature, json_mode):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model.startswith("openai/gpt-oss"):
        # Reasoning tokens share the output budget; keep them short so the
        # answer itself (often a long JSON list) isn't cut off.
        kwargs["reasoning_effort"] = "low"
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def chat(client, prompt, *, primary_env=None, temperature=0.1, json_mode=False):
    """Return the first successful completion text, or None if every model fails."""
    if client is None:
        return None

    for model in candidate_models(primary_env):
        try:
            content = _create(client, model, prompt, temperature, json_mode)
            if content:
                return content
            logger.warning("Model %s returned an empty response", model)
        except Exception as e:
            _record_failure(model, e)
    return None


def chat_json(client, prompt, *, primary_env=None, temperature=0.1):
    """Like chat(), but parses a JSON object; moves to the next model on invalid JSON."""
    if client is None:
        return None

    for model in candidate_models(primary_env):
        try:
            return json.loads(_create(client, model, prompt, temperature, json_mode=True))
        except json.JSONDecodeError as e:
            logger.warning("Model %s returned invalid JSON: %s", model, e)
        except Exception as e:
            _record_failure(model, e)
    return None


def reset_unavailable_models():
    """Forget which models were missing or rate-limited (used by tests)."""
    _unavailable_models.clear()
    _rate_limited_until.clear()
