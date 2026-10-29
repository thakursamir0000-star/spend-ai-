"""Shared Groq chat helper with ordered model failover.

Every LLM call in the app goes through here, so the failover order lives in one
place and a failing model is logged instead of silently skipped.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_MODEL = "openai/gpt-oss-20b"
DEFAULT_FALLBACK_MODELS = [
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
]


def candidate_models(primary_env=None):
    """Primary model (from `primary_env` if set) followed by de-duplicated fallbacks.

    Fallbacks can be overridden with a comma-separated GROQ_FALLBACK_MODELS.
    """
    primary = os.environ.get(primary_env, "") if primary_env else ""
    fallbacks_env = os.environ.get("GROQ_FALLBACK_MODELS", "")
    fallbacks = [m.strip() for m in fallbacks_env.split(",") if m.strip()] or DEFAULT_FALLBACK_MODELS

    models = []
    for m in [primary or DEFAULT_PRIMARY_MODEL, *fallbacks]:
        if m and m not in models:
            models.append(m)
    return models


def chat(client, prompt, *, primary_env=None, temperature=0.1, json_mode=False):
    """Return the first successful completion text, or None if every model fails."""
    if client is None:
        return None

    for model in candidate_models(primary_env):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            if content:
                return content
            logger.warning("Model %s returned an empty response", model)
        except Exception as e:
            logger.warning("Model %s failed: %s", model, e)
    return None


def chat_json(client, prompt, *, primary_env=None, temperature=0.1):
    """Like chat(), but parses a JSON object; moves to the next model on invalid JSON."""
    if client is None:
        return None

    for model in candidate_models(primary_env):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.warning("Model %s returned invalid JSON: %s", model, e)
        except Exception as e:
            logger.warning("Model %s failed: %s", model, e)
    return None
