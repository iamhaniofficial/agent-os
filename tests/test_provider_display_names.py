"""`_provider_display_name` must name every OpenAI-compatible provider.

`_format_chat_http_error` is the only consumer of the display table, and its
whole job is telling an operator *which* provider rejected the request. A
missing entry degrades that to a generic "Provider chat request failed", which
is useless on an install with several providers configured.

The table is keyed on `ProviderSpec.provider_kind` (that is what
`provider/selector.py` hands `OpenAIProvider`), not on the provider id, so the
drift guard below derives its expectations from the registry.
"""

from __future__ import annotations

import pytest

from agentos.provider.failures import _OPENAI_COMPAT_PROVIDERS
from agentos.provider.openai import _format_chat_http_error, _provider_display_name
from agentos.provider.registry import list_provider_specs


def _openai_compat_kinds() -> set[str]:
    """Provider kinds that can reach `OpenAIProvider` at runtime."""

    return {
        spec.provider_kind
        for spec in list_provider_specs()
        if spec.backend == "openai_compat" and spec.runtime_supported
    }


@pytest.mark.parametrize("provider_kind", sorted(_openai_compat_kinds()))
def test_every_openai_compat_kind_has_a_display_name(provider_kind: str) -> None:
    assert _provider_display_name(provider_kind) != "Provider", (
        f"provider_kind {provider_kind!r} falls back to the generic label; "
        "add it to _provider_display_name"
    )


def test_every_openai_compat_failure_provider_resolves_to_a_named_kind() -> None:
    """Entries in `_OPENAI_COMPAT_PROVIDERS` must map to a named display name.

    Some entries there are provider *ids* (`vllm`, `minimax_openai`) rather
    than kinds; resolve those through the registry before checking.
    """

    id_to_kind = {spec.provider_id: spec.provider_kind for spec in list_provider_specs()}
    unnamed = sorted(
        name
        for name in _OPENAI_COMPAT_PROVIDERS
        if _provider_display_name(id_to_kind.get(name, name)) == "Provider"
    )

    assert unnamed == []


def test_newly_named_providers_use_each_vendors_own_casing() -> None:
    """Pins the spelling of the kinds this fix added.

    The parametrized guard above only demands *a* name; these vendors spell
    themselves in ways an autocomplete will happily get wrong.
    """

    expected = {
        "aihubmix": "AIHubMix",
        "azure": "Azure OpenAI",
        "bailian_coding": "Bailian Coding",
        "bankr": "Bankr",
        "byteplus": "BytePlus",
        "groq": "Groq",
        "lm_studio": "LM Studio",
        "minimax": "MiniMax",
        "mistral": "Mistral",
        "ovms": "OVMS",
        "siliconflow": "SiliconFlow",
    }

    assert {kind: _provider_display_name(kind) for kind in expected} == expected


def test_an_unknown_kind_still_falls_back_to_the_generic_label() -> None:
    assert _provider_display_name("definitely-not-a-provider") == "Provider"


def test_the_formatted_http_error_names_the_provider() -> None:
    message = _format_chat_http_error("mistral", 401, b'{"error": {"message": "bad key"}}')

    assert message.startswith("Mistral chat request failed (HTTP 401)")
    assert "bad key" in message
