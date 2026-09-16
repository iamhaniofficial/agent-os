"""Python dunder names stay literal in Telegram output (#2076).

``__init__`` is the same shape as ``__bold__``, so the underscore-bold pass
rendered ``call __init__ method`` as ``call <b>init</b> method`` and the
table-label path stripped it to ``init`` outright. Python's special names are
a fixed vocabulary, so a ``__name__`` from that vocabulary is kept as written
while ``__also__`` still renders bold.
"""

from __future__ import annotations

import pytest

from agentos.channels._telegram_formatting import _plain_inline, render_telegram_html


@pytest.mark.parametrize(
    "markdown",
    [
        "call __init__ method",
        "see __main__ and __all__",
        "override __str__ and __repr__",
        "__slots__ = ()",
        "define __post_init__ on the dataclass",
        "async with needs __aenter__ and __aexit__",
        "implement __radd__ and __iadd__ too",
        "the __name__ == '__main__' guard",
        "__version__ lives in __init__.py",
        "set __tablename__ on the model",
        "__cached__ points at the .pyc",
        "__tracebackhide__ = True hides the frame",
        "dataclasses set __replace__ on 3.13",
    ],
)
def test_dunder_names_are_not_bolded_in_body_text(markdown: str) -> None:
    rendered = render_telegram_html(markdown)

    assert "<b>" not in rendered
    assert rendered == markdown.replace("'", "&#x27;")


def test_dunder_names_are_not_stripped_in_table_labels() -> None:
    assert _plain_inline("call __init__ method") == "call __init__ method"
    assert _plain_inline("see __main__ and __all__") == "see __main__ and __all__"
    assert _plain_inline("__tracebackhide__") == "__tracebackhide__"


def test_dunder_name_and_bold_in_the_same_line() -> None:
    assert render_telegram_html("__init__ is __important__") == "__init__ is <b>important</b>"
    assert _plain_inline("__init__ is __important__") == "__init__ is important"


def test_dunder_case_is_significant() -> None:
    """``__INIT__`` is not a Python special name; it is ordinary bold."""

    assert render_telegram_html("__INIT__") == "<b>INIT</b>"


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("__also__", "<b>also</b>"),
        ("say __hello world__ now", "say <b>hello world</b> now"),
        ("__bold__ and __init__", "<b>bold</b> and __init__"),
    ],
)
def test_ordinary_underscore_bold_still_renders(markdown: str, expected: str) -> None:
    assert render_telegram_html(markdown) == expected


def test_table_header_strips_italic_markers() -> None:
    """``_Status_`` in a header lost the markers' meaning but kept the
    underscores; the header strips it the way it strips ``**bold**``."""

    assert _plain_inline("_Status_") == "Status"
    assert _plain_inline("Value is _pending_ now") == "Value is pending now"


@pytest.mark.parametrize(
    "label",
    [
        "snake_case_identifier stays intact",
        "use _private and _internal names",
        "the value_ trailing_ ones",
        "sha_a1_b2 and sha_c3_d4",
        "a _ lone underscore _ pair",
    ],
)
def test_table_label_keeps_identifier_underscores(label: str) -> None:
    assert _plain_inline(label) == label


def test_table_with_dunder_and_italic_headers() -> None:
    markdown = "| _Status_ | __init__ |\n|---|---|\n| _active_ | see __main__ |\n"

    assert render_telegram_html(markdown) == (
        "<b>Status — __init__</b>\n<b>active:</b> see __main__"
    )


def test_dunder_inside_a_code_span_is_untouched() -> None:
    assert render_telegram_html("`__init__`") == "<code>__init__</code>"
