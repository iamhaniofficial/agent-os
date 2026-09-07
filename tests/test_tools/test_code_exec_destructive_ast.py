"""Tests for AST-based detection of destructive operations in code_exec (Issue #848).

Verifies that static regex evasion techniques (getattr, string concatenation,
__import__, importlib, exec/eval, wildcard imports, and import aliasing)
are accurately caught without false positives on benign code.
"""

from __future__ import annotations

import pytest

from agentos.tools.builtin.code_exec import _check_code_destructive


@pytest.mark.parametrize(
    ("code", "expected_keyword"),
    [
        # Concatenated strings in getattr
        ('import os; getattr(os, "rem" + "ove")("/tmp/x")', "remove"),
        ('import os; getattr(os, "un" + "link")("/tmp/x")', "unlink"),
        ('import os; getattr(os, "rm" + "dir")("/tmp/x")', "rmdir"),
        ('import shutil; getattr(shutil, "rm" + "tree")("/tmp/x")', "rmtree"),
        # f-strings in getattr
        ('import os; getattr(os, f"{\'rem\'}ove")("/tmp/x")', "remove"),
        # Dynamic __import__
        ('__import__("os").remove("/tmp/x")', "remove"),
        ('__import__("os").unlink("/tmp/x")', "unlink"),
        ('__import__("shutil").rmtree("/tmp/x")', "rmtree"),
        # Dynamic importlib.import_module
        ('import importlib; importlib.import_module("os").remove("/tmp/x")', "remove"),
        ('import importlib; importlib.import_module("shutil").rmtree("/tmp/x")', "rmtree"),
        # exec / eval with nested destructive code
        ("exec(\"os.remove('/tmp/x')\")", "remove"),
        ("eval(\"os.remove('/tmp/x')\")", "remove"),
        ('exec(\'getattr(os, "rem" + "ove")("/tmp/x")\')', "remove"),
        # Wildcard imports
        ("from os import *; remove('/tmp/x')", "remove"),
        ("from os import *; unlink('/tmp/x')", "unlink"),
        ("from shutil import *; rmtree('/tmp/x')", "rmtree"),
        # Aliased imports
        ("from os import remove as delete_file; delete_file('/tmp/x')", "remove"),
        ("from shutil import rmtree as nukedir; nukedir('/tmp/x')", "rmtree"),
        ("import os as my_os; my_os.remove('/tmp/x')", "remove"),
        ("import shutil as s; s.rmtree('/tmp/x')", "rmtree"),
        # Path methods via getattr
        ('from pathlib import Path; getattr(Path("/tmp/x"), "unlink")()', "unlink"),
        ('from pathlib import Path; getattr(Path("/tmp/x"), "rmdir")()', "rmdir"),
        # Subprocess list invocation of rm
        ('import subprocess; subprocess.run(["rm", "-rf", "/tmp/x"])', "subprocess invoking rm"),
        ('import subprocess as sp; sp.call(["rmdir", "/tmp/x"])', "subprocess invoking rm"),
    ],
)
def test_destructive_ast_evasions_detected(code: str, expected_keyword: str) -> None:
    warning = _check_code_destructive(code)
    assert warning is not None, f"Expected warning for: {code}"
    assert "destructive Python operation detected:" in warning
    assert expected_keyword.lower() in warning.lower()


@pytest.mark.parametrize(
    "code",
    [
        # Standard list.remove should NOT trigger
        "items = [1, 2, 3]\nitems.remove(2)",
        # Set.remove should NOT trigger
        "s = {1, 2, 3}\ns.remove(2)",
        # Benign math/sys/os calls
        "import math\nx = math.sqrt(16)",
        "import os\ncwd = os.getcwd()",
        "import os\nfiles = os.listdir('.')",
        "import shutil\nshutil.copy('a.txt', 'b.txt')",
        "from pathlib import Path\np = Path('a.txt').read_text()",
        # Benign getattr
        "import os\npath_fn = getattr(os, 'getcwd')",
        "getattr(dict, 'get')",
    ],
)
def test_benign_code_does_not_trigger_warning(code: str) -> None:
    warning = _check_code_destructive(code)
    assert warning is None, f"Unexpected warning for safe code: {warning}"


@pytest.mark.parametrize(
    ("code", "expected_keyword"),
    [
        # compile() as a code carrier: exec()/eval() receive the compiled form,
        # so the source argument has to be read or the inner code is never
        # scanned. _eval_const_str returned None for every ast.Call.
        ("exec(compile('os.re' + 'move(\"/tmp/x\")', '', 'exec'))", "remove"),
        ("eval(compile('os.remove(\"/tmp/x\")', '', 'eval'))", "remove"),
        ("exec(compile(source='os.remove(\"/tmp/x\")', filename='', mode='exec'))", "remove"),
        # getattr(__builtins__, "__import__")("os") — the module resolver
        # handled __import__ as a bare Name and importlib.import_module as an
        # Attribute, but not this nested-Call spelling.
        ('getattr(__builtins__, "__import__")("os").system("rm -rf /tmp/x")', "os.system"),
        ('import builtins; builtins.__import__("os").remove("/tmp/x")', "remove"),
        ('getattr(__builtins__, "__im" + "port__")("shutil").rmtree("/tmp/x")', "rmtree"),
        # Shell-exec attrs via getattr: os.system/os.popen and subprocess.* are
        # deliberately not in _ALL_DESTRUCTIVE_NAMES (only their argv is
        # destructive), so the getattr branch skipped them, and the callee is
        # an ast.Call rather than an ast.Attribute so that branch missed it too.
        ('import os; getattr(os, "system")("rm -rf /tmp/x")', "os.system"),
        ('import os; getattr(os, "sys" + "tem")("rm -rf /tmp/x")', "os.system"),
        ('import os; getattr(os, "popen")("rm -rf /tmp/x")', "os.popen"),
        (
            'import subprocess; getattr(subprocess, "run")(["rm", "-rf", "/tmp/x"])',
            "subprocess invoking rm",
        ),
        (
            'import subprocess as sp; getattr(sp, "Popen")("rm -rf /tmp/x")',
            "subprocess invoking rm",
        ),
    ],
)
def test_indirect_destructive_calls_detected(code: str, expected_keyword: str) -> None:
    # Residual bypasses of the Issue #848 gate (Issue #1102).
    warning = _check_code_destructive(code)
    assert warning is not None, f"Expected warning for: {code}"
    assert "destructive Python operation detected:" in warning
    assert expected_keyword.lower() in warning.lower()


@pytest.mark.parametrize(
    "code",
    [
        # compile()/getattr are ordinary tools; only destructive payloads count.
        "exec(compile('print(1)', '', 'exec'))",
        "code = compile('x = 1 + 1', '<gen>', 'exec')",
        "import os; getattr(os, 'system')('ls -la')",
        "import subprocess; getattr(subprocess, 'run')(['ls', '-la'])",
        "import os; fn = getattr(os, 'popen')",
        "getattr(__builtins__, '__import__')('math').sqrt(16)",
    ],
)
def test_benign_indirect_calls_do_not_trigger(code: str) -> None:
    assert _check_code_destructive(code) is None, f"Unexpected warning for: {code}"


def test_syntax_error_code_falls_back_to_regex() -> None:
    # Syntax error with os.remove() still caught by regex fallback
    bad_syntax_destructive = "os.remove( unclosed string"
    warning = _check_code_destructive(bad_syntax_destructive)
    assert warning is not None
    assert "os.remove()" in warning

    # Benign syntax error returns None
    bad_syntax_benign = "def foo( unclosed"
    assert _check_code_destructive(bad_syntax_benign) is None
