r"""Regression: the advanced dashboard's inline JS must stay valid JavaScript.

The page's inline <script> lives inside a Python triple-quoted string. A
`\n` escape meant for JavaScript was previously interpreted by Python as a
real newline, splitting a JS string literal across two lines. The whole script
then failed to parse, so the page never issued its API fetches and stayed stuck
on the initial "연결 중..." header.

These tests pin the two invariants that keep the served page executable:
  - the delete-confirm prompt keeps its `\n` escape as backslash + 'n';
  - the full inline script parses under `node --check`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from scripts import experiment_dashboard  # noqa: E402


def _inline_script(html: str) -> str:
    match = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
    assert match is not None, "dashboard page must contain exactly one inline <script> block"
    return match.group(1)


def test_delete_prompt_keeps_newline_escape_intact() -> None:
    script = _inline_script(experiment_dashboard.PAGE)
    needle = "확인하려면 DELETE 입력"
    needle_start = script.index(needle)
    string_start = script.rfind('"', 0, needle_start)
    string_literal = script[string_start : needle_start + len(needle)]
    assert "\\n" in string_literal, "JS string must keep the \\n escape as backslash + 'n'"
    assert "\n" not in string_literal, "a raw newline inside a JS string literal breaks the page"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_inline_script_parses_as_javascript() -> None:
    script = _inline_script(experiment_dashboard.PAGE)
    result = subprocess.run(
        ["node", "--check"], input=script.encode("utf-8"), capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
