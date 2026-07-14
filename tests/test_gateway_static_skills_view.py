from __future__ import annotations

from pathlib import Path


def test_skills_view_exposes_direct_github_install_control() -> None:
    view = Path("src/agentos/gateway/static/js/views/skills.js").read_text(encoding="utf-8")

    assert 'id="skills-github-url"' in view
    assert 'class="btn btn--primary" id="skills-github-install"' in view
    assert "_installSkill(githubInput.value.trim(), 'github'," in view


def test_skills_view_browses_community_catalog_without_source_picker() -> None:
    view = Path("src/agentos/gateway/static/js/views/skills.js").read_text(encoding="utf-8")

    # No redundant source dropdown — sources are aggregated by the router.
    assert 'id="skills-registry-source"' not in view
    # Registry search aggregates across community sources (no ClawHub-only copy).
    assert "Searching ClawHub" not in view
    assert "community skills" in view
    # Opening the Community tab browses the full catalog (empty-query search).
    assert "_registryBrowsed" in view
    assert "_searchRegistry('')" in view
    # Browse requests a larger page than a targeted search.
    assert "limit: browsing ? 200 : 40" in view


def test_skills_view_renders_provider_and_logo() -> None:
    view = Path("src/agentos/gateway/static/js/views/skills.js").read_text(encoding="utf-8")

    assert "_providerCell" in view
    assert "sk-registry__logo" in view
    # Falls back to initials when a skill has no logo asset.
    assert "sk-registry__logo--initials" in view


def test_skills_view_distinguishes_bundled_from_local_layers() -> None:
    view = Path("src/agentos/gateway/static/js/views/skills.js").read_text(encoding="utf-8")

    assert "Bundled skills ship with AgentOS." in view
    assert "Managed skills are locally installed into AgentOS state." in view
    assert "Personal skills are local user installs, not bundled." in view

