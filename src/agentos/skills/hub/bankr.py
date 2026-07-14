"""Bankr skill source — browses and installs skills from BankrBot/skills.

The Bankr repository (https://github.com/BankrBot/skills) publishes each skill
as a directory containing ``SKILL.md`` + ``catalog.json``. This source reads the
catalog live from GitHub (cached in-memory for a short TTL) so users can browse
the full catalog and install with one click. Downloading and installation are
delegated to :class:`GitHubSource`, which already fetches the whole skill
directory and reads the ``SKILL.md`` frontmatter; the security scan, quarantine,
and lockfile handling live in :class:`SkillInstaller` and are reused unchanged.

Only skills whose ``catalog.json`` declares ``install.type == "bankr"`` (i.e.
they live in the repo and install directly) are listed. ``external`` skills —
whose install runs a third-party command — are skipped.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import structlog

from agentos.env import trust_env as _trust_env
from agentos.skills.hub.github import GitHubSource
from agentos.skills.hub.source import SkillBundle, SkillMeta, SkillSource

log = structlog.get_logger(__name__)

_REPO = "BankrBot/skills"
_REF = "main"
_TREE_URL = f"https://api.github.com/repos/{_REPO}/git/trees/{_REF}?recursive=1"
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/{_REF}"
_CATALOG_TTL_SECONDS = 15 * 60
_CATALOG_CONCURRENCY = 16


def _tree_url(slug: str) -> str:
    return f"https://github.com/{_REPO}/tree/{_REF}/{slug}"


@dataclass
class _CatalogCache:
    """In-memory cache of the parsed Bankr catalog listing."""

    metas: list[SkillMeta]
    fetched_at: float  # time.monotonic() timestamp

    def is_fresh(self, ttl: float) -> bool:
        return (time.monotonic() - self.fetched_at) < ttl


def _catalog_slugs(tree: dict) -> list[str]:
    """Return skill slugs that own a top-level ``<slug>/catalog.json`` path."""
    slugs: list[str] = []
    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = str(item.get("path") or "")
        parts = path.split("/")
        # Only top-level skill directories: exactly "<slug>/catalog.json".
        if len(parts) == 2 and parts[1] == "catalog.json" and parts[0]:
            slugs.append(parts[0])
    return slugs


def _meta_from_catalog(slug: str, catalog: dict) -> SkillMeta | None:
    """Build a browse-time SkillMeta from a parsed catalog.json.

    Returns ``None`` when the skill is not a directly-installable ``bankr``
    skill (e.g. an ``external`` install), so callers can skip it.
    """
    install = catalog.get("install")
    if not isinstance(install, dict) or install.get("type") != "bankr":
        return None

    provider = str(catalog.get("provider") or "")
    logo_name = catalog.get("logo")
    logo = f"{_RAW_BASE}/{slug}/{logo_name}" if isinstance(logo_name, str) and logo_name else ""

    return SkillMeta(
        name=slug,
        # description lives in SKILL.md frontmatter; filled at fetch() time.
        description="",
        source_id="bankr",
        trust_level="community",
        identifier=_tree_url(slug),
        homepage=str(catalog.get("providerUrl") or _tree_url(slug)),
        provider=provider,
        logo=logo,
    )


def _matches(meta: SkillMeta, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    haystack = " ".join([meta.name, meta.provider, meta.description, *meta.tags]).lower()
    return q in haystack


class BankrSource(SkillSource):
    """Skill source backed by the BankrBot/skills GitHub catalog."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._github = GitHubSource(token=token)
        self._cache: _CatalogCache | None = None
        self._lock = asyncio.Lock()

    @property
    def source_id(self) -> str:
        return "bankr"

    @property
    def trust_level(self) -> str:
        return "community"

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            h["Authorization"] = f"token {self._token}"
        return h

    async def search(self, query: str, limit: int = 200) -> list[SkillMeta]:
        """List Bankr skills (all when query is empty; filtered otherwise)."""
        metas = await self._load_catalog()
        results = [m for m in metas if _matches(m, query)]
        return results[:limit]

    async def inspect(self, identifier: str) -> SkillMeta | None:
        return await self._github.inspect(identifier)

    async def fetch(self, identifier: str) -> SkillBundle | None:
        return await self._github.fetch(identifier)

    async def _load_catalog(self) -> list[SkillMeta]:
        cache = self._cache
        if cache is not None and cache.is_fresh(_CATALOG_TTL_SECONDS):
            return cache.metas

        async with self._lock:
            # Re-check after acquiring the lock — another coroutine may have
            # populated the cache while we waited.
            cache = self._cache
            if cache is not None and cache.is_fresh(_CATALOG_TTL_SECONDS):
                return cache.metas

            metas = await self._fetch_catalog()
            if metas is None:
                # Fetch failed: serve stale cache if we have one, else empty.
                return self._cache.metas if self._cache is not None else []

            self._cache = _CatalogCache(metas=metas, fetched_at=time.monotonic())
            return metas

    async def _fetch_catalog(self) -> list[SkillMeta] | None:
        """Fetch the tree + all catalog.json files. Returns None on tree failure."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=15, trust_env=_trust_env()) as client:
                tree_resp = await client.get(_TREE_URL, headers=self._headers())
                tree_resp.raise_for_status()
                tree = tree_resp.json()
                if tree.get("truncated"):
                    log.warning("bankr.tree_truncated")
                    return None

                slugs = _catalog_slugs(tree)
                if not slugs:
                    return []

                sem = asyncio.Semaphore(_CATALOG_CONCURRENCY)

                async def _load_one(slug: str) -> SkillMeta | None:
                    async with sem:
                        return await self._load_catalog_entry(client, slug)

                loaded = await asyncio.gather(*(_load_one(s) for s in slugs))
        except Exception as exc:
            log.warning("bankr.tree_failed", error=str(exc))
            return None

        metas = [m for m in loaded if m is not None]
        metas.sort(key=lambda m: m.name)
        return metas

    async def _load_catalog_entry(self, client, slug: str) -> SkillMeta | None:
        """Fetch and parse one skill's catalog.json. Skips on any error."""
        url = f"{_RAW_BASE}/{slug}/catalog.json"
        try:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            catalog = json.loads(resp.content)
        except Exception as exc:
            log.warning("bankr.catalog_failed", slug=slug, error=str(exc))
            return None
        if not isinstance(catalog, dict):
            return None
        return _meta_from_catalog(slug, catalog)
