"""Generic ripgrep-backed repository search tool for DB-GPT.

The tool searches files under REPO_RG_ROOT, or PROJECT_PATH when REPO_RG_ROOT is
not set. It is intended as a lightweight replacement for knowledge-base RAG
retrieval when the source of truth is a mounted repository.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

from dbgpt.agent.resource.tool.base import tool


_REPO_ROOT = Path(
    os.environ.get("REPO_RG_ROOT", os.environ.get("PROJECT_PATH", "/knowledge"))
)
_ENV_IGNORE_PATHS = "REPO_RG_IGNORE_PATHS"

_DEFAULT_IGNORE_PATHS = [
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "vendor",
    "dist",
    "build",
]


def _repo_root() -> Path:
    return _REPO_ROOT.resolve()


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            part.strip()
            for part in value.replace(";", ",").replace("\n", ",").split(",")
            if part.strip()
        ]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value).strip().replace("\\", "/").strip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _pick(kwargs, *names, default=None):
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return default


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _bounded_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _default_ignore_paths() -> List[str]:
    return _dedupe([*_DEFAULT_IGNORE_PATHS, *_as_list(os.environ.get(_ENV_IGNORE_PATHS, ""))])


def _combined_ignore_paths(ignore_paths=None) -> List[str]:
    return _dedupe([*_default_ignore_paths(), *_as_list(ignore_paths)])


def _resolve_scope(paths=None) -> tuple[List[str], List[str]]:
    root = _repo_root()
    if not root.exists():
        return [], []
    requested = _dedupe(_as_list(paths))
    if not requested:
        return ["."], ["."]

    scope: List[str] = []
    for rel_path in requested:
        target = (root / rel_path).resolve()
        try:
            rel = target.relative_to(root)
        except ValueError:
            continue
        if target.exists():
            scope.append(str(rel).replace("\\", "/") or ".")
    return scope, scope


def _rg_exclude_globs(ignore_paths: Iterable[str]) -> List[str]:
    globs: List[str] = []
    for pattern in _dedupe(ignore_paths):
        pat = pattern.replace("\\", "/").strip("/")
        if not pat:
            continue
        globs.append(f"!{pat}")
        if not any(char in pat for char in "*?["):
            globs.append(f"!{pat}/**")
            if "/" not in pat:
                globs.append(f"!**/{pat}/**")
    return _dedupe(globs)


def _parse_rg_matches(output: str, max_results: int, max_chars: int) -> List[dict]:
    matches: List[dict] = []
    seen = set()
    for line in output.splitlines():
        if len(matches) >= max_results:
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        path = ((data.get("path") or {}).get("text") or "").strip()
        line_number = data.get("line_number", 0)
        key = (path, line_number)
        if not path or key in seen:
            continue
        seen.add(key)
        snippet = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        submatches = []
        for item in data.get("submatches") or []:
            match_text = ((item.get("match") or {}).get("text") or "")
            submatches.append(
                {
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "match": match_text[:max_chars],
                }
            )
        matches.append(
            {
                "path": path,
                "line": line_number,
                "text": snippet,
                "submatches": submatches,
            }
        )
    return matches


@tool(
    description=(
        "Search a mounted repository with ripgrep (`rg`). Use this as a direct "
        "repository retrieval tool instead of knowledge-base RAG when source files "
        "are mounted locally. Supports caller-provided include paths and ignore paths."
    ),
    args={
        "query": {
            "type": "string",
            "description": "Search text or regex pattern.",
            "required": True,
        },
        "paths": {
            "type": "array",
            "description": "Optional repository-relative files or directories to search. Omit to search the repository root.",
            "required": False,
        },
        "ignore_paths": {
            "type": "array",
            "description": "Optional repository-relative paths or glob patterns to exclude, e.g. ['external/**', 'tmp/**'].",
            "required": False,
        },
        "file_globs": {
            "type": "array",
            "description": "Optional include globs passed to rg, e.g. ['*.go', '*.md'].",
            "required": False,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum match rows to return (default 20, max 100).",
            "required": False,
            "default": 20,
        },
        "regex": {
            "type": "boolean",
            "description": "Interpret query as a regex. Defaults to false, using fixed-string search.",
            "required": False,
            "default": False,
        },
        "case_sensitive": {
            "type": "boolean",
            "description": "Force case-sensitive matching. Defaults to ripgrep smart-case.",
            "required": False,
        },
        "max_chars_per_match": {
            "type": "integer",
            "description": "Maximum characters per returned line (default 500, max 2000).",
            "required": False,
            "default": 500,
        },
    },
)
def repo_rg(**kwargs) -> str:
    query = _pick(kwargs, "query", "pattern", "keyword", "search", "q")
    if not query or not str(query).strip():
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    query = str(query).strip()

    rg = shutil.which("rg")
    if not rg:
        return json.dumps(
            {
                "error": "ripgrep_unavailable",
                "message": "repo_rg requires ripgrep (`rg`) to be installed and available on PATH",
            },
            ensure_ascii=False,
        )

    root = _repo_root()
    if not root.exists():
        return json.dumps(
            {"error": "repo_root_unavailable", "root": str(root)},
            ensure_ascii=False,
        )

    paths = _pick(kwargs, "paths", "path", "scope_paths", "search_paths")
    scope_args, scope = _resolve_scope(paths)
    if not scope_args:
        return json.dumps(
            {"error": "scope_unavailable", "root": str(root), "paths": _as_list(paths)},
            ensure_ascii=False,
        )

    ignore_paths = _pick(kwargs, "ignore_paths", "ignore", "exclude_paths", "exclude")
    ignored = _combined_ignore_paths(ignore_paths)
    file_globs = _pick(kwargs, "file_globs", "include_globs", "glob", "globs")
    max_results = _bounded_int(_pick(kwargs, "max_results", "limit", "count", default=20), 20, 1, 100)
    max_chars = _bounded_int(_pick(kwargs, "max_chars_per_match", "max_chars", default=500), 500, 80, 2000)
    regex = _as_bool(_pick(kwargs, "regex", "regexp"), default=False)
    case_sensitive = _pick(kwargs, "case_sensitive", "caseSensitive")

    args = [
        rg,
        "--json",
        "--color",
        "never",
        "--line-number",
        "--no-messages",
        "--max-columns",
        str(max_chars),
        "--max-columns-preview",
    ]
    if not regex:
        args.append("--fixed-strings")
    if case_sensitive is None:
        args.append("--smart-case")
    elif _as_bool(case_sensitive):
        args.append("--case-sensitive")
    else:
        args.append("--ignore-case")

    for glob in _rg_exclude_globs(ignored):
        args.extend(["--glob", glob])
    for glob in _as_list(file_globs):
        args.extend(["--glob", glob])
    args.append(query)
    args.extend(scope_args)

    try:
        result = subprocess.run(
            args,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"error": "ripgrep_timeout", "query": query, "scope": scope},
            ensure_ascii=False,
        )
    except OSError as exc:
        return json.dumps(
            {"error": "ripgrep_failed", "message": str(exc), "query": query},
            ensure_ascii=False,
        )

    if result.returncode not in (0, 1):
        return json.dumps(
            {
                "error": "ripgrep_failed",
                "query": query,
                "scope": scope,
                "stderr": result.stderr.strip()[:1000],
            },
            ensure_ascii=False,
        )

    matches = _parse_rg_matches(result.stdout, max_results, max_chars)
    return json.dumps(
        {
            "query": query,
            "engine": "ripgrep",
            "root": str(root),
            "scope": scope,
            "ignored": ignored,
            "file_globs": _as_list(file_globs),
            "regex": regex,
            "count": len(matches),
            "matches": matches,
        },
        ensure_ascii=False,
    )
