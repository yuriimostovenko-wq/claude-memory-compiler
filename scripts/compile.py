"""
Compile daily conversation logs into structured knowledge articles.

This is the "LLM compiler" - it reads daily logs (source code) and produces
organized knowledge articles (the executable).

Usage:
    uv run python compile.py                    # compile new/changed logs only
    uv run python compile.py --all              # force recompile everything
    uv run python compile.py --file daily/2026-04-01.md  # compile a specific log
    uv run python compile.py --dry-run          # show what would be compiled
"""

from __future__ import annotations

import argparse
import asyncio
import time
import sys
from pathlib import Path

from config import AGENTS_FILE, CONCEPTS_DIR, CONNECTIONS_DIR, DAILY_DIR, KNOWLEDGE_DIR, RAW_DIR, now_iso
from utils import (
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_state,
    read_wiki_index,
    save_state,
)

# ── Paths for the LLM to use ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


async def compile_daily_log(log_path: Path, state: dict) -> float:
    """Compile a single daily log into knowledge articles.

    Returns the API cost of the compilation.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    log_content = log_path.read_text(encoding="utf-8")
    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()

    # Read existing articles for context
    existing_articles_context = ""
    existing = {}
    for article_path in list_wiki_articles():
        rel = article_path.relative_to(KNOWLEDGE_DIR)
        existing[str(rel)] = article_path.read_text(encoding="utf-8")

    if existing:
        parts = []
        for rel_path, content in existing.items():
            parts.append(f"### {rel_path}\n```markdown\n{content}\n```")
        existing_articles_context = "\n\n".join(parts)

    timestamp = now_iso()

    prompt = f"""You are a knowledge compiler. Your job is to read a daily conversation log
and extract knowledge into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Existing Wiki Articles

{existing_articles_context if existing_articles_context else "(No existing articles yet)"}

## Daily Log to Compile

**File:** {log_path.name}

{log_content}

## Your Task

Read the daily log above and compile it into wiki articles following the schema exactly.

### Rules:

1. **Extract key concepts** - Identify 3-7 distinct concepts worth their own article
2. **Create concept articles** in `knowledge/concepts/` - One .md file per concept
   - Use the exact article format from AGENTS.md (YAML frontmatter + sections)
   - Include `sources:` in frontmatter pointing to the daily log file
   - Use `[[concepts/slug]]` wikilinks to link to related concepts
   - Write in encyclopedia style - neutral, comprehensive
3. **Create connection articles** in `knowledge/connections/` if this log reveals non-obvious
   relationships between 2+ existing concepts
4. **Update existing articles** if this log adds new information to concepts already in the wiki
   - Read the existing article, add the new information, add the source to frontmatter
5. **Update knowledge/index.md** - Add new entries to the table
   - Each entry: `| [[path/slug]] | One-line summary | source-file | {timestamp[:10]} |`
6. **Append to knowledge/log.md** - Add a timestamped entry:
   ```
   ## [{timestamp}] compile | {log_path.name}
   - Source: daily/{log_path.name}
   - Articles created: [[concepts/x]], [[concepts/y]]
   - Articles updated: [[concepts/z]] (if any)
   ```

### File paths:
- Write concept articles to: {CONCEPTS_DIR}
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Append log at: {KNOWLEDGE_DIR / 'log.md'}

### Quality standards:
- Every article must have complete YAML frontmatter
- Every article must link to at least 2 other articles via [[wikilinks]]
- Key Points section should have 3-5 bullet points
- Details section should have 2+ paragraphs
- Related Concepts section should have 2+ entries
- Sources section should cite the daily log with specific claims extracted
"""

    cost = 0.0

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model="claude-haiku-4-5-20251001",
                cwd=str(ROOT_DIR),
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="acceptEdits",
                max_turns=30,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        pass  # compilation output - LLM writes files directly
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
                print(f"  Cost: ${cost:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0

    # Update state
    rel_path = log_path.name
    state.setdefault("ingested", {})[rel_path] = {
        "hash": file_hash(log_path),
        "compiled_at": now_iso(),
        "cost_usd": cost,
    }
    state["total_cost"] = state.get("total_cost", 0.0) + cost
    save_state(state)

    return cost


BINARY_SUFFIXES = {".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt",
                   ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".tar"}


async def compile_raw_source(raw_path: Path, state: dict) -> float:
    """Compile a single markdown file from /raw into knowledge articles.

    Binary files (PDF, DOCX, etc.) are skipped — process them via skills first.
    Returns the API cost.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    if raw_path.suffix.lower() in BINARY_SUFFIXES:
        print(f"  SKIP (binary): {raw_path.name} — use pdf-viewer/office-docx skill first")
        return 0.0

    content = raw_path.read_text(encoding="utf-8")
    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()

    existing_articles_context = ""
    existing = {}
    for article_path in list_wiki_articles():
        rel = article_path.relative_to(KNOWLEDGE_DIR)
        existing[str(rel)] = article_path.read_text(encoding="utf-8")
    if existing:
        parts = [f"### {p}\n```markdown\n{c}\n```" for p, c in existing.items()]
        existing_articles_context = "\n\n".join(parts)

    timestamp = now_iso()

    prompt = f"""You are a knowledge compiler processing a raw source document from the project's /raw folder.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Existing Wiki Articles

{existing_articles_context if existing_articles_context else "(No existing articles yet)"}

## Raw Source Document to Compile

**File:** raw/{raw_path.name}

{content}

## Your Task

Extract knowledge from this raw document into structured wiki articles.

### Rules:
1. **Extract key concepts** - Identify concepts worth their own article
2. **Create concept articles** in `knowledge/concepts/` with YAML frontmatter
   - Set `sources: ["raw/{raw_path.name}"]` in frontmatter
   - Write encyclopedia style, use [[wikilinks]]
3. **Create connection articles** in `knowledge/connections/` for non-obvious cross-concept links
4. **Update existing articles** if this document adds new information
5. **Update knowledge/index.md** with new entries
6. **Append to knowledge/log.md**:
   ```
   ## [{timestamp}] compile-raw | {raw_path.name}
   - Source: raw/{raw_path.name}
   - Articles created: [[concepts/x]]
   - Articles updated: (if any)
   ```

### File paths:
- Write concept articles to: {CONCEPTS_DIR}
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Append log at: {KNOWLEDGE_DIR / 'log.md'}
"""

    cost = 0.0
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model="claude-haiku-4-5-20251001",
                cwd=str(ROOT_DIR),
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="acceptEdits",
                max_turns=30,
            ),
        ):
            if isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
                print(f"  Cost: ${cost:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0

    state.setdefault("raw_ingested", {})[raw_path.name] = {
        "hash": file_hash(raw_path),
        "compiled_at": now_iso(),
        "cost_usd": cost,
    }
    state["total_cost"] = state.get("total_cost", 0.0) + cost
    save_state(state)
    return cost


def list_raw_markdown_files() -> list[Path]:
    """List markdown files in /raw, skip binaries."""
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() == ".md")


def list_raw_binaries() -> list[Path]:
    """List binary files in /raw that need manual skill processing."""
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in BINARY_SUFFIXES)


def main():
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true", help="Force recompile all logs")
    parser.add_argument("--file", type=str, help="Compile a specific daily log file")
    parser.add_argument("--raw", action="store_true", help="Also compile markdown files from /raw")
    parser.add_argument("--raw-file", type=str, help="Compile a specific file from /raw (name only, e.g. IP_Box_Verification_v1.md)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compiled")
    args = parser.parse_args()

    state = load_state()

    # --raw-file: compile a single specific raw file
    if args.raw_file:
        name = Path(args.raw_file).name
        target = RAW_DIR / name
        if not target.exists():
            print(f"Error: {name} not found in {RAW_DIR}")
            print(f"Available .md files in /raw:")
            for p in sorted(RAW_DIR.glob("*.md")):
                compiled = "✓" if state.get("raw_ingested", {}).get(p.name) else "○"
                print(f"  {compiled} {p.name}")
            sys.exit(1)
        if args.dry_run:
            print(f"[DRY RUN] Would compile: raw/{name}")
            return
        print(f"Compiling raw/{name}...")
        cost = asyncio.run(compile_raw_source(target, state))
        articles = list_wiki_articles()
        print(f"\nDone. Cost: ${cost:.4f} | Knowledge base: {len(articles)} articles")
        return

    # Determine which files to compile
    if args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = DAILY_DIR / target.name
        if not target.exists():
            # Try resolving relative to project root
            target = ROOT_DIR / args.file
        if not target.exists():
            print(f"Error: {args.file} not found")
            sys.exit(1)
        to_compile = [target]
    else:
        all_logs = list_raw_files()
        if args.all:
            to_compile = all_logs
        else:
            to_compile = []
            for log_path in all_logs:
                rel = log_path.name
                prev = state.get("ingested", {}).get(rel, {})
                if not prev or prev.get("hash") != file_hash(log_path):
                    to_compile.append(log_path)

    # --raw: find new/changed markdown files in /raw
    raw_to_compile: list[Path] = []
    if args.raw:
        raw_md = list_raw_markdown_files()
        raw_bins = list_raw_binaries()
        if raw_bins:
            print(f"  /raw binaries (need skill processing first): "
                  + ", ".join(p.name for p in raw_bins))
        for raw_path in raw_md:
            prev = state.get("raw_ingested", {}).get(raw_path.name, {})
            if args.all or not prev or prev.get("hash") != file_hash(raw_path):
                raw_to_compile.append(raw_path)

    if not to_compile and not raw_to_compile:
        print("Nothing to compile - all logs and /raw sources are up to date.")
        return

    if to_compile:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Daily logs to compile ({len(to_compile)}):")
        for f in to_compile:
            print(f"  - {f.name}")
    if raw_to_compile:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}/raw sources to compile ({len(raw_to_compile)}):")
        for f in raw_to_compile:
            print(f"  - raw/{f.name}")

    if args.dry_run:
        return

    total_cost = 0.0
    INTER_FILE_DELAY = 8  # seconds between files to avoid rate limiting

    for i, log_path in enumerate(to_compile, 1):
        print(f"\n[{i}/{len(to_compile)}] Compiling {log_path.name}...")
        cost = asyncio.run(compile_daily_log(log_path, state))
        total_cost += cost
        print(f"  Done.")
        if i < len(to_compile) or raw_to_compile:
            time.sleep(INTER_FILE_DELAY)

    for i, raw_path in enumerate(raw_to_compile, 1):
        print(f"\n[raw {i}/{len(raw_to_compile)}] Compiling raw/{raw_path.name}...")
        cost = asyncio.run(compile_raw_source(raw_path, state))
        total_cost += cost
        print(f"  Done.")
        if i < len(raw_to_compile):
            time.sleep(INTER_FILE_DELAY)

    articles = list_wiki_articles()
    print(f"\nCompilation complete. Total cost: ${total_cost:.2f}")
    print(f"Knowledge base: {len(articles)} articles")


if __name__ == "__main__":
    main()
