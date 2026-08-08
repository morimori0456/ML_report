#!/usr/bin/env python3
"""Extract the slash-command inventory from an installed Claude Code binary.

`/help` shows the commands that are *enabled for you right now*. This script shows
every command that is *defined in the build*, including the ones gated behind a
plan, a platform, a session type, or a feature flag. The difference between the two
lists is often what you were looking for.

The binary is a single bundled JS blob, so the command registry survives as literal
string fragments like:

    {type:"local-jsx",name:"permissions",aliases:["allowed-tools"],description:"..."}

We recover those with regexes rather than by parsing JS. That is deliberate: the
bundle is minified and its variable names change every release, but these object
literals have stayed stable across versions.

Usage:
    python extract_slash_commands.py                  # auto-detect newest install
    python extract_slash_commands.py --binary PATH    # a specific version
    python extract_slash_commands.py --format md      # markdown table
    python extract_slash_commands.py --diff OLD NEW   # what changed between builds

No dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --- Where installers put the binary ------------------------------------------------
# The native installer keeps every version it has downloaded; we take the newest by
# mtime rather than by version string so that "newest" is unambiguous.
SEARCH_GLOBS = [
    "~/.local/share/claude/versions/*",
    "~/.claude/local/node_modules/@anthropic-ai/claude-code/cli.js",
    "/usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js",
    "/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js",
]

# Command names are kebab-case in every release so far. That single rule already
# excludes the snake_case computer-use / Chrome tool schemas (`left_click`,
# `read_page`, `computer_batch`) that are otherwise indistinguishable from a command:
# a `name` sitting next to a `description`.
NAME = r'[a-z][a-z0-9-]{1,30}'
RE_NAME = re.compile(rf'name:"(?P<name>{NAME})"')

# Sibling keys that only ever appear on a slash-command object. A bare
# name/description pair is not evidence — the bundle holds thousands of those. At
# least one of these must sit near the name for us to treat it as a command.
COMMAND_MARKERS = (
    'type:"local"', 'type:"local-jsx"', 'type:"prompt"',
    "aliases:[", "menuDescription:", "argumentHint:", "supportsNonInteractive:",
    "progressMessage:", "thinClientDispatch:", "fleetHostCall:",
    "survivesBundledKillSwitch:", "userFacingName",
)

# Keys that positively identify a tool schema, which we never want.
TOOL_MARKERS = ("input_schema", "inputSchema", "inputJSONSchema")

# How far around the name to look for those markers: wide enough to span a small
# object literal, tight enough to rarely reach into the neighbouring one.
WINDOW = 260

# Matched with .match(blob, pos), which already anchors at pos — no ^ (the blob has
# newlines, so ^ would additionally demand start-of-line and never fire).
RE_ALIASES = re.compile(r',aliases:\[(?P<aliases>[^\]]*)\]')

# Static descriptions only. Many commands compute theirs at runtime
# (`get description(){...}`), so a blank description here is expected, not a bug.
RE_DESCRIPTION = re.compile(
    rf'name:"(?P<name>{NAME})",'
    rf'(?:aliases:\[[^\]]*\],)?'
    rf'(?:[a-zA-Z$_][\w$]*:!?[01],)?'
    rf'description:"(?P<description>(?:[^"\\]|\\.){{4,200}})"'
)

# Gating flags, so a command can be reported as conditional rather than as available.
RE_GATED = re.compile(rf'name:"(?P<name>{NAME})"[^{{}}]{{0,400}}?(?:isHidden|isEnabled)')

# `name:"..."` literals that survive every filter above but are still not commands.
# Each was confirmed by hand: Bash-tool argument specs, and highlight.js language
# definitions that happen to carry a command-shaped sibling key.
NOT_COMMANDS = {
    "command", "count", "definition", "duration", "files", "default", "stub",
    "sleep", "timeout", "nohup", "time", "srun", "crmsh", "sharp", "pyright",
    "function", "callback",
}

# Commands registered through the `Ru({...})` helper often keep their `description`
# in a variable and only inline a `menuDescription`. Use it as the fallback.
RE_MENU_DESCRIPTION = re.compile(
    rf'name:"(?P<name>{NAME})",menuDescription:"(?P<description>(?:[^"\\]|\\.){{4,200}})"'
)

# Key order is not guaranteed — a few objects (/rewind, /release-notes) put the
# description first. Without this, they report no description at all.
RE_DESCRIPTION_BEFORE = re.compile(
    rf'description:"(?P<description>(?:[^"\\]|\\.){{4,200}})",name:"(?P<name>{NAME})"'
)

CATEGORIES: list[tuple[str, list[str]]] = [
    ("Session & conversation", [
        "help", "clear", "compact", "autocompact", "context", "resume", "branch",
        "fork", "rename", "recap", "export", "copy", "diff", "rewind", "exit",
    ]),
    ("Model & how work gets done", [
        "model", "effort", "fast", "advisor", "plan", "plan-artifact", "goal",
        "btw", "subtask", "batch", "list-agents", "agents",
    ]),
    ("Configuration", [
        "config", "permissions", "hooks", "sandbox", "add-dir", "cd", "memory",
        "pause-memory", "alias", "keybindings", "terminal-setup", "statusline",
        "theme", "color", "tui", "focus", "scroll-speed", "brief", "voice",
        "import", "privacy-settings", "auto-mode-setup", "setup-bedrock",
        "setup-vertex", "wellbeing",
    ]),
    ("Extensibility (MCP / plugins / skills)", [
        "mcp", "plugin", "reload-plugins", "skills", "reload-skills",
        "skill-doctor", "run-skill-generator", "claude-in-chrome", "powerup",
        "insights", "team-onboarding",
    ]),
    # Skills bundled with the build. Skills you install yourself live on disk and are
    # resolved at startup, so they never appear here — `/skills` is the list for those.
    ("Bundled skills (invocable as /name)", [
        "init", "run", "loop", "code-review", "review",
        "security-review", "simplify", "update-config", "keybindings-help",
        "fewer-permission-prompts", "claude-api", "dataviz", "design-sync",
    ]),
    ("Remote, cloud & background", [
        "remote-control", "teleport", "session", "web-setup", "remote-env",
        "schedule", "daemon", "tasks", "background", "stop", "loops", "workflows",
        "artifacts", "desktop", "mobile", "ultraplan", "ultrareview", "autofix-pr",
        "install-github-app", "install-slack-app", "design", "design-login",
        "design-revoke", "design-consent", "workflow-launch-exec",
    ]),
    ("Account, usage & diagnostics", [
        "status", "usage", "usage-credits", "extra-usage", "upgrade", "passes",
        "login", "logout", "version", "update", "install", "release-notes",
        "doctor", "ide", "chrome", "bug", "feedback", "explain-usage",
        "setup-cowork", "debug", "stickers", "radio", "heapdump",
        "pro-trial-expired", "rate-limit-options",
    ]),
]


def find_binary() -> Path:
    """Return the most recently modified Claude Code binary we can locate."""
    candidates: list[Path] = []
    for pattern in SEARCH_GLOBS:
        expanded = Path(pattern).expanduser()
        if "*" in pattern:
            candidates.extend(p for p in expanded.parent.glob(expanded.name) if p.is_file())
        elif expanded.is_file():
            candidates.append(expanded)
    if not candidates:
        sys.exit(
            "No Claude Code binary found. Pass one explicitly with --binary, or run "
            "`which claude` and follow the symlink."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_aliases(raw: str | None) -> list[str]:
    if not raw:
        return []
    return re.findall(r'"([^"]+)"', raw)


def extract(binary: Path) -> dict[str, dict]:
    """Recover the command registry. Returns {name: {aliases, description, gated}}."""
    # latin-1 never raises on arbitrary bytes and leaves ASCII intact, which is all
    # the regexes look at. Reading the whole ~300 MB blob keeps matches from being
    # split across chunk boundaries.
    blob = binary.read_bytes().decode("latin-1")

    commands: dict[str, dict] = {}

    for match in RE_NAME.finditer(blob):
        name = match.group("name")
        if name in NOT_COMMANDS:
            continue
        window = blob[max(0, match.start() - WINDOW):match.end() + WINDOW]
        if any(t in window for t in TOOL_MARKERS):
            continue
        if not any(m in window for m in COMMAND_MARKERS):
            continue
        entry = commands.setdefault(name, {"aliases": [], "description": "", "gated": False})
        alias_match = RE_ALIASES.match(blob, match.end())
        for alias in parse_aliases(alias_match.group("aliases") if alias_match else None):
            if alias not in entry["aliases"]:
                entry["aliases"].append(alias)

    # Collect every static description a name carries, then choose. Some commands are
    # defined twice — once for the terminal and once for an alternate host (FleetView)
    # — and the host-specific copy is the misleading one to show.
    candidates: dict[str, list[str]] = {}
    for match in RE_DESCRIPTION.finditer(blob):
        name = match.group("name")
        if name in commands:
            # The bundle stores \u escapes and escaped quotes; decode via JSON.
            text = json.loads(f'"{match.group("description")}"')
            candidates.setdefault(name, []).append(text)
    for pattern in (RE_DESCRIPTION_BEFORE, RE_MENU_DESCRIPTION):
        for match in pattern.finditer(blob):
            name = match.group("name")
            if name in commands:
                text = json.loads(f'"{match.group("description")}"')
                candidates.setdefault(name, []).append(text)
    for name, texts in candidates.items():
        generic = [t for t in texts if "FleetView" not in t]
        commands[name]["description"] = (generic or texts)[0]

    for match in RE_GATED.finditer(blob):
        if match.group("name") in commands:
            commands[match.group("name")]["gated"] = True

    # An alias is not a command in its own right; drop any name we also saw as an alias.
    aliases = {a for entry in commands.values() for a in entry["aliases"]}
    return {name: entry for name, entry in commands.items() if name not in aliases}


def categorize(commands: dict[str, dict]) -> list[tuple[str, list[str]]]:
    """Group commands, appending anything new to an 'Uncategorized' bucket.

    New releases add commands, so the leftover bucket is the interesting one to read.
    """
    assigned: set[str] = set()
    grouped: list[tuple[str, list[str]]] = []
    for title, names in CATEGORIES:
        present = [n for n in names if n in commands]
        assigned.update(present)
        if present:
            grouped.append((title, present))
    leftover = sorted(set(commands) - assigned)
    if leftover:
        grouped.append(("Uncategorized (new since this script was written)", leftover))
    return grouped


def format_text(commands: dict[str, dict]) -> str:
    lines = []
    for title, names in categorize(commands):
        lines.append(f"\n## {title}")
        for name in names:
            entry = commands[name]
            head = "/" + name
            if entry["aliases"]:
                head += " (" + ", ".join("/" + a for a in entry["aliases"]) + ")"
            flag = "  [conditional]" if entry["gated"] else ""
            desc = entry["description"] or "(runtime description)"
            lines.append(f"  {head:<44} {desc}{flag}")
    return "\n".join(lines)


def format_md(commands: dict[str, dict]) -> str:
    lines = []
    for title, names in categorize(commands):
        lines.append(f"\n### {title}\n")
        lines.append("| Command | Aliases | Description |")
        lines.append("|---|---|---|")
        for name in names:
            entry = commands[name]
            aliases = ", ".join(f"`/{a}`" for a in entry["aliases"]) or "—"
            # A literal pipe in a description (e.g. "default | fullscreen") would end
            # the table cell, silently mangling the row.
            desc = entry["description"].replace("|", "\\|") or "_(computed at runtime)_"
            if entry["gated"]:
                desc += " **[conditional]**"
            lines.append(f"| `/{name}` | {aliases} | {desc} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, help="path to a Claude Code binary or cli.js")
    parser.add_argument("--format", choices=["text", "md", "json"], default="text")
    parser.add_argument(
        "--diff", nargs=2, metavar=("OLD", "NEW"), type=Path,
        help="compare two builds and print added/removed commands",
    )
    args = parser.parse_args()

    if args.diff:
        old, new = (extract(p) for p in args.diff)
        added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
        print(f"{args.diff[0].name} -> {args.diff[1].name}")
        print(f"  {len(old)} -> {len(new)} commands")
        for name in added:
            print(f"  + /{name}  {new[name]['description']}")
        for name in removed:
            print(f"  - /{name}")
        if not added and not removed:
            print("  (no command added or removed)")
        return

    binary = args.binary or find_binary()
    commands = extract(binary)
    if args.format == "json":
        print(json.dumps({"binary": binary.name, "commands": commands}, indent=2, ensure_ascii=False))
    else:
        print(f"# {len(commands)} slash commands defined in {binary.name}")
        print(format_md(commands) if args.format == "md" else format_text(commands))


if __name__ == "__main__":
    main()
