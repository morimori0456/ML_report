---
title: "Claude Code Slash Commands — Recovering the Full Command Registry from the Bundle"
description: "Why /help shows fewer commands than the build contains, how to extract the complete registry from the shipped binary, and the full v2.1.225 inventory."
---

> A companion script, [extract_slash_commands.py](extract_slash_commands.py), recovers the slash-command registry from an installed Claude Code build and prints it as text, Markdown, or a diff between two versions. This document explains why that is necessary, how the extraction works, and what the v2.1.225 registry actually contains.

`/help` answers the question "what can I type right now?" — which is not the same question as "what does this build support?". Commands are gated by plan, platform, session type, authentication mode, and feature flags, so a large fraction of the registry is invisible on any given machine. That gap matters if you are building loops and automations on top of Claude Code: a command like `/subtask`, `/goal`, or `/tasks` changes what a loop can delegate and supervise, and discovering it a release late is discovering it after you have already hand-rolled a worse version. The registry survives in the shipped bundle as string literals, so the complete list is recoverable with about a hundred lines of regex — and the diff between two installed versions is a precise, zero-guesswork changelog of the command surface.

---

## Table of Contents
1. [Three Sources of Slash Commands](#1-three-sources-of-slash-commands)
2. [Why `/help` Is an Incomplete Answer](#2-why-help-is-an-incomplete-answer)
3. [Anatomy of the Bundle](#3-anatomy-of-the-bundle)
4. [Building the Extractor](#4-building-the-extractor)
5. [The v2.1.225 Inventory](#5-the-v21225-inventory)
6. [Diffing Two Releases](#6-diffing-two-releases)
7. [Adding Your Own Commands](#7-adding-your-own-commands)
8. [Command vs Skill vs Plugin vs Subagent](#8-command-vs-skill-vs-plugin-vs-subagent)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. Three Sources of Slash Commands

Anything typed as `/name` resolves through one of three mechanisms, and they have different discovery rules. Conflating them is the reason "the list of slash commands" feels unknowable.

| Source | Where it lives | Discovered when | Appears in the bundle? |
|---|---|---|---|
| Built-in command | Compiled into the binary | Startup, from a static registry | Yes — as object literals |
| Bundled skill | Shipped inside the binary as instruction text | Startup | Partially — registration call only |
| On-disk skill / custom command / plugin | `~/.claude/commands/`, `.claude/skills/`, installed plugins | Startup, by scanning the filesystem | No |

Only the first is fully recoverable from the binary. Bundled skills appear inconsistently, because a skill is registered by a helper call whose payload is instruction text rather than a command descriptor — `/security-review` inlines a static description and is recoverable, while `/init` computes its description at runtime and yields only a name. On-disk commands are invisible by construction: they are files on your machine, so `/skills` and `/help` are the only authorities for those.

### Why this matters

The registry extraction below is therefore a **superset for built-ins and a subset for skills**. Use it to discover built-in commands you did not know existed and to diff releases; use `/help` and `/skills` for what is live in your session. The two are complements, not competitors.

---

## 2. Why `/help` Is an Incomplete Answer

Every command object carries optional gating callbacks. Two are load-bearing:

```javascript
// Reconstructed shape — variable names are minified and change every release.
{
  type: "local-jsx",
  name: "session",
  aliases: ["remote"],
  description: "Show cloud session URL and QR code",
  isEnabled: () => Wa(),                  // omitted from /help entirely when false
  get isHidden() { return !X4("fanout") } // typable, but not listed
}
```

`isEnabled: () => false` removes a command from the menu; `isHidden` keeps it working but unlisted. In v2.1.225, **58 of 119 commands (49%) carry at least one gate**. The conditions include:

| Gate condition | Example commands | Consequence |
|---|---|---|
| Plan / subscription | `/upgrade`, `/usage-credits`, `/passes` | Hidden on plans where they are meaningless |
| Session origin (local / cloud / background) | `/stop`, `/session`, `/background`, `/workflow-launch-exec` | A background session exposes a different menu than your terminal |
| Host application | `/desktop`, `/model` (has a separate FleetView definition) | Same name, different implementation and description per host |
| Feature flag | `/radio`, `/session` (`X4("fanout")`) | Server-side rollout, invisible locally |
| Authentication mode | `/setup-bedrock`, `/setup-vertex`, `/login` | Bedrock/Vertex users see a different set than OAuth users |
| Hard-disabled | `/wellbeing` (`isEnabled: () => false`) | Present in the build, reachable by nobody |
| Environment variable | `/doctor` (`!DISABLE_DOCTOR_COMMAND`) | Enterprise deployments can strip commands |

The practical consequence: **a colleague's `/help` and yours are legitimately different**, and a command missing from your menu is more often gated than absent. Before concluding a feature does not exist, check the registry.

### Why this matters

Gating explains the whole confusion. `/help` is correct but local; the bundle is complete but unconditional. Reading them together tells you both what exists and why you cannot see it.

---

## 3. Anatomy of the Bundle

The native install keeps one self-contained executable per downloaded version:

```
~/.local/share/claude/versions/
├── 2.1.187      # ~235 MB
├── 2.1.220      # ~275 MB
└── 2.1.225      # ~298 MB   <- newest by mtime
```

Each is a single-file JavaScript bundle embedded in a runtime. Minification renames every identifier, hoists declarations, and collapses whitespace — but **string literals are preserved verbatim**, because they are data. The command registry is built from object literals, so each command leaves an intact fingerprint:

```javascript
// The three shapes that matter, all recoverable:
{type:"local-jsx",name:"permissions",aliases:["allowed-tools"],description:"Manage allow and deny tool permission rules"}

Ru({name:"doctor",aliases:["checkup"],isEnabled:()=>!te.DISABLE_DOCTOR_COMMAND,survivesBundledKillSwitch:!0,requires:{workspace:!0}})

{type:"prompt",name:"init",get description(){return _r(process.env.CLAUDE_CODE_NEW_INIT)?"Initialize new CLAUDE.md file(s)…":"…"}}
```

This is why the approach is regex over parsing. A JavaScript parser would give an AST whose variable names (`Ru`, `te`, `_r`) are meaningless and different next release; the literals are the stable part. Parsing buys nothing and costs a dependency.

### The discrimination problem

A naive `name:"…",description:"…"` search over a 298 MB blob is useless: the same shape occurs in tool schemas, MCP definitions, npm package blurbs, and the bundled highlight.js language definitions. An early pass of this script returned 156 "commands", of which 48 were computer-use and browser tool schemas (`left_click`, `read_page`, `computer_batch`) and one was a `highlight.js` grammar for the Pacemaker cluster shell (`crmsh`, with `aliases:["crm","pcmk"]` — command-shaped and entirely unrelated).

Three filters bring it down to a clean 119:

| Filter | Rule | What it removes |
|---|---|---|
| Naming convention | Command names are kebab-case; no underscores | The snake_case computer-use / Chrome tool family |
| Positive markers | Require a command-only sibling key nearby | Tool schemas, package metadata |
| Negative markers | Reject anything with `input_schema` / `inputSchema` | Tools whose names happen to be kebab-case |

The positive markers are keys that only a command object carries: `type:"local"` / `"local-jsx"` / `"prompt"`, `aliases:[`, `menuDescription:`, `argumentHint:`, `supportsNonInteractive:`, `progressMessage:`, `thinClientDispatch:`, `fleetHostCall:`, `survivesBundledKillSwitch:`, `userFacingName`. Requiring one of them inside a ±260-character window around the name is what separates signal from noise. After all three filters, exactly two hand-verified names survive that are still not commands (`function`, `callback`, both from a type-discriminator object), so the script carries a four-entry denylist.

### Why this matters

The bundle is minified, not obfuscated. Anything the program must display to a human — command names, descriptions, argument hints — is sitting there in plain text. The engineering problem is not extraction, it is discrimination: deciding which of thousands of similarly shaped literals are the ones you asked for.

---

## 4. Building the Extractor

The whole script is standard library only and runs in about half a second on a 298 MB binary. Four details are worth calling out, because each one was a bug first.

### 4.1 Decode as latin-1, read whole

```python
blob = binary.read_bytes().decode("latin-1")
```

`latin-1` maps every possible byte to a code point, so it never raises on the arbitrary binary sections around the JavaScript, and it leaves ASCII — all the regexes look at — untouched. Reading the file whole rather than in chunks avoids matches being split across a boundary; 300 MB of `str` is affordable, a silently truncated inventory is not.

### 4.2 Descriptions come from two keys, and there can be several

Commands registered through the internal helper often hold their `description` in a variable and only inline a `menuDescription`. Both must be read, with `menuDescription` as the fallback. Worse, some commands are **defined twice** — once for the terminal, once for an alternate host — and the naive "first match wins" picks the wrong one:

```
name:"model",description:"Set model for this FleetView session (not persisted)"
name:"model",supportsNonInteractive:!0,description:"Set the AI model for Claude Code",argumentHint:"<model>"
```

The first is real but host-specific and misleading in a general reference. The script collects every candidate and prefers one that is not host-qualified.

Key order is not guaranteed either. A handful of objects — `/rewind`, `/release-notes` — put `description` *before* `name`, so a single name-then-description pattern reports them as having no description at all. After adding the reversed pattern, 11 of 119 commands remain description-less, and every one of those genuinely computes it at runtime (`/fast` interpolates the current mode, `/doctor` and `/login` vary by auth state).

### 4.3 An alias is not a command

`/cost` and `/stats` are aliases of `/usage`; `/checkpoint` and `/undo` are aliases of `/rewind`. Listing them as separate entries inflates the count and implies capability that is not there. The fix is one line after collection:

```python
aliases = {a for entry in commands.values() for a in entry["aliases"]}
return {name: entry for name, entry in commands.items() if name not in aliases}
```

### 4.4 `^` and `.match(string, pos)` do not mean what you want

To grab the aliases that follow a name, the script anchors a second pattern at the position where the name match ended:

```python
RE_ALIASES = re.compile(r',aliases:\[(?P<aliases>[^\]]*)\]')   # correct
RE_ALIASES = re.compile(r'^,aliases:\[(?P<aliases>[^\]]*)\]')  # silently matches nothing
```

`pattern.match(blob, pos)` already anchors at `pos`. Adding `^` demands *start of line as well* — and since the bundle contains newlines, `pos` almost never is one. The symptom is not an error: every command simply reports zero aliases, and the output looks plausible. This is the failure mode to fear in any regex-over-blob work, and the reason the script's real check is the next section's cross-validation against `/help`.

### Why this matters

Every one of these four is a silent-wrong-answer bug, not a crash. Extraction code that cannot fail loudly needs an external ground truth to check against — here, the live `/help` menu and the fact that categories with a known membership stay full.

---

## 5. The v2.1.225 Inventory

Generated by the companion script; 119 commands after de-aliasing. `[conditional]` marks a command carrying an `isEnabled` or `isHidden` gate — present in the build, not necessarily in your menu. *(computed at runtime)* means the description is a function rather than a literal, so only the name is recoverable.

```bash
python extract_slash_commands.py --format md
```

### Session & conversation

| Command | Aliases | Description |
|---|---|---|
| `/help` | — | Show help and available commands |
| `/clear` | — | Start a new session with empty context; previous session stays on disk (resumable with /resume) |
| `/compact` | — | Free up context by summarizing the conversation so far **[conditional]** |
| `/autocompact` | — | Set how full the context gets before auto-summarizing **[conditional]** |
| `/context` | — | Visualize current context usage as a colored grid **[conditional]** |
| `/resume` | — | Resume a previous conversation |
| `/branch` | — | Create a branch of the current conversation at this point |
| `/fork` | — | Spawn a background agent that inherits the full conversation **[conditional]** |
| `/rename` | `/name` | Rename the current conversation **[conditional]** |
| `/recap` | — | Generate a one-line session recap now |
| `/export` | — | Export the current conversation to a file or clipboard |
| `/copy` | — | Copy Claude's last response to clipboard (or /copy N for the Nth-latest) |
| `/diff` | — | View uncommitted changes and per-turn diffs |
| `/rewind` | `/checkpoint`, `/undo` | Restore the code and/or conversation to a previous point |
| `/exit` | `/quit` | _(computed at runtime)_ |

### Model & how work gets done

| Command | Aliases | Description |
|---|---|---|
| `/model` | — | Set the AI model for Claude Code **[conditional]** |
| `/effort` | — | Set effort level for model usage |
| `/fast` | — | _(computed at runtime)_ |
| `/advisor` | — | Let Claude consult a stronger model at key moments |
| `/plan` | — | Enable plan mode or view the current session plan |
| `/plan-artifact` | — | Publish a plan as a shareable Artifact **[conditional]** |
| `/goal` | — | Set a goal Claude checks before stopping **[conditional]** |
| `/btw` | — | Ask a quick side question without interrupting the main conversation |
| `/subtask` | — | Send a subagent off with your full context; its result comes back here **[conditional]** |
| `/batch` | — | Plan a large change; background agents each open a PR |
| `/list-agents` | `/peers` | List subagents and other Claude sessions you can message **[conditional]** |
| `/agents` | — | (removed) Ask Claude to create/manage subagents, or edit .claude/agents/ |

### Configuration

| Command | Aliases | Description |
|---|---|---|
| `/config` | `/settings` | Open settings **[conditional]** |
| `/permissions` | `/allowed-tools` | Manage allow and deny tool permission rules |
| `/hooks` | — | View hook configurations for tool events |
| `/add-dir` | — | Add a new working directory |
| `/cd` | — | Move this session to a new working directory |
| `/memory` | — | Open a memory file in your editor |
| `/pause-memory` | `/memory-pause`, `/toggle-memory` | Pause automemory for this session **[conditional]** |
| `/keybindings` | — | Open your keyboard shortcuts file **[conditional]** |
| `/terminal-setup` | — | _(computed at runtime)_ |
| `/statusline` | — | _(computed at runtime)_ |
| `/theme` | — | Change the theme |
| `/color` | — | Set the prompt bar color for this session |
| `/tui` | — | Set the terminal UI renderer (default \| fullscreen) |
| `/focus` | — | Toggle focus view: just your prompt, summary, and response |
| `/scroll-speed` | — | Adjust mouse wheel scroll speed **[conditional]** |
| `/brief` | — | Toggle brief-only mode **[conditional]** |
| `/voice` | — | Toggle voice mode **[conditional]** |
| `/import` | — | Import config from another AI coding agent **[conditional]** |
| `/privacy-settings` | — | View and update your privacy settings **[conditional]** |
| `/auto-mode-setup` | — | Set up and customise auto mode — environment context, plus optional rule tweaks **[conditional]** |
| `/setup-bedrock` | — | Reconfigure Amazon Bedrock authentication, region, or model pins **[conditional]** |
| `/setup-vertex` | — | Reconfigure Google Vertex AI authentication, project, region, or model pins **[conditional]** |
| `/wellbeing` | `/breaks`, `/break-reminder`, `/downtime` | Configure optional break reminders and quiet-hours nudges **[conditional]** |

### Extensibility (MCP / plugins / skills)

| Command | Aliases | Description |
|---|---|---|
| `/mcp` | — | Manage MCP servers **[conditional]** |
| `/plugin` | `/plugins`, `/marketplace` | Manage Claude Code plugins |
| `/reload-plugins` | — | Activate pending plugin changes in the current session |
| `/skills` | — | List available skills |
| `/reload-skills` | — | Pick up skills added or changed on disk during this session |
| `/skill-doctor` | — | Show which loaded skills are unused and costing context **[conditional]** |
| `/run-skill-generator` | — | Create a skill that knows how to run this project’s app |
| `/claude-in-chrome` | — | Let Claude browse and interact with pages in your Chrome |
| `/powerup` | — | Discover Claude Code features through quick interactive lessons |
| `/insights` | — | Generate a report analyzing your Claude Code sessions |
| `/team-onboarding` | — | Help teammates ramp on Claude Code with a guide from your usage **[conditional]** |

### Bundled skills (invocable as /name)

| Command | Aliases | Description |
|---|---|---|
| `/init` | — | _(computed at runtime)_ |
| `/run` | — | Launch this project’s app to see your change working |
| `/loop` | — | Repeat a prompt or command on an interval (e.g. /loop 5m /foo) |
| `/security-review` | — | Complete a security review of the pending changes on the current branch |
| `/update-config` | — | Change settings: hooks, permissions, environment variables |
| `/fewer-permission-prompts` | — | _(computed at runtime)_ |
| `/claude-api` | — | Build and debug apps that use the Claude API |
| `/design-sync` | — | Push your design system components to claude.ai/design **[conditional]** |

### Remote, cloud & background

| Command | Aliases | Description |
|---|---|---|
| `/remote-control` | `/rc` | Control this session from your phone or claude.ai/code |
| `/teleport` | `/tp` | Resume a Claude Code session from claude.ai **[conditional]** |
| `/session` | `/remote` | Show cloud session URL and QR code **[conditional]** |
| `/web-setup` | — | Set up Claude Code on the web with your GitHub account **[conditional]** |
| `/remote-env` | — | Choose the default environment for cloud agents **[conditional]** |
| `/schedule` | `/routines` | Create and manage scheduled remote Claude Code agents |
| `/daemon` | — | Manage background services and routines |
| `/tasks` | `/bashes` | View and manage everything running in the background |
| `/background` | `/bg` | Send this session to the background and free the terminal **[conditional]** |
| `/stop` | — | Stop this background session; transcript and worktree are kept **[conditional]** |
| `/loops` | — | List, create, and delete loops **[conditional]** |
| `/workflows` | — | Browse running and completed workflows **[conditional]** |
| `/artifacts` | — | Browse your published and shared artifacts **[conditional]** |
| `/desktop` | `/app` | Continue the current session in Claude Desktop **[conditional]** |
| `/mobile` | `/ios`, `/android` | Show QR code to download the Claude mobile app |
| `/ultraplan` | — | Claude Code on the web drafts a plan you can edit and approve |
| `/ultrareview` | — | Find and verify bugs in your branch using Claude Code on the web |
| `/autofix-pr` | — | Monitor and autofix any issues with the current PR **[conditional]** |
| `/install-github-app` | — | Set up Claude GitHub Actions for a repository **[conditional]** |
| `/install-slack-app` | — | Install the Claude Slack app |
| `/design` | — | Grant or revoke Claude agent access to your Design projects **[conditional]** |
| `/design-login` | — | Authorize design-system access for /design-sync with your claude.ai account **[conditional]** |
| `/design-revoke` | — | Revoke Claude agent access to your Design projects **[conditional]** |
| `/design-consent` | — | Grant Claude agent access to your Design projects **[conditional]** |
| `/workflow-launch-exec` | — | Execute a server-launched workflow handoff (workflow_launch event sessions only) **[conditional]** |

### Account, usage & diagnostics

| Command | Aliases | Description |
|---|---|---|
| `/status` | — | Show Claude Code status including version, model, account, API connectivity, and tool statuses |
| `/usage` | `/cost`, `/stats` | Show session cost, plan usage, and activity stats **[conditional]** |
| `/usage-credits` | — | Configure usage credits or request them from your admin when you hit a limit **[conditional]** |
| `/extra-usage` | — | Renamed to /usage-credits **[conditional]** |
| `/upgrade` | — | Upgrade to Max for higher rate limits and more Opus **[conditional]** |
| `/passes` | — | _(computed at runtime)_ |
| `/login` | — | _(computed at runtime)_ |
| `/logout` | — | Sign out from your Anthropic account **[conditional]** |
| `/version` | — | Show this session's version (autoupdate may have a newer one) **[conditional]** |
| `/update` | `/restart` | Switch to the latest version (conversation continues) **[conditional]** |
| `/install` | — | Install Claude Code native build |
| `/release-notes` | — | View release notes |
| `/doctor` | `/checkup` | _(computed at runtime)_ **[conditional]** |
| `/ide` | — | Manage IDE integrations and show status |
| `/chrome` | — | Open Claude in Chrome settings **[conditional]** |
| `/bug` | — | Report a bug or share your conversation |
| `/feedback` | — | Send feedback to Anthropic or report a bug |
| `/explain-usage` | — | _(computed at runtime)_ **[conditional]** |
| `/setup-cowork` | — | _(computed at runtime)_ **[conditional]** |
| `/debug` | — | Turn on debug logging and investigate problems |
| `/stickers` | — | Order Claude Code stickers |
| `/radio` | — | Listen to Claude FM lo-fi radio **[conditional]** |
| `/heapdump` | — | Dump the JS heap to ~/Desktop **[conditional]** |
| `/pro-trial-expired` | — | Options shown when the Pro plan Claude Code trial has ended **[conditional]** |
| `/rate-limit-options` | — | Show options when rate limit is reached **[conditional]** |

---

## 6. Diffing Two Releases

Keeping several versions installed turns the extractor into a changelog for the command surface:

```bash
python extract_slash_commands.py --diff ~/.local/share/claude/versions/2.1.220 \
                                        ~/.local/share/claude/versions/2.1.225
```

```
2.1.220 -> 2.1.225
  121 -> 119 commands
  + /list-agents  List subagents and other Claude sessions you can message
  - /review
  - /whiteboard
  - /workshop
```

Over a longer span the shape of the roadmap becomes visible:

```
2.1.187 -> 2.1.225
  108 -> 119 commands
  + /artifacts, /plan-artifact          # publishing work products
  + /subtask, /list-agents, /skill-doctor  # multi-agent + skill hygiene
  + /auto-mode-setup, /setup-cowork, /explain-usage
  + /design-consent, /design-revoke, /import
  - /init-verifiers, /review
```

Read as a signal, the additions cluster: agent-to-agent delegation (`/subtask`, `/list-agents`), skill economics (`/skill-doctor`), and shareable output (`/artifacts`, `/plan-artifact`).

### The caveat that keeps this honest

A diff reports *what the extractor can see*, not ground truth. If a command's object literal changes shape between releases — gaining a computed description, losing a marker key — it appears as a removal and an addition even though nothing about the command changed. In the 2.1.187 → 2.1.225 diff above, `/bug` shows up as added; it existed in 2.1.187 with a shape the marker filter did not match. Treat single-command deltas as leads to verify, and trust the aggregate.

### Why this matters

Release notes describe features; this describes the interface. For anyone scripting Claude Code, the disappearance of `/review` or the arrival of `/subtask` is an API change, and this is the cheapest way to notice one.

---

## 7. Adding Your Own Commands

A Markdown file in a commands directory becomes a slash command with no registration step. The filename is the command name.

```
~/.claude/commands/add-report.md      ->  /add-report        (all projects)
<project>/.claude/commands/deploy.md  ->  /deploy            (this project only)
```

Frontmatter is optional; all five fields below are:

```markdown
---
description: Review a PR for quality and security
argument-hint: [pr-number] [priority]
allowed-tools: Read, Grep, Bash(gh pr *)
model: sonnet
disable-model-invocation: false
---

Review pull request #$1 with priority $2.

Diff to review: !`gh pr diff $1`
Project conventions: @CLAUDE.md
```

Three substitutions do most of the work:

| Syntax | Expands to | Note |
|---|---|---|
| `$ARGUMENTS` | Everything typed after the command name | Whole string, unsplit |
| `$1`, `$2`, `$3` | Positional arguments | Splits on whitespace |
| `@path/to/file` | Contents of that file | Resolved before the prompt is sent |
| `` !`shell command` `` | Output of that command | Runs at expansion time, subject to `allowed-tools` |

`allowed-tools` narrows permissions for the command's duration — worth setting on anything that touches `Bash`, since it converts a blanket approval into a scoped one. `model: sonnet` on a mechanical command keeps the expensive model for work that needs judgment.

### Why this matters

The interesting commands on a mature setup are the local ones. A command file is a prompt with a stable name, which is exactly the unit a loop can call repeatedly — and unlike a skill, it is invoked deterministically, by you, at a moment of your choosing.

---

## 8. Command vs Skill vs Plugin vs Subagent

These four extension points overlap enough to be chosen wrongly. The distinction that matters is **who decides to invoke it** and **whose context pays**.

| Mechanism | Invoked by | Context cost | Use when |
|---|---|---|---|
| Custom command (`commands/*.md`) | You, explicitly | Prompt text, only when called | A repeatable prompt you want a name for |
| Skill (`skills/<name>/SKILL.md`) | The model, on relevance — or you, via `/name` | Description always loaded; body on use | Procedural knowledge the model should apply unprompted |
| Plugin | Ships commands, skills, hooks, MCP servers together | Whatever it contains | Distributing a workflow to a team |
| Subagent (`.claude/agents/*.md`) | The model, or you via `/subtask` | Separate context window; only its report returns | Fan-out search or verification that would flood your context |

Two consequences worth internalizing. First, **skill descriptions are always in context** — a dozen installed skills is a standing token cost, which is why `/skill-doctor` exists to name the ones not earning it. Second, a subagent's isolation is the point: work that reads fifty files and reports three lines belongs in one, and `/subtask` hands it your full context to start from.

### Why this matters

The default instinct — write a command — is right for repeatable prompts and wrong for knowledge the model should reach for on its own. Picking the wrong mechanism produces either a command nobody remembers to type or a skill burning context on every unrelated turn.

---

## 9. Common Pitfalls

**Trusting `/help` as the complete list.** It is the complete list *for your session*. Two-thirds of commands are gated, so absence from the menu is weak evidence of absence from the build.

**Trusting the extractor as the complete list.** Symmetrically wrong: it cannot see on-disk skills, custom commands, or installed plugins, because those are not in the binary. `/help` plus `/skills` covers what it misses.

**Reading a bare `name`/`description` pair as a command.** The bundle contains thousands of them from tool schemas and vendored libraries. Without a positive marker key, a hit means nothing — this is how a Pacemaker shell grammar ends up in your command list.

**Regex `^` inside `.match(string, pos)`.** Anchors at start-of-line, not at `pos`, and fails silently. Any extraction that can return a plausible-but-empty field needs a ground-truth check.

**Assuming the newest directory entry is the running version.** Autoupdate downloads a new build while your session keeps running the old one. `/version` reports what the session is executing; the newest file on disk is what the next session will start. Diff the file you actually ran.

**Hardcoding a version path into tooling.** `~/.local/share/claude/versions/2.1.225` is stale within days. Resolve by mtime, or follow `which claude`.

**Treating a diff delta as ground truth.** Object-literal shape changes produce phantom additions and removals. Verify a single-command delta before acting on it.

**Assuming aliases are separate commands.** `/cost`, `/stats`, `/checkpoint`, `/undo`, `/bg`, `/tp`, `/rc` are all aliases. Counting them inflates the inventory by roughly a quarter.

---

## 10. References

- Claude Code documentation — [Slash commands](https://docs.claude.com/en/docs/claude-code/slash-commands)
- Claude Code documentation — [Skills](https://docs.claude.com/en/docs/claude-code/skills)
- Claude Code documentation — [Plugins](https://docs.claude.com/en/docs/claude-code/plugins) and [Plugin marketplaces](https://docs.claude.com/en/docs/claude-code/plugin-marketplaces)
- Claude Code documentation — [Subagents](https://docs.claude.com/en/docs/claude-code/sub-agents)
- Claude Code documentation — [Settings](https://docs.claude.com/en/docs/claude-code/settings) (permissions, hooks, environment variables)
- `plugin-dev` skill, official plugin marketplace — command frontmatter and argument-substitution reference, installed at `~/.claude/plugins/marketplaces/claude-plugins-official/`
- Companion tooling in this repository: [loop_engineering.md](loop_engineering.md) for where commands sit in a loop architecture, and [loop_design_playbook.md](loop_design_playbook.md) for the operations side
