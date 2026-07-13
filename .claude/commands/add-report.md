---
description: Add a new ML report (md + executed ipynb) to this repository following the established conventions
---

You are helping add a new report to the ML_report repository. Follow these steps exactly.

## Step 1 — Clarify scope (if not already given)

Ask for:
- **Topic**: what concept or paper are we covering?
- **Category**: which top-level directory? (distillation / llm / autonomous_driving / ema / experiment_tracking / infrastructure — or new one?)
- **Notebook dependency level**: core (numpy/matplotlib only) | transformer (torch + sklearn) | llm-gpu (CUDA)
- **Kaggle GPU verification**: does this topic need proof that its pipeline actually runs on a real CUDA GPU? Default yes when dependency level is `llm-gpu`; otherwise only if the user asks. Skip for core/transformer/llm-cpu topics — Step 4 already executes and verifies those locally.

## Step 2 — Create the markdown report

File: `<category>/<snake_case_topic>.md`

**MD structure** (copy this template exactly):

```markdown
# <Title> — <Subtitle>

> <1-2 sentence lead: what this doc is and what companion notebook to see>

<1-paragraph motivation: why does this topic matter? What problem does it solve?>

---

## Table of Contents
1. [Section 1](#1-section-1)
2. ...

---

## 1. <Section>

<Theory with LaTeX math in $$ blocks>

### Key insight
> **<Callout>**: ...

<Comparison table if relevant>

| Concept | Description | When to use |
|---|---|---|

<Code snippet in Python/PyTorch if relevant>
```python
# ...
```

<"Why this matters" summary at end of each major section>
```

**Style rules**:
- Use `$$...$$` for display math (GitHub renders it)
- Keep section summaries to 2-3 sentences — concrete, not vague
- Always include a comparison table for methods
- Always include a "Common Pitfalls" section near the end
- References section last, with arXiv links
- DO NOT use emojis

## Step 3 — Generate the notebook

Create a Python script in the scratchpad that uses `nbformat` to generate the `.ipynb` file:

```python
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
# Title cell (markdown)
cells.append(new_markdown_cell("# Title\n\n> lead sentence\n\nSee [report.md](report.md)."))

# Imports cell — IMPORTANT: never include matplotlib.use('Agg'); Jupyter handles the backend
cells.append(new_code_cell("""\
import numpy as np
import torch
import matplotlib.pyplot as plt
# etc.
"""))

# ... more cells following the md structure

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

with open('/home/jetson/w/ML_report/<category>/<topic>.ipynb', 'w') as f:
    nbformat.write(nb, f)
```

**Notebook conventions**:
- Every markdown section header in the md should have a corresponding notebook section
- Title cell links to the `.md` companion
- Imports cell: no `matplotlib.use('Agg')` — it causes FigureCanvasAgg UserWarning in Jupyter
- Print key results (accuracy, loss values) at the end of each experiment cell
- End with a "Takeaways" markdown cell summarizing the main findings
- Use `plt.tight_layout()` before `plt.show()` on every plot

## Step 4 — Execute the notebook

```bash
cd /home/jetson/w/ML_report
uv run python <scratchpad_script>.py   # generates the .ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  <category>/<topic>.ipynb
```

Verify execution succeeded: check that the output contains no `Error` or `Traceback` in stderr.

## Step 5 — GPU verification via Kaggle (only when triggered in Step 1)

Skip this entire step for core/transformer/llm-cpu topics. When triggered, this produces a **second**, separate notebook — `<category>/<topic>_gpu_smoke_kaggle.ipynb` — that proves the pipeline scales to a real CUDA GPU. It is not a copy of the Step 3 notebook: use a real-sized/instruct-capable base model for the topic (e.g. an instruct LLM for fine-tuning topics, not a CPU toy model), and encode every check as a hard `assert` so a green run is self-verifying, not just eyeballed.

### 5a. Prerequisites (check once, don't repeat if already satisfied)
- `kaggle` CLI installed (`uv tool install kaggle` if missing).
- `~/.kaggle/access_token` contains a valid API token (test with `kaggle kernels list --mine`). If missing, ask the user to generate one at kaggle.com/settings/api and save the raw token to that file with `chmod 600`. **Do not use the legacy `kaggle.json` format for this** — it does not authenticate `kernels push` in current CLI versions; only the `access_token` file does.
- The Kaggle account username needed for kernel slugs: read it off any existing kernel in `kaggle kernels list --mine` output. Never hardcode a specific username in this file — it is committed to a public repo.

### 5b. Write the notebook (same nbformat approach as Step 3)
Build it in scratchpad first, matching the md's structure. Requirements specific to the GPU leg:
- An environment cell that prints GPU name, compute capability, and library versions.
- **bf16 decision must use `torch.cuda.get_device_capability() >= (8, 0)`, never `torch.cuda.is_bf16_supported()`** — the latter returns `True` on pre-Ampere GPUs (software emulation) and silently gives you the wrong dtype.
- An `assert torch.cuda.get_device_capability() >= (7, 0)` guard as a second line of defense against being handed an unsupported old GPU.
- Assert-based checkpoints at every stage that matters (loss dropped, VRAM under some ceiling, eval metric above some threshold, saved artifact reloads and still works) — the point is a green run means "safe to scale to the production GPU," not "looked fine."
- A closing cell that prints what changes when moving off the free GPU (precision, library-version pins, batch size, attention kernel, etc).

### 5c. kernel-metadata.json (write alongside the notebook in the same scratch folder)
```json
{
  "id": "<kaggle_username>/<topic>-gpu-smoke",
  "title": "<Topic> GPU Smoke",
  "code_file": "<topic>_gpu_smoke_kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```
Keep `title` resolving to the same slug as `id` (e.g. via matching words) or the push emits a non-fatal warning.

### 5d. Push with an explicit accelerator, then poll to a terminal state
```bash
cd <scratch_dir_with_notebook_and_kernel-metadata.json>
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```
**Always pass `--accelerator NvidiaTeslaT4` explicitly.** Left to Kaggle's default you can be handed an NvidiaTeslaP100 kernel — current PyTorch/bitsandbytes wheels have dropped `sm_60` support, and the kernel dies with a bitsandbytes `Error named symbol not found`.

Poll in the background (this can take several minutes):
```bash
while true; do
  s=$(kaggle kernels status <kaggle_username>/<topic>-gpu-smoke 2>&1)
  case "$s" in
    *RUNNING*|*QUEUED*) sleep 60 ;;
    *) echo "terminal status: $s"; break ;;
  esac
done
```

### 5e. Retrieve and verify
```bash
kaggle kernels output <kaggle_username>/<topic>-gpu-smoke -p <out_dir>
```
This downloads `<out_dir>/<topic>-gpu-smoke.log` (a JSON list of `{stream_name, data}` events — not a plain-text log). Read it and confirm no `Error`/`Traceback` in stderr and every assert-guarded checkpoint printed its success line. On status `ERROR` the log is still downloadable and is the only way to see what broke — always fetch it before retrying.

### 5f. Bake the real run's outputs into the committed notebook
`kaggle kernels pull` returns source only, never outputs — outputs must be reconstructed from the log:
1. Load the pushed source notebook and the log JSON.
2. Strip ANSI escape codes and pip progress-bar lines from stdout.
3. Split stdout into per-cell buckets using a distinctive first line of each cell's expected output as a marker; assert every code cell receives at least one output line (catches silent misalignment between cells and log).
4. Set each cell's `outputs` to a `nbformat.v4.new_output("stream", name="stdout", text=...)` and its `execution_count` to its position.
5. Append a short **Provenance** note to the title cell: GPU model, kernel slug, run date, and the exact `kaggle kernels push` command to reproduce it.
6. Write the result to `<category>/<topic>_gpu_smoke_kaggle.ipynb` in the repo — this baked version is the committed artifact, not the plain notebook from 5b.

## Step 6 — Update README.md

Four places to update (the last only if Step 5 ran):

### A. Notebook dependency table (around line 42–56)
Add a row:
```
| `<category>/<topic>.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
```
If Step 5 ran, also add:
```
| `<category>/<topic>_gpu_smoke_kaggle.ipynb` | Kaggle GPU (T4) via Kaggle API — not run locally; contains outputs from an actual run |
```

### B. Directory structure block (around line 62–104)
Add the new files under the correct category:
```
│   ├── <topic>.md   # <one-line description>
│   └── <topic>.ipynb  # <one-line description>
```
If Step 5 ran, also add the `_gpu_smoke_kaggle.ipynb` file with its own one-line description.

### C. Report list table (section matching the category)
Add a row:
```
| <Title> | <Topics comma-separated> | [<topic>.md](<category>/<topic>.md) + [demo](<category>/<topic>.ipynb) |
```
If Step 5 ran, append the GPU demo to the same cell rather than adding a new row:
```
| <Title> | <Topics comma-separated> | [<topic>.md](<category>/<topic>.md) + [demo](<category>/<topic>.ipynb) / [Kaggle T4 GPU smoke test](<category>/<topic>_gpu_smoke_kaggle.ipynb) |
```

### D. md report cross-link (only if Step 5 ran)
In the `.md` report's lead paragraph, add a one-line pointer to the GPU notebook as the "GPU leg" of the CPU experiment.

## Step 7 — Commit and push

```bash
cd /home/jetson/w/ML_report
git add <category>/<topic>.md <category>/<topic>.ipynb README.md
# If Step 5 ran, also: git add <category>/<topic>_gpu_smoke_kaggle.ipynb
git commit -m "Add <topic> guide: md + executed ipynb

<2-sentence description of what was added and why>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin main
```

## Dependency management

If the notebook needs packages not in the current `pyproject.toml`, add them to the appropriate extra:
- CPU torch / sklearn → `transformer` extra
- CUDA only → `llm-gpu` extra
- Core only (numpy, matplotlib, scipy, opencv) → top-level `dependencies`

After editing `pyproject.toml`, run `uv lock` to update `uv.lock`.

## Quick checklist

Before reporting the task done, verify:
- [ ] `uv run jupyter nbconvert --execute` completed with exit code 0
- [ ] No `UserWarning: FigureCanvasAgg` in notebook outputs
- [ ] No `Traceback` in notebook outputs
- [ ] README.md updated in all 3 places (4 if Step 5 ran)
- [ ] `git push` succeeded

If Step 5 (Kaggle GPU verification) ran, also verify:
- [ ] Kernel pushed with `--accelerator NvidiaTeslaT4` explicitly
- [ ] `kaggle kernels status` reached `COMPLETE`, not `ERROR`
- [ ] Downloaded log contains no `Error`/`Traceback` and every assert-guarded checkpoint's success line
- [ ] Committed notebook has real outputs baked in from the log (not blank cells)
