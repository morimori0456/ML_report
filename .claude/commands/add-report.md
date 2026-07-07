---
description: Add a new ML report (md + executed ipynb) to this repository following the established conventions
---

You are helping add a new report to the ML_report repository. Follow these steps exactly.

## Step 1 — Clarify scope (if not already given)

Ask for:
- **Topic**: what concept or paper are we covering?
- **Category**: which top-level directory? (distillation / llm / autonomous_driving / ema / experiment_tracking / infrastructure — or new one?)
- **Notebook dependency level**: core (numpy/matplotlib only) | transformer (torch + sklearn) | llm-gpu (CUDA)

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

## Step 5 — Update README.md

Three places to update:

### A. Notebook dependency table (around line 42–56)
Add a row:
```
| `<category>/<topic>.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
```

### B. Directory structure block (around line 62–104)
Add the new files under the correct category:
```
│   ├── <topic>.md   # <one-line description>
│   └── <topic>.ipynb  # <one-line description>
```

### C. Report list table (section matching the category)
Add a row:
```
| <Title> | <Topics comma-separated> | [<topic>.md](<category>/<topic>.md) + [demo](<category>/<topic>.ipynb) |
```

## Step 6 — Commit and push

```bash
cd /home/jetson/w/ML_report
git add <category>/<topic>.md <category>/<topic>.ipynb README.md
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
- [ ] README.md updated in all 3 places
- [ ] `git push` succeeded
