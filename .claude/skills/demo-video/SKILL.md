---
name: demo-video
description: >
  Generate an animated demo video summarizing a completed HACC research
  experiment.  Reads the experiment directory, plans a scene structure matched
  to the research question, writes a self-contained assembler script, and
  encodes the final MP4.  Invoke AFTER analysis is complete.
allowed-tools: >
  Bash(mkdir *), Bash(python3 *), Bash(ls *),
  Read, Write, Glob
argument-hint: [experiment_dir_path or description of what to show]
---

# Demo Video Generator

**Invoked for:** $ARGUMENTS

You are creating a polished animated MP4 demo video that summarizes a
completed HACC cosmological simulation experiment.  The video is meant for
presentations and demonstrations — it should be visually compelling and
explain the science clearly.

The reusable video toolkit lives at:
  `/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/tools/`
  - `video_tools.py`  — canvas constants, image utils, text overlays,
                        particle renders, frame I/O, encoding
  - `video_scenes.py` — scene generators: title card, halo grid,
                        group reveal, field collage, dual rotation

---

## Phase 1: Understand the Experiment

1. **Locate the experiment directory.**
   If an absolute path was given, use it directly.  Otherwise search:
   ```
   ls /data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/runs/
   ```
   Pick the most relevant directory (newest, or matching a keyword).

2. **Read the research context.**
   Read `prompt.txt` to understand the research question.
   Read any available `results.csv`, `methodology.md`, or key plot files.
   Identify:
   - The scientific question being answered
   - Which halo groups / subsamples exist (and their particle HDF5 files)
   - Which fields are scientifically meaningful for this question
   - Key halos of interest (most massive, most extreme, most representative)

3. **List particle data files.**
   Check `/data/a/cpac/nramachandra/Projects/AmSC/particle_data/` for `.hdf5`
   files and match them to the experiment's groups.

---

## Phase 2: Plan the Video

Based on what you found, choose a scene structure.  A typical ~90 s video:

| Scene | Generator | Purpose | When to include |
|-------|-----------|---------|-----------------|
| Title card | `scene_title_card` | Introduce the question and key concepts | Always |
| Halo grid | `scene_halo_grid` | Show diversity across groups/fields | When ≥2 distinct groups each have particle data |
| Group reveal | `scene_group_reveal` | Progressively expose group membership | When groups have a nested/hierarchical structure |
| Field collage | `scene_field_collage` | Showcase multifield appearance per group | When ≥2 groups × ≥2 scientifically relevant fields |
| Dual rotation | `scene_dual_rotation` | Deep-dive 3D view of representative halos | When a single halo's structure tells the story |

You do **not** need all five scenes.  A focused 2–3 scene video is better
than padding with scenes that don't add insight.

Decide:
- `title`  and `subtitle` for the title card
- `criteria` list: what key measurements / cuts define the groups
- `bracket_groups`: which criteria are "baseline" vs "novel" (if applicable)
- Which groups go in the grid / reveal / collage
- Which halos to use for rotation (pick by mass rank or scientific interest)
- Approximate durations: title ~8 s, grid ~20 s, reveal ~12 s,
  collage ~20 s, rotation ~20 s
- Scientifically appropriate field pairs for rotation panels

Present the plan to the user in a clear table before writing any code.

---

## Phase 3: Write the Assembler Script

Create `make_demo_video.py` inside the experiment directory.
Use **absolute paths** for everything.

```python
#!/usr/bin/env python3
"""
make_demo_video.py — Demo video assembler for <experiment_name>
Research question: <one-line summary>

Usage:
    /home/nramachandra/anaconda3/envs/cosmodev/bin/python3 \
        <experiment_dir>/make_demo_video.py

Scenes
------
  Scene 1  <name>   ~<N> s
  ...
  Total ≈ <T> s  ·  1920×1080  ·  24 fps
"""
import os, sys, shutil
import numpy as np
from PIL import Image

TOOLS_DIR    = "/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/tools"
PARTICLE_DIR = "/data/a/cpac/nramachandra/Projects/AmSC/particle_data"
EXPERIMENT_DIR = "<absolute path to experiment>"

sys.path.insert(0, TOOLS_DIR)
from video_tools  import (W, H, FPS, FIELD_PALETTE, get_top_halos,
                           render_halo_field, load_render_set,
                           save_frames, encode_video)
from video_scenes import (scene_title_card, scene_halo_grid,
                           scene_group_reveal, scene_field_collage,
                           scene_dual_rotation)

RENDERS_DIR  = os.path.join(EXPERIMENT_DIR, "video_renders")
FRAMES_DIR   = os.path.join(EXPERIMENT_DIR, "video_frames")
OUTPUT_VIDEO = os.path.join(EXPERIMENT_DIR, "demo_video.mp4")

os.makedirs(RENDERS_DIR, exist_ok=True)

# ── Group / file configuration ────────────────────────────────────────────────
# Fill in per-experiment particle files and desired groups
FILE_MAP = {
    "<group_name>": os.path.join(PARTICLE_DIR, "<filename>.hdf5"),
    ...
}

def main():
    # Phase 1: Pre-render tiles
    print("=== Pre-rendering tiles ===")
    # render_halo_field(...) for each group × halo × field needed

    # Phase 2: Clear and prepare frames directory
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)
    frame_idx = 0

    # Phase 3: Generate scenes
    print("=== Generating frames ===")

    # Scene 1: Title card
    frame_idx, _ = save_frames(scene_title_card(
        title     = "...",
        subtitle  = "...",
        info_line = "...",
        criteria  = [...],
        bracket_groups = [...],
        duration  = 8.0,
    ), FRAMES_DIR, start_idx=frame_idx, label="title")

    # ... additional scenes ...

    print(f"Total frames: {frame_idx}  (~{frame_idx/FPS:.0f} s)")

    # Phase 4: Encode
    encode_video(FRAMES_DIR, OUTPUT_VIDEO, fps=FPS, w=W, h=H, crf=16)
    print(f"Output: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
```

### Field / criteria conventions
- Always use `FIELD_PALETTE` keys: `"dm"`, `"stars"`, `"gas"`, `"gas_temp"`
- Prefer scientifically meaningful fields first (e.g. gas and temperature for
  thermodynamic studies; DM for morphology studies)
- For `criteria` in `scene_title_card`, each entry needs:
  ```python
  {"symbol": "short name — full name",
   "line1":  "one-sentence description",
   "line2":  "one-sentence interpretation",
   "color_rgb": (r, g, b)}
  ```
- For `bracket_groups`, group criteria into categories:
  ```python
  {"indices": [0],     "label": "Standard\ncriteria", "color_rgb": (215,215,235)},
  {"indices": [1,2,3], "label": "Novel criteria",     "color_rgb": (160,210,255)},
  ```

### Grid layout for `scene_halo_grid`
```python
render_sets = [load_render_set(RENDERS_DIR, prefix) for prefix in prefixes]
# OR build render_sets manually:
render_sets = [
    {"dm": img_dm, "gas": img_gas, ...},
    ...
]
```

### Group reveal configuration
```python
reveal_phases = [
    {"group_key": "group_A", "color_rgb": (220, 60, 60),
     "inset": 0, "border_w": 5, "has_xfade": False, "t_start": 1.5},
    {"group_key": "group_B", "color_rgb": (80, 140, 255),
     "inset": 10, "border_w": 3, "has_xfade": True,  "t_start": 4.0},
    ...
]
```

### Dual rotation pair config
```python
pair_configs = [
    {"hdf5_path": ..., "tag": tag1,
     "left_field": "gas", "right_field": "dm", "duration": 10},
    {"hdf5_path": ..., "tag": tag2,
     "left_field": "stars", "right_field": "gas_temp", "duration": 10},
]
```

---

## Phase 4: Run and Validate

1. Syntax-check the assembler script:
   ```bash
   /home/nramachandra/anaconda3/envs/cosmodev/bin/python3 -m py_compile \
       <experiment_dir>/make_demo_video.py
   ```

2. Run it (this may take several minutes for particle renders):
   ```bash
   /home/nramachandra/anaconda3/envs/cosmodev/bin/python3 \
       <experiment_dir>/make_demo_video.py
   ```

3. Confirm `demo_video.mp4` exists and report its size and duration.

---

## Constraints

- All paths must be **absolute** — never relative
- Never use `pandas` on OpenCosmo data
- Never fabricate or placeholder data — only render halos that actually exist
  in the particle HDF5 files
- Tile renders are cached (`cache=True`); re-running skips already-rendered tiles
- If a particle HDF5 file is missing for a group, skip that group gracefully
  and warn the user rather than failing
- Keep scene durations consistent with FPS=24; total video should be 60–120 s
- For the rotation scene, use halos where the structure is most visually
  interesting — typically the most massive, or most extreme in the target metric
