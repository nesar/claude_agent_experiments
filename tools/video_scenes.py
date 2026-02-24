"""
video_scenes.py — Reusable animated scene generators for HACC demo videos.

Each public function is a generator that yields 1920×1080 PIL RGB frames.
All scene parameters are fully configurable; no experiment-specific constants.
Import from video_tools for canvas constants and image utilities.

Scene inventory
---------------
  scene_title_card        Animated intro with title, subtitle, info lines,
                          criteria cards with per-card colors, and optional
                          square-bracket group annotations on the right.

  scene_halo_grid         N×M grid of halos cycling through multiple fields.
                          Supports crossfade transitions between fields and
                          optional field-name label overlays.

  scene_group_reveal      Progressive reveal — halos start as DM, then colored
                          boxes and field crossfades expose group membership
                          one group at a time.  Title crossfades between two
                          phase-dependent strings.

  scene_field_collage     Groups of halos shown in a scrolling multi-row
                          collage; cycles through field keys with crossfades.

  scene_dual_rotation     Dual-panel synchronized 3D rotation across two halos
                          and two field-pairs (loaded live from HDF5).

Typical usage (in an assembler script)
---------------------------------------
    import sys
    sys.path.insert(0, TOOLS_DIR)
    from video_tools  import W, H, FPS, save_frames, encode_video, ...
    from video_scenes import scene_title_card, scene_halo_grid, ...

    # Scene 1: title card
    save_frames(scene_title_card(
        title     = "My HACC Analysis",
        subtitle  = "A cluster comparison study",
        criteria  = [...],
        ...
    ), FRAMES_DIR, start_idx=0, label="intro")

    # Scene 2: halo grid
    render_sets = [load_render_set(RENDERS_DIR, prefix) for prefix in prefixes]
    save_frames(scene_halo_grid(render_sets, n_cols=3, n_rows=2, ...), ...)

    encode_video(FRAMES_DIR, OUTPUT_VIDEO)
"""

import sys
import os
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from video_tools import (
    W, H, FPS,
    FIELD_PALETTE, FIELD_ORDER_DEFAULT,
    smoothstep, blend, composite_rgba, fit_to_frame,
    make_text_overlay, make_separator_bar, make_square_bracket,
    get_top_halos,
)

import opencosmo as oc                                    # type: ignore
from opencosmo.analysis import halo_projection_array      # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# Scene 1 — Title card
# ══════════════════════════════════════════════════════════════════════════════

def scene_title_card(
        title,
        subtitle=None,
        info_line=None,
        header_text=None,
        criteria=None,
        bracket_groups=None,
        duration=8.0,
        fade_out=True,
        fps=FPS,
        w=W, h=H):
    """
    Animated intro/title card.  Elements fade in sequentially.

    Parameters
    ----------
    title         : str   Large title text (top of frame)
    subtitle      : str   Subtitle line below title (optional)
    info_line     : str   Small info/attribution text (optional)
    header_text   : str   Section header above criteria (e.g. "Key Criteria")
    criteria      : list of dicts, each with keys:
                      symbol    str   Short name / symbol (e.g. "δ₁ — CoM offset")
                      line1     str   First description line
                      line2     str   Second description line (optional)
                      color_rgb (r,g,b)  0–255 accent colour
    bracket_groups: list of dicts, each with keys:
                      indices   list of int  which criteria to span (0-based)
                      label     str
                      color_rgb (r,g,b)
                      bracket_w int (default 48)
    duration      : float  Total scene length in seconds
    fade_out      : bool   Fade to black in the last ~0.7 s

    Yields
    ------
    PIL RGB Image  (w × h)
    """
    # ── Timing ──────────────────────────────────────────────────────────────
    t_title   = 0.0
    t_sub     = 0.4
    t_info    = 0.8
    t_header  = 1.1
    n_crit    = len(criteria) if criteria else 0
    # criteria fan in over the first 2/3 of the scene, each 1 s apart
    crit_start = 1.5
    crit_sep   = min(1.0, (duration * 0.6) / max(n_crit, 1))
    t_crit    = [crit_start + i * crit_sep for i in range(n_crit)]
    fade_dur  = 0.7
    t_hold_end = duration - (0.7 if fade_out else 0.0)
    n_frames  = int(duration * fps)

    # ── Layout constants (relative to h, w) ─────────────────────────────────
    y_title   = int(h * 0.098)
    y_sub     = int(h * 0.160)
    y_info    = int(h * 0.212)
    y_header  = int(h * 0.274)
    crit_y_top    = int(h * 0.340)
    crit_y_step   = int(h * 0.120)

    # ── Pre-render overlays ──────────────────────────────────────────────────
    ovs = {}

    title_items = [{"text": title, "x": w // 2, "y": y_title,
                    "size": 66, "color_rgb": (255, 255, 255), "bold": True, "stroke_w": 3}]
    if subtitle:
        title_items.append({"text": subtitle, "x": w // 2, "y": y_sub,
                             "size": 40, "color_rgb": (170, 205, 255),
                             "bold": True, "stroke_w": 2})
    ovs["title"] = make_text_overlay(title_items, fw=w, fh=h)

    if info_line:
        ovs["info"] = make_text_overlay([
            {"text": info_line, "x": w // 2, "y": y_info, "size": 17,
             "color_rgb": (120, 160, 210), "bold": False, "stroke_w": 1},
        ], fw=w, fh=h)

    if header_text:
        ovs["header"] = make_text_overlay([
            {"text": header_text, "x": w // 2, "y": y_header, "size": 21,
             "color_rgb": (190, 210, 240), "bold": False, "stroke_w": 1},
        ], fw=w, fh=h)
        ovs["hbar_l"] = make_separator_bar(
            (140, 170, 210), 0.10, 0.38, y_header / h, fw=w, fh=h, thickness=2)
        ovs["hbar_r"] = make_separator_bar(
            (140, 170, 210), 0.62, 0.90, y_header / h, fw=w, fh=h, thickness=2)

    if criteria:
        for i, crit in enumerate(criteria):
            y0     = crit_y_top + i * crit_y_step
            r, g, b = crit["color_rgb"]
            items  = [
                {"text": crit["symbol"], "x": 190, "y": y0,
                 "size": 24, "color_rgb": (r, g, b), "bold": True, "stroke_w": 2,
                 "ha": "left", "va": "center"},
                {"text": crit.get("line1", ""), "x": 200, "y": y0 + 38,
                 "size": 14, "color_rgb": (200, 205, 215), "bold": False, "stroke_w": 1,
                 "ha": "left", "va": "center"},
            ]
            if crit.get("line2"):
                items.append(
                    {"text": crit["line2"], "x": 200, "y": y0 + 62,
                     "size": 14, "color_rgb": (170, 175, 190), "bold": False, "stroke_w": 1,
                     "ha": "left", "va": "center"}
                )
            ovs[f"crit_{i}"]     = make_text_overlay(items, fw=w, fh=h)
            ovs[f"crit_bar_{i}"] = make_separator_bar(
                (r, g, b), 0.08, 0.09, y0 / h, fw=w, fh=h, thickness=16)

    if bracket_groups and criteria:
        # Compute y extents for each bracket from the criteria they span
        y_top_per_crit = [crit_y_top + i * crit_y_step for i in range(n_crit)]
        y_bot_per_crit = [y_top_per_crit[i] + 85 for i in range(n_crit)]
        BX = int(w * 0.755)
        for bi, bg in enumerate(bracket_groups):
            idxs = bg["indices"]
            y_top_br = y_top_per_crit[min(idxs)] - 17
            y_bot_br = y_bot_per_crit[max(idxs)] + 5
            ovs[f"bracket_{bi}"] = make_square_bracket(
                x_left=BX,
                y_top_img=y_top_br,
                y_bot_img=y_bot_br,
                label=bg["label"],
                color_rgb=bg.get("color_rgb", (200, 210, 240)),
                bracket_w=bg.get("bracket_w", 48),
                lw=2.5,
                label_size=17,
                fw=w, fh=h,
            )

    # ── Frame generator ──────────────────────────────────────────────────────
    def _alpha(start_t, t):
        return smoothstep(min(max(t - start_t, 0) / fade_dur, 1))

    for i in range(n_frames):
        t = i / fps
        g = (smoothstep(1 - (t - t_hold_end) / 0.8)
             if fade_out and t >= t_hold_end else 1.0)

        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas = composite_rgba(canvas, ovs["title"],  _alpha(t_title, t) * g)
        if subtitle:
            canvas = composite_rgba(canvas, ovs["title"], _alpha(t_sub, t) * g)
        if info_line:
            canvas = composite_rgba(canvas, ovs["info"],   _alpha(t_info,   t) * g)
        if header_text:
            canvas = composite_rgba(canvas, ovs["header"], _alpha(t_header, t) * g)
            canvas = composite_rgba(canvas, ovs["hbar_l"], _alpha(t_header, t) * g * 0.7)
            canvas = composite_rgba(canvas, ovs["hbar_r"], _alpha(t_header, t) * g * 0.7)
        if criteria:
            for ci, tc in enumerate(t_crit):
                a = _alpha(tc, t) * g
                canvas = composite_rgba(canvas, ovs[f"crit_{ci}"],     a)
                canvas = composite_rgba(canvas, ovs[f"crit_bar_{ci}"], a * 0.85)
        if bracket_groups and criteria:
            for bi, bg in enumerate(bracket_groups):
                # Bracket appears when its last criterion appears
                last_crit_t = t_crit[max(bg["indices"])]
                canvas = composite_rgba(canvas, ovs[f"bracket_{bi}"],
                                        _alpha(last_crit_t, t) * g * 0.92)
        yield canvas


# ══════════════════════════════════════════════════════════════════════════════
# Scene 2 — Halo grid with field cycling
# ══════════════════════════════════════════════════════════════════════════════

def scene_halo_grid(
        render_sets,
        n_cols=3, n_rows=2,
        field_keys=None,
        hold_sec=3.0,
        xfade_sec=1.5,
        group_labels=None,
        field_label_overlay=True,
        fade_in_sec=0.8,
        fade_out_sec=0.8,
        fps=FPS, w=W, h=H):
    """
    N×M grid of halos cycling through multiple fields with crossfade transitions.

    Parameters
    ----------
    render_sets   : list of dicts, one per grid cell.
                    Each dict maps field_key → PIL RGB Image.
                    Length must equal n_cols × n_rows.
    n_cols, n_rows: Grid dimensions
    field_keys    : Ordered list of field keys to cycle through.
                    Defaults to FIELD_ORDER_DEFAULT.
    hold_sec      : Seconds to hold each field
    xfade_sec     : Seconds for the crossfade between fields
    group_labels  : Optional list of str, one per cell (drawn in cell corners)
    field_label_overlay : If True, show field name in top-centre of frame
    fade_in_sec   : Fade-from-black duration at start
    fade_out_sec  : Fade-to-black duration at end

    Yields
    ------
    PIL RGB Image  (w × h)
    """
    if field_keys is None:
        field_keys = FIELD_ORDER_DEFAULT

    n_cells = n_cols * n_rows
    assert len(render_sets) == n_cells, (
        f"render_sets has {len(render_sets)} items; expected {n_cells}")

    # Cell geometry
    GAP   = 6
    cell_w = (w - (n_cols + 1) * GAP) // n_cols
    cell_h = (h - (n_rows + 1) * GAP) // n_rows

    def _cell_xy(idx):
        row = idx // n_cols
        col = idx  % n_cols
        return GAP + col * (cell_w + GAP), GAP + row * (cell_h + GAP)

    # Resize all tiles once
    tiles = []   # tiles[cell_idx][field_key] = PIL RGB (cell_w × cell_h)
    for rs in render_sets:
        resized = {}
        for k in field_keys:
            img = rs.get(k, Image.new("RGB", (cell_w, cell_h), (8, 10, 18)))
            resized[k] = img.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        tiles.append(resized)

    # Field label overlays
    field_label_ovs = {}
    if field_label_overlay:
        for k in field_keys:
            fp = FIELD_PALETTE.get(k, {})
            label = fp.get("label", k)
            accent = fp.get("accent", (200, 200, 200))
            field_label_ovs[k] = make_text_overlay([
                {"text": label, "x": w // 2, "y": int(h * 0.038),
                 "size": 22, "color_rgb": accent, "bold": True, "stroke_w": 2},
            ], fw=w, fh=h)

    # Group label overlays per cell
    cell_label_ovs = [None] * n_cells
    if group_labels:
        for ci, lbl in enumerate(group_labels):
            if lbl is None:
                continue
            x0, y0 = _cell_xy(ci)
            cell_label_ovs[ci] = make_text_overlay([
                {"text": lbl,
                 "x": x0 + 6, "y": y0 + 16, "size": 13,
                 "color_rgb": (200, 200, 200), "bold": False, "stroke_w": 1,
                 "ha": "left", "va": "top"},
            ], fw=w, fh=h)

    # Timing: one full cycle per field
    cycle_sec  = hold_sec + xfade_sec
    n_fields   = len(field_keys)
    total_sec  = (fade_in_sec + n_fields * cycle_sec
                  - xfade_sec + fade_out_sec)  # last field has no trailing xfade
    n_frames   = int(total_sec * fps)

    for fi in range(n_frames):
        t = fi / fps

        # Global fade envelope
        g = smoothstep(min(t / max(fade_in_sec, 0.001), 1))
        t_after_fadein = t - fade_in_sec
        t_before_end   = total_sec - fade_out_sec
        if t >= t_before_end:
            g *= smoothstep(1 - (t - t_before_end) / max(fade_out_sec, 0.001))

        if t_after_fadein < 0:
            yield Image.new("RGB", (w, h), (0, 0, 0))
            continue

        # Which field pair and crossfade fraction?
        cycle_pos  = t_after_fadein % cycle_sec
        field_from = int(t_after_fadein // cycle_sec) % n_fields
        field_to   = (field_from + 1) % n_fields
        xf = smoothstep(min(max(cycle_pos - hold_sec, 0) / max(xfade_sec, 0.001), 1))
        key_from = field_keys[field_from]
        key_to   = field_keys[field_to]

        canvas = Image.new("RGB", (w, h), (0, 0, 0))

        for ci in range(n_cells):
            x0, y0 = _cell_xy(ci)
            img_from = tiles[ci][key_from]
            img_to   = tiles[ci][key_to]
            tile     = blend(img_from, img_to, xf)
            if g < 1.0:
                tile = Image.fromarray(
                    (np.array(tile, np.float32) * g).clip(0, 255).astype(np.uint8))
            canvas.paste(tile, (x0, y0))

        # Field label overlay
        if field_label_overlay:
            key_show = key_to if xf > 0.5 else key_from
            canvas = composite_rgba(canvas, field_label_ovs[key_show], g * 0.9)

        # Cell group labels
        if group_labels:
            for ci, ov in enumerate(cell_label_ovs):
                if ov is not None:
                    canvas = composite_rgba(canvas, ov, g * 0.75)

        yield canvas


# ══════════════════════════════════════════════════════════════════════════════
# Scene 3 — Group membership reveal
# ══════════════════════════════════════════════════════════════════════════════

def scene_group_reveal(
        grid_layout,
        dm_tiles,
        alt_tiles,
        reveal_phases,
        title_baseline,
        title_exceptions=None,
        legend_lines=None,
        n_cols=5, n_rows=3,
        header_h=45,
        footer_h=100,
        total_sec=12.5,
        fade_in_sec=1.5,
        fade_out_sec=1.0,
        xfade_dur=1.5,
        fps=FPS, w=W, h=H):
    """
    Progressive group-membership reveal on a grid of halos.

    Halos start as DM projections.  Colored boxes and field crossfades are
    progressively composited to reveal which groups each halo belongs to.

    Parameters
    ----------
    grid_layout    : list of (group_key, halo_idx)  — one entry per cell, row-major
    dm_tiles       : dict  (group_key, halo_idx) → PIL RGB  (Dark Matter tiles)
    alt_tiles      : dict  (group_key, halo_idx) → PIL RGB  (alternate field tiles)
    reveal_phases  : list of dicts, one per reveal phase (starting after fade-in):
                       group_key    str    Which group this phase reveals
                       color_rgb    (r,g,b) Box colour
                       inset        int    Pixel inset from cell edge (0 = outer)
                       border_w     int    Border line width
                       has_xfade    bool   Whether to crossfade to alt_tiles
                       t_start      float  Phase start time (seconds)
    title_baseline : str   Title shown before first reveal phase
    title_exceptions: str  Title shown after first reveal phase (crossfades in)
                           If None, title stays as title_baseline throughout.
    legend_lines   : list of dicts:
                       text         str
                       color_rgb    (r,g,b)
                       phase_idx    int    Which phase (0-based) triggers this line
    n_cols, n_rows : Grid dimensions
    header_h       : Header height in pixels
    footer_h       : Footer height in pixels
    total_sec, fade_in_sec, fade_out_sec, xfade_dur : Timing parameters

    Yields
    ------
    PIL RGB Image  (w × h)
    """
    MARGIN = 15
    GAP    = 6
    cell_w = (w  - 2 * MARGIN - (n_cols - 1) * GAP) // n_cols
    cell_h = (h  - header_h - footer_h - (n_rows - 1) * GAP) // n_rows

    def _cell_xy(idx):
        row = idx // n_cols
        col = idx  % n_cols
        return MARGIN + col * (cell_w + GAP), header_h + row * (cell_h + GAP)

    # Resize tiles
    def _resize(img):
        return img.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)

    dm_r  = {k: _resize(v) for k, v in dm_tiles.items()}
    alt_r = {k: _resize(v) for k, v in alt_tiles.items()}

    # Pre-build box overlays
    def _box_overlay(positions, color_rgb, inset, border_w):
        ov  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        drw = ImageDraw.Draw(ov)
        r, g, b = color_rgb
        for (x0, y0) in positions:
            drw.rectangle(
                [x0 + inset, y0 + inset,
                 x0 + cell_w - 1 - inset, y0 + cell_h - 1 - inset],
                outline=(r, g, b, 255), width=border_w)
        return ov

    box_ovs = []
    phase_groups = {}
    for pi, ph in enumerate(reveal_phases):
        gk    = ph["group_key"]
        pos   = [_cell_xy(ci) for ci, (g, _) in enumerate(grid_layout) if g == gk]
        box_ovs.append(_box_overlay(pos, ph["color_rgb"], ph["inset"], ph["border_w"]))
        phase_groups[pi] = gk

    # Title overlays
    ov_title_base = make_text_overlay([
        {"text": title_baseline, "x": w // 2, "y": header_h // 2 + 4,
         "size": 18, "color_rgb": (190, 210, 240), "bold": True, "stroke_w": 1},
    ], fw=w, fh=h)
    ov_title_exc = None
    if title_exceptions:
        ov_title_exc = make_text_overlay([
            {"text": title_exceptions, "x": w // 2, "y": header_h // 2 + 4,
             "size": 18, "color_rgb": (160, 210, 255), "bold": True, "stroke_w": 1},
        ], fw=w, fh=h)

    # Legend overlays
    leg_ys = [h - footer_h + 18 + i * 22 for i in range(len(legend_lines or []))]
    leg_ovs = []
    if legend_lines:
        for j, ll in enumerate(legend_lines):
            leg_ovs.append(make_text_overlay([
                {"text": ll["text"], "x": w // 2, "y": leg_ys[j], "size": 13,
                 "color_rgb": ll["color_rgb"], "bold": False, "stroke_w": 1},
            ], fw=w, fh=h))

    n_frames    = int(total_sec * fps)
    t_fade_out  = total_sec - fade_out_sec

    def _phase_alpha(t, ph_idx, rise=0.8):
        return smoothstep(min(max(t - reveal_phases[ph_idx]["t_start"], 0) / rise, 1))

    def _xfade_alpha(t, ph_idx):
        return smoothstep(min(max(t - reveal_phases[ph_idx]["t_start"], 0) / xfade_dur, 1))

    for fi in range(n_frames):
        t = fi / fps
        g = smoothstep(min(t / max(fade_in_sec, 0.001), 1))
        if t >= t_fade_out:
            g *= smoothstep(1 - (t - t_fade_out) / max(fade_out_sec, 0.001))

        phase_alphas = [_phase_alpha(t, pi) for pi in range(len(reveal_phases))]
        xfade_alphas = [_xfade_alpha(t, pi) if reveal_phases[pi].get("has_xfade") else 0.0
                        for pi in range(len(reveal_phases))]
        xf_by_group  = {phase_groups[pi]: xfade_alphas[pi]
                        for pi in range(len(reveal_phases))}

        canvas = Image.new("RGB", (w, h), (0, 0, 0))

        for ci, (gk, hi) in enumerate(grid_layout):
            x0, y0 = _cell_xy(ci)
            base = dm_r.get((gk, hi), Image.new("RGB", (cell_w, cell_h), (8, 10, 18)))
            xf   = xf_by_group.get(gk, 0.0)
            if xf > 0 and (gk, hi) in alt_r:
                tile = blend(base, alt_r[(gk, hi)], xf)
            else:
                tile = base
            if g < 1.0:
                tile = Image.fromarray(
                    (np.array(tile, np.float32) * g).clip(0, 255).astype(np.uint8))
            canvas.paste(tile, (x0, y0))

        # Gap separators
        drw = ImageDraw.Draw(canvas)
        for col in range(1, n_cols):
            sx = MARGIN + col * (cell_w + GAP) - GAP
            drw.rectangle([sx, header_h, sx + GAP - 1, h - footer_h], fill=(20, 20, 20))
        for row in range(1, n_rows):
            sy = header_h + row * (cell_h + GAP) - GAP
            drw.rectangle([MARGIN, sy, w - MARGIN, sy + GAP - 1], fill=(20, 20, 20))

        for pi, ov in enumerate(box_ovs):
            canvas = composite_rgba(canvas, ov, alpha_mult=phase_alphas[pi] * g)

        # Title crossfade: base → exceptions driven by first reveal phase alpha
        if ov_title_exc is not None and len(phase_alphas) > 0:
            a_exc = phase_alphas[0]
            canvas = composite_rgba(canvas, ov_title_base,
                                    alpha_mult=(1.0 - a_exc) * g * 0.90)
            canvas = composite_rgba(canvas, ov_title_exc,
                                    alpha_mult=a_exc * g * 0.90)
        else:
            canvas = composite_rgba(canvas, ov_title_base, alpha_mult=g * 0.90)

        if legend_lines:
            for j, ll in enumerate(legend_lines):
                pi_for_leg = ll.get("phase_idx", 0)
                canvas = composite_rgba(canvas, leg_ovs[j],
                                        alpha_mult=phase_alphas[pi_for_leg] * g * 0.95)

        yield canvas


# ══════════════════════════════════════════════════════════════════════════════
# Scene 4 — Multi-group multifield collage tour
# ══════════════════════════════════════════════════════════════════════════════

def scene_field_collage(
        groups,
        n_cols=4,
        hold_sec=4.0,
        xfade_sec=1.2,
        fade_in_sec=0.6,
        fade_out_sec=0.6,
        show_group_label=True,
        show_field_label=True,
        fps=FPS, w=W, h=H):
    """
    Scrolling multifield collage of halo groups.

    One group at a time fills the screen in a grid; field cycling animates
    across all cells simultaneously, with group crossfades between groups.

    Parameters
    ----------
    groups     : list of dicts, one per group:
                   label       str
                   color_rgb   (r,g,b)
                   halos       list of dicts:  {field_key: PIL_RGB, ...}
    n_cols     : Number of columns in the grid (rows auto-calculated)
    hold_sec   : Seconds to hold each field within a group
    xfade_sec  : Crossfade duration for field and group transitions
    fade_in_sec, fade_out_sec : Scene envelope durations
    show_group_label : Overlay group name in top-left corner
    show_field_label : Overlay field name top-centre

    Yields
    ------
    PIL RGB Image  (w × h)
    """
    import math

    # Build a flat list of (group_idx, field_key) animation steps
    # Each group shows all its halos for each field before moving to next group
    all_steps = []   # (group_idx, field_key_from, field_key_to, is_last_of_group)
    for gi, grp in enumerate(groups):
        fkeys = list(grp["halos"][0].keys()) if grp["halos"] else FIELD_ORDER_DEFAULT
        for fki, fk in enumerate(fkeys):
            fk_next = fkeys[(fki + 1) % len(fkeys)]
            is_last = (fki == len(fkeys) - 1)
            all_steps.append((gi, fk, fk_next, is_last))

    n_steps     = len(all_steps)
    step_sec    = hold_sec + xfade_sec
    total_sec   = fade_in_sec + n_steps * step_sec + fade_out_sec
    n_frames    = int(total_sec * fps)

    # Compute grid geometry for the largest group
    max_halos = max(len(grp["halos"]) for grp in groups)
    n_rows    = math.ceil(max_halos / n_cols)
    GAP       = 8
    cell_w    = (w - (n_cols + 1) * GAP) // n_cols
    cell_h    = (h - (n_rows + 1) * GAP) // n_rows

    def _xy(idx):
        r = idx // n_cols
        c = idx  % n_cols
        return GAP + c * (cell_w + GAP), GAP + r * (cell_h + GAP)

    # Pre-scale all tiles
    scaled = []
    for grp in groups:
        grp_tiles = []
        for halo in grp["halos"]:
            scaled_fields = {}
            for k, img in halo.items():
                scaled_fields[k] = img.convert("RGB").resize(
                    (cell_w, cell_h), Image.LANCZOS)
            grp_tiles.append(scaled_fields)
        scaled.append(grp_tiles)

    # Field and group label overlays
    field_label_cache = {}
    group_label_cache = {}
    for grp in groups:
        lbl = grp["label"]
        r, g, b = grp.get("color_rgb", (200, 200, 200))
        group_label_cache[lbl] = make_text_overlay([
            {"text": lbl, "x": int(w * 0.08), "y": int(h * 0.035),
             "size": 22, "color_rgb": (r, g, b), "bold": True, "stroke_w": 2,
             "ha": "left"},
        ], fw=w, fh=h)
    for k in FIELD_PALETTE:
        fp = FIELD_PALETTE[k]
        field_label_cache[k] = make_text_overlay([
            {"text": fp["label"], "x": w // 2, "y": int(h * 0.035),
             "size": 22, "color_rgb": fp["accent"], "bold": True, "stroke_w": 2},
        ], fw=w, fh=h)

    for fi in range(n_frames):
        t = fi / fps
        g_fade = smoothstep(min(t / max(fade_in_sec, 0.001), 1))
        if t >= total_sec - fade_out_sec:
            g_fade *= smoothstep(1 - (t - (total_sec - fade_out_sec)) / max(fade_out_sec, 0.001))

        t_in   = t - fade_in_sec
        step   = max(0, min(int(t_in // step_sec), n_steps - 1))
        frac   = (t_in - step * step_sec) / step_sec if t_in >= 0 else 0.0
        xf     = smoothstep(min(max(frac - hold_sec / step_sec, 0) /
                                max(xfade_sec / step_sec, 0.001), 1))

        gi, fk_from, fk_to, _ = all_steps[step]
        grp     = groups[gi]
        halo_tiles = scaled[gi]

        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        for hi, ht in enumerate(halo_tiles):
            x0, y0 = _xy(hi)
            img_f = ht.get(fk_from, Image.new("RGB", (cell_w, cell_h), (8, 10, 18)))
            img_t = ht.get(fk_to,   Image.new("RGB", (cell_w, cell_h), (8, 10, 18)))
            tile  = blend(img_f, img_t, xf)
            if g_fade < 1.0:
                tile = Image.fromarray(
                    (np.array(tile, np.float32) * g_fade).clip(0, 255).astype(np.uint8))
            canvas.paste(tile, (x0, y0))

        key_show = fk_to if xf > 0.5 else fk_from
        if show_group_label:
            ov_g = group_label_cache.get(grp["label"])
            if ov_g:
                canvas = composite_rgba(canvas, ov_g, g_fade * 0.90)
        if show_field_label and key_show in field_label_cache:
            canvas = composite_rgba(canvas, field_label_cache[key_show], g_fade * 0.80)

        yield canvas


# ══════════════════════════════════════════════════════════════════════════════
# Scene 5 — Dual-panel synchronized 3D rotation
# ══════════════════════════════════════════════════════════════════════════════

def scene_dual_rotation(
        pair_configs,
        n_azimuthal=2,
        phi_max=None,
        n_bins=900,
        fade_in_sec=0.7,
        fade_mid_sec=0.7,
        fade_out_sec=0.7,
        fps=FPS, w=W, h=H):
    """
    Dual-panel (1×2) synchronized 3D rotation for a sequence of halo pairs.

    Each pair shows two projections side-by-side with identical viewing angles,
    rotating smoothly in azimuth and elevation.  Multiple pairs are played
    back-to-back with a mid-scene crossfade.

    Parameters
    ----------
    pair_configs  : list of dicts, one per pair to show:
                      hdf5_path    str
                      tag          int   Halo unique_tag
                      left_field   str   FIELD_PALETTE key for the left panel
                      right_field  str   FIELD_PALETTE key for the right panel
                      duration     float Seconds for this pair
                      left_label   str   (optional) Panel label override
                      right_label  str   (optional) Panel label override
    n_azimuthal   : int   Number of full 360° azimuthal rotations
    phi_max       : float Max elevation angle (radians).  Default π/2.
    n_bins        : int   Projection resolution (pixels per side)
    fade_in_sec   : float Fade from black at the very start
    fade_mid_sec  : float Crossfade between consecutive pairs
    fade_out_sec  : float Fade to black at the very end

    Yields
    ------
    PIL RGB Image  (w × h)
    """
    import numpy as np

    if phi_max is None:
        phi_max = np.pi / 2

    SEP  = 4
    PW   = (w - SEP) // 2
    PROJ = n_bins
    px_L = (PW - PROJ) // 2
    px_R = PW + SEP + (PW - PROJ) // 2
    py   = (h - PROJ) // 2

    def _project(pos_arr, weights, theta, phi, bins=PROJ):
        """Fast weighted 2D projection along an arbitrary viewing direction."""
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_p, s_p = np.cos(phi),   np.sin(phi)
        # Rotation: azimuth then elevation
        x, y, z = pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2]
        xr =  c_t * x + s_t * y
        yr = -s_t * x + c_t * y
        zr =  z
        # elevation
        xr2 =  c_p * xr + s_p * zr
        yr2 =  yr
        norm = np.max(np.abs(pos_arr)) * 1.05 + 1e-10
        xi = np.clip(((xr2 + norm) / (2 * norm) * bins).astype(int), 0, bins - 1)
        yi = np.clip(((yr2 + norm) / (2 * norm) * bins).astype(int), 0, bins - 1)
        grid = np.zeros((bins, bins), dtype=np.float32)
        np.add.at(grid, (yi, xi), weights)
        return grid

    def _to_rgb(grid, cmap_name, log_scale=True):
        g = np.log1p(grid) if log_scale else grid
        hi = np.percentile(g[g > 0], 98) if np.any(g > 0) else 1.0
        norm = np.clip(g / (hi + 1e-10), 0, 1)
        rgb  = (plt.get_cmap(cmap_name)(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[grid == 0] = 0
        return Image.fromarray(rgb)

    def _load_particles(hdf5_path, tag, field_key):
        fp   = FIELD_PALETTE[field_key]
        ptype, qty = fp["ptype"], fp["quantity"]
        data = oc.open(hdf5_path)
        for halo in data.halos():
            props = halo["halo_properties"]
            if int(props["unique_tag"]) != tag:
                continue
            ptcl  = halo.get(f"{ptype}_particles")
            if ptcl is None:
                return None, None, fp["cmap"]
            cx = float(props["sod_halo_min_pot_x"])
            cy = float(props["sod_halo_min_pot_y"])
            cz = float(props["sod_halo_min_pot_z"])
            x  = np.array(ptcl["x"]) - cx
            y  = np.array(ptcl["y"]) - cy
            z  = np.array(ptcl["z"]) - cz
            pos = np.stack([x, y, z], axis=1)
            w_arr = np.array(ptcl[qty], dtype=np.float32)
            return pos, w_arr, fp["cmap"]
        return None, None, fp["cmap"]

    # Panel label overlays
    def _label_ov(text, accent_rgb, x_center):
        return make_text_overlay([
            {"text": text, "x": x_center, "y": int(h * 0.04),
             "size": 18, "color_rgb": accent_rgb, "bold": True, "stroke_w": 2},
        ], fw=w, fh=h)

    # Separator overlay
    sep_ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sep_arr = np.array(sep_ov)
    sep_arr[:, PW:PW + SEP, :] = (30, 30, 30, 255)
    sep_ov = Image.fromarray(sep_arr, "RGBA")

    n_pairs = len(pair_configs)
    for pair_idx, pc in enumerate(pair_configs):
        is_first = (pair_idx == 0)
        is_last  = (pair_idx == n_pairs - 1)
        dur      = pc["duration"]
        n_frames = int(dur * fps)

        l_key  = pc["left_field"]
        r_key  = pc["right_field"]
        l_fp   = FIELD_PALETTE[l_key]
        r_fp   = FIELD_PALETTE[r_key]

        l_label = pc.get("left_label",  l_fp["label"])
        r_label = pc.get("right_label", r_fp["label"])

        ov_label_l = _label_ov(l_label, l_fp["accent"], px_L + PROJ // 2)
        ov_label_r = _label_ov(r_label, r_fp["accent"], px_R + PROJ // 2)

        l_pos, l_w, l_cmap = _load_particles(pc["hdf5_path"], pc["tag"], l_key)
        r_pos, r_w, r_cmap = _load_particles(pc["hdf5_path"], pc["tag"], r_key)

        for fi in range(n_frames):
            t_norm = fi / n_frames

            # Fade envelope
            if is_first and fi / fps < fade_in_sec:
                g_alpha = smoothstep(fi / fps / max(fade_in_sec, 0.001))
            elif is_last and (n_frames - fi) / fps < fade_out_sec:
                g_alpha = smoothstep((n_frames - fi) / fps / max(fade_out_sec, 0.001))
            elif not is_first and fi / fps < fade_mid_sec:
                g_alpha = smoothstep(fi / fps / max(fade_mid_sec, 0.001))
            else:
                g_alpha = 1.0

            theta = 2 * np.pi * n_azimuthal * t_norm
            phi   = phi_max * np.sin(2 * np.pi * t_norm)

            canvas = Image.new("RGB", (w, h), (0, 0, 0))

            if l_pos is not None:
                grid_l = _project(l_pos, l_w, theta, phi)
                img_l  = _to_rgb(grid_l, l_cmap).resize((PROJ, PROJ), Image.LANCZOS)
                canvas.paste(img_l, (px_L, py))

            if r_pos is not None:
                grid_r = _project(r_pos, r_w, theta, phi)
                img_r  = _to_rgb(grid_r, r_cmap).resize((PROJ, PROJ), Image.LANCZOS)
                canvas.paste(img_r, (px_R, py))

            canvas = composite_rgba(canvas, sep_ov, 1.0)
            canvas = composite_rgba(canvas, ov_label_l, g_alpha * 0.9)
            canvas = composite_rgba(canvas, ov_label_r, g_alpha * 0.9)

            if g_alpha < 1.0:
                canvas = Image.fromarray(
                    (np.array(canvas, np.float32) * g_alpha
                     ).clip(0, 255).astype(np.uint8))

            yield canvas
