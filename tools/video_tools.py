"""
video_tools.py — General-purpose video production utilities for HACC simulation data.

Core primitives for building animated scientific demo videos from OpenCosmo
particle/catalog data.  Fully configurable — no experiment-specific constants.

Canonical HACC hydro field palette
-----------------------------------
  "dm"       → pink         Dark matter mass
  "stars"    → gist_yarg_r  Stellar mass
  "gas"      → plasma_r     Gas mass
  "gas_temp" → rainbow_r    Gas temperature

Typical usage
-------------
    import sys
    sys.path.insert(0, "/data/a/cpac/nramachandra/Projects/AmSC/claude_agent_experiments/tools")
    from video_tools import (
        W, H, FPS, FIELD_PALETTE, FIELD_ORDER_DEFAULT,
        smoothstep, blend, composite_rgba, fit_to_frame,
        make_text_overlay, make_separator_bar, make_square_bracket,
        get_top_halos, render_halo_field, render_halo_multifield_row,
        save_frames, encode_video,
    )

Contents
--------
  Canvas defaults    W, H, FPS
  Field palette      FIELD_PALETTE, FIELD_ORDER_DEFAULT, field_types_for_projection()
  Image operations   smoothstep, blend, fit_to_frame, zoom_image,
                     composite_rgba, recolor_image, add_glow, add_vignette,
                     crop_margins
  Annotations        make_text_overlay, make_separator_bar, make_square_bracket
  Particle renders   get_top_halos, render_halo_field, render_halo_multifield_row,
                     load_render_set
  Frame I/O          save_frames, encode_video
"""

import os
import subprocess
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw, ImageFilter

import opencosmo as oc                                    # type: ignore
from opencosmo.analysis import halo_projection_array      # type: ignore


# ── Canvas defaults ─────────────────────────────────────────────────────────────

W, H = 1920, 1080
FPS  = 24


# ── Canonical HACC hydro field palette ─────────────────────────────────────────
#
# These four fields cover the standard HACC hydrodynamical outputs and match
# the colormaps used by opencosmo's halo_projection_array visualizations.
# Experiments can reference subsets or reorder as needed.

FIELD_PALETTE = {
    "dm": {
        "ptype":    "dm",
        "quantity": "particle_mass",
        "cmap":     "pink",
        "label":    "Dark Matter",
        "accent":   (255, 180, 200),   # warm pink
    },
    "stars": {
        "ptype":    "star",
        "quantity": "particle_mass",
        "cmap":     "gist_yarg_r",
        "label":    "Stars",
        "accent":   (230, 230, 230),   # near-white
    },
    "gas": {
        "ptype":    "gas",
        "quantity": "particle_mass",
        "cmap":     "plasma_r",
        "label":    "Gas",
        "accent":   (255, 200,  60),   # plasma gold
    },
    "gas_temp": {
        "ptype":    "gas",
        "quantity": "temperature",
        "cmap":     "rainbow_r",
        "label":    "Gas Temperature",
        "accent":   ( 60, 210, 255),   # rainbow cyan
    },
}

FIELD_ORDER_DEFAULT = ["dm", "stars", "gas", "gas_temp"]


def field_types_for_projection(field_keys=None):
    """
    Return (types_list, labels_list, cmaps_list) suitable for
    halo_projection_array's ``params`` argument.

    Parameters
    ----------
    field_keys : list of str, optional
        Keys from FIELD_PALETTE.  Defaults to FIELD_ORDER_DEFAULT.

    Returns
    -------
    types  : list of (ptype_str, quantity_str)
    labels : list of str
    cmaps  : list of str
    """
    if field_keys is None:
        field_keys = FIELD_ORDER_DEFAULT
    p      = FIELD_PALETTE
    types  = [(p[k]["ptype"], p[k]["quantity"]) for k in field_keys]
    labels = [p[k]["label"]  for k in field_keys]
    cmaps  = [p[k]["cmap"]   for k in field_keys]
    return types, labels, cmaps


# ── Image processing utilities ─────────────────────────────────────────────────

def smoothstep(t, power=2):
    """Smooth easing: maps [0, 1] → [0, 1] with zero derivative at ends."""
    t = float(np.clip(t, 0, 1))
    if power == 2:
        return t * t * (3 - 2 * t)
    return t**power / (t**power + (1 - t)**power)


def blend(img1, img2, alpha):
    """Linear pixel blend.  alpha=0 → img1; alpha=1 → img2."""
    alpha = float(np.clip(alpha, 0, 1))
    a1 = np.array(img1.convert("RGB"), dtype=np.float32)
    a2 = np.array(img2.convert("RGB"), dtype=np.float32)
    return Image.fromarray((a1 * (1 - alpha) + a2 * alpha).clip(0, 255).astype(np.uint8))


def composite_rgba(base_rgb, overlay_rgba, alpha_mult=1.0):
    """Alpha-composite an RGBA PIL overlay onto an RGB base canvas."""
    base = np.array(base_rgb.convert("RGB"),   dtype=np.float32)
    ov   = np.array(overlay_rgba.convert("RGBA"), dtype=np.float32)
    mask = ov[:, :, 3:4] / 255.0 * float(np.clip(alpha_mult, 0, 1))
    return Image.fromarray((base * (1 - mask) + ov[:, :, :3] * mask).clip(0, 255).astype(np.uint8))


def fit_to_frame(pil_img, fw=W, fh=H):
    """Scale image to fill (fw × fh) letterbox-style; black bars on shorter axis."""
    aspect = pil_img.width / pil_img.height
    target = fw / fh
    if aspect > target:
        nw, nh = fw, int(fw / aspect)
    else:
        nw, nh = int(fh * aspect), fh
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (fw, fh), (0, 0, 0))
    canvas.paste(resized, ((fw - nw) // 2, (fh - nh) // 2))
    return canvas


def zoom_image(pil_img, zoom=1.0, cx_frac=0.5, cy_frac=0.5):
    """Zoom into image (zoom > 1 magnifies around cx/cy fraction of the image)."""
    if zoom <= 1.001:
        return pil_img
    iw, ih = pil_img.size
    nw, nh = int(iw / zoom), int(ih / zoom)
    x0 = max(0, min(int(iw * cx_frac) - nw // 2, iw - nw))
    y0 = max(0, min(int(ih * cy_frac) - nh // 2, ih - nh))
    return pil_img.crop((x0, y0, x0 + nw, y0 + nh)).resize((iw, ih), Image.LANCZOS)


def recolor_image(pil_img, cmap_name, p_lo=2, p_hi=90,
                  invert=False, bg_thresh=None):
    """
    Convert image to grayscale, clip and normalize, apply a matplotlib colormap.

    Parameters
    ----------
    pil_img    : PIL Image (any mode)
    cmap_name  : matplotlib colormap name (e.g. "plasma_r")
    p_lo/p_hi  : Percentile clip range for brightness normalization
    invert     : Invert normalized value before applying colormap.
                 Use invert=True with "plasma_r" to match opencosmo conventions.
    bg_thresh  : Pixels at or below this gray value are forced to black.
                 Auto-estimated from the image if None.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    if bg_thresh is None:
        bg_thresh = max(3.0, float(np.percentile(gray, 3)) + 1.0)
    bg_mask = gray <= bg_thresh
    nonzero = gray[~bg_mask]
    if len(nonzero) == 0:
        return Image.fromarray(np.zeros((*gray.shape[:2], 3), np.uint8))
    lo = np.percentile(nonzero, p_lo)
    hi = np.percentile(nonzero, p_hi)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((gray - lo) / (hi - lo), 0, 1)
    if invert:
        norm = 1.0 - norm
    rgb  = (plt.get_cmap(cmap_name)(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[bg_mask] = 0
    return Image.fromarray(rgb)


def add_glow(pil_img, radius=15, intensity=0.4):
    """Additive glow effect: blend image with its Gaussian-blurred version."""
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius))
    a = np.array(pil_img, dtype=np.float32)
    b = np.array(blurred, dtype=np.float32)
    return Image.fromarray(np.clip(a + b * intensity, 0, 255).astype(np.uint8))


def add_vignette(pil_img, strength=0.55):
    """Darken image edges with a radial gradient vignette."""
    arr  = np.array(pil_img, dtype=np.float32)
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    r    = np.sqrt(((x - w / 2) / (w / 2))**2 + ((y - h / 2) / (h / 2))**2)
    mask = 1 - np.clip(r * strength, 0, 1)
    return Image.fromarray((arr * mask[:, :, None]).clip(0, 255).astype(np.uint8))


def crop_margins(pil_img, margin_frac=0.025):
    """Trim a fractional margin on all sides (strips matplotlib figure borders)."""
    w, h = pil_img.size
    mx, my = int(w * margin_frac), int(h * margin_frac)
    return pil_img.crop((mx, my, w - mx, h - my))


# ── Annotation utilities ───────────────────────────────────────────────────────

def make_text_overlay(lines, fw=W, fh=H):
    """
    Render a list of text items as an RGBA PIL Image (transparent background).

    Each item in *lines* is a dict with keys:
      text       : str
      x, y       : int  pixel position (y measured from top of frame)
      size       : float  font size in points
      color_rgb  : (r, g, b)  0–255 each
      bold       : bool   (default True)
      alpha      : float  0–1  (default 1.0)
      stroke_w   : int    outline/stroke width  (default 2)
      ha         : 'left' | 'center' | 'right'  (default 'center')
      va         : 'top'  | 'center' | 'bottom' (default 'center')
    """
    fig, ax = plt.subplots(figsize=(fw / 100, fh / 100))
    fig.patch.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0.0)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, fw);  ax.set_ylim(0, fh)
    ax.axis("off");      ax.set_facecolor((0, 0, 0, 0))

    for L in lines:
        r, g, b = L.get("color_rgb", (255, 255, 255))
        a       = L.get("alpha", 1.0)
        weight  = "bold" if L.get("bold", True) else "normal"
        sw      = L.get("stroke_w", 2)
        ax.text(
            L["x"], fh - L["y"], L["text"],
            ha=L.get("ha", "center"),
            va=L.get("va", "center"),
            fontsize=L["size"],
            fontweight=weight,
            color=(r / 255, g / 255, b / 255, a),
            path_effects=[pe.withStroke(linewidth=sw, foreground="black")],
        )

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rw, rh = fig.canvas.get_width_height()
    arr = buf.reshape(rh, rw, 4).copy()
    plt.close(fig)
    return Image.fromarray(arr, "RGBA")


def make_separator_bar(color_rgb, x0_frac, x1_frac, y_frac,
                        fw=W, fh=H, thickness=4):
    """
    Draw a thin horizontal colour bar as an RGBA overlay.

    Parameters
    ----------
    color_rgb            : (r, g, b)
    x0_frac, x1_frac     : Start/end x positions as fractions of frame width
    y_frac               : Y position as fraction of frame height (0 = top)
    thickness            : Bar height in pixels
    """
    overlay = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    y  = int(fh * y_frac)
    x0 = int(fw * x0_frac)
    x1 = int(fw * x1_frac)
    r, g, b = color_rgb
    draw.rectangle([x0, y - thickness // 2, x1, y + thickness // 2],
                   fill=(r, g, b, 220))
    return overlay


def make_square_bracket(x_left, y_top_img, y_bot_img, label,
                         color_rgb=(200, 210, 240),
                         bracket_w=48, lw=2.5, label_size=17,
                         fw=W, fh=H):
    """
    Render a right-facing square bracket  ]  as an RGBA PIL Image (fw × fh).

    Three straight lines — top cap, vertical bar, bottom cap — with a label
    placed to the right of the bar.

    Parameters
    ----------
    x_left      : x of the open (left) edges of the caps, in pixels
    y_top_img   : y of the top anchor in image coordinates (y=0 at top)
    y_bot_img   : y of the bottom anchor in image coordinates
    label       : Label text (may include newlines)
    color_rgb   : (r, g, b) colour for bracket + label
    bracket_w   : Width of the caps; vertical bar is at x_left + bracket_w
    lw          : Line width in points
    label_size  : Font size for the label
    fw, fh      : Frame width/height
    """
    import matplotlib.patheffects as _pe

    fig, ax = plt.subplots(figsize=(fw / 100, fh / 100))
    fig.patch.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0.0)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, fw);  ax.set_ylim(0, fh)
    ax.axis("off");      ax.set_facecolor((0, 0, 0, 0))

    # Convert image coords (y=0 top) → matplotlib coords (y=0 bottom)
    my_top = fh - y_top_img
    my_bot = fh - y_bot_img
    my_mid = (my_top + my_bot) / 2
    x_bar  = x_left + bracket_w
    r, g, b = color_rgb
    col = (r / 255, g / 255, b / 255, 1.0)

    kw = dict(color=col, linewidth=lw, solid_capstyle="butt",
              solid_joinstyle="miter")
    ax.plot([x_left, x_bar], [my_top, my_top], **kw)   # top cap
    ax.plot([x_bar,  x_bar], [my_top, my_bot], **kw)   # vertical bar
    ax.plot([x_left, x_bar], [my_bot, my_bot], **kw)   # bottom cap

    ax.text(x_bar + 18, my_mid, label,
            ha="left", va="center",
            fontsize=label_size, fontweight="bold",
            color=col,
            path_effects=[_pe.withStroke(linewidth=1.5, foreground="black")])

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rw, rh = fig.canvas.get_width_height()
    arr = buf.reshape(rh, rw, 4).copy()
    plt.close(fig)
    return Image.fromarray(arr, "RGBA")


# ── OpenCosmo particle render helpers ──────────────────────────────────────────

def get_top_halos(hdf5_path, n=4):
    """
    Return [(fof_mass, unique_tag), …] sorted descending by FoF mass
    from an opencosmo particle HDF5 file.

    Parameters
    ----------
    hdf5_path : str   Path to the opencosmo HDF5 file
    n         : int   Number of top halos to return

    Returns
    -------
    list of (float, int)
    """
    data = oc.open(hdf5_path)
    info = []
    for halo in data.halos():
        props = halo["halo_properties"]
        tag   = int(props["unique_tag"])
        m     = props["fof_halo_mass"]
        mass  = float(m.value) if hasattr(m, "value") else float(m)
        info.append((mass, tag))
    info.sort(reverse=True)
    return info[:n]


def render_halo_field(hdf5_path, tag, field_key, out_path,
                      width=4.5, cache=True):
    """
    Render a single field projection for one halo using halo_projection_array.
    Extracts the main projection axes (strips colorbar and figure margins).

    Parameters
    ----------
    hdf5_path : str   Path to opencosmo particle HDF5 file
    tag       : int   Halo unique_tag
    field_key : str   Key in FIELD_PALETTE  ("dm", "stars", "gas", "gas_temp")
    out_path  : str   Where to save the PNG
    width     : float Projection half-width in units of R_halo
    cache     : bool  If True, skip rendering when out_path already exists

    Returns
    -------
    PIL RGB Image (square crop of the projection axes)
    """
    if cache and os.path.exists(out_path):
        print(f"    [cache] {os.path.basename(out_path)}")
        return Image.open(out_path).convert("RGB")

    fp = FIELD_PALETTE[field_key]
    print(f"    Rendering {fp['label']} for tag {tag} …")
    t0   = time.time()
    data = oc.open(hdf5_path)

    halo_ids = np.array([[tag]])
    params   = {
        "fields": ([(fp["ptype"], fp["quantity"])],),
        "labels": ([fp["label"]],),
        "cmaps":  ([fp["cmap"]],),
    }
    fig = halo_projection_array(halo_ids, data, params=params, width=width)
    fig.patch.set_facecolor("black")
    fig.set_size_inches(12, 12)
    fig.canvas.draw()

    cands   = [ax for ax in fig.axes if ax.get_position().width > 0.08]
    main_ax = max(cands, key=lambda ax: ax.get_position().width * ax.get_position().height)
    pos     = main_ax.get_position()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rw, rh = fig.canvas.get_width_height()
    arr    = buf.reshape(rh, rw, 4)[:, :, :3].copy()
    x0, x1 = max(0, int(pos.x0 * rw)), min(rw, int(pos.x1 * rw))
    y0, y1 = max(0, int((1 - pos.y1) * rh)), min(rh, int((1 - pos.y0) * rh))
    cropped = arr[y0:y1, x0:x1]
    plt.close(fig)

    img = Image.fromarray(cropped)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    print(f"      {cropped.shape[1]}×{cropped.shape[0]} px  ({time.time() - t0:.1f} s)")
    return img


def render_halo_multifield_row(hdf5_path, tag, out_path,
                                field_keys=None, width=4.5, cache=True):
    """
    Render a 1×N multifield row for one halo and save as a single PNG.

    Parameters
    ----------
    hdf5_path  : str
    tag        : int   Halo unique_tag
    out_path   : str   Where to save the combined PNG
    field_keys : list of str, optional.  Defaults to FIELD_ORDER_DEFAULT.
    width      : float
    cache      : bool

    Returns
    -------
    PIL RGB Image (full multifield row)
    """
    if cache and os.path.exists(out_path):
        print(f"    [cache] {os.path.basename(out_path)}")
        return Image.open(out_path).convert("RGB")

    if field_keys is None:
        field_keys = FIELD_ORDER_DEFAULT

    types, labels, cmaps = field_types_for_projection(field_keys)
    n = len(field_keys)

    print(f"    Rendering {n}-field row for tag {tag} …")
    t0   = time.time()
    data = oc.open(hdf5_path)

    halo_ids = np.array([[tag] * n])
    params   = {"fields": (types,), "labels": (labels,), "cmaps": (cmaps,)}
    fig = halo_projection_array(halo_ids, data, params=params,
                                length_scale="all left", width=width)
    fig.patch.set_facecolor("black")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"      saved  ({time.time() - t0:.1f} s)")
    return Image.open(out_path).convert("RGB")


def load_render_set(renders_dir, prefix, field_keys=None):
    """
    Load a set of pre-rendered field PNGs into a dict {field_key: PIL_RGB}.

    Expected filenames: {renders_dir}/{prefix}_{field_key}.png

    Parameters
    ----------
    renders_dir : str
    prefix      : str   Common filename prefix (e.g. "R1_tag12345")
    field_keys  : list of str, optional.  Defaults to FIELD_ORDER_DEFAULT.

    Returns
    -------
    dict  field_key → PIL RGB Image  (missing files are placeholder dark images)
    """
    if field_keys is None:
        field_keys = FIELD_ORDER_DEFAULT
    result = {}
    for k in field_keys:
        p = os.path.join(renders_dir, f"{prefix}_{k}.png")
        if os.path.exists(p):
            result[k] = Image.open(p).convert("RGB")
        else:
            result[k] = Image.new("RGB", (512, 512), (8, 10, 18))
    return result


# ── Frame I/O & video encoding ─────────────────────────────────────────────────

def save_frames(gen, frames_dir, start_idx=0, label="scene"):
    """
    Write frames from a PIL-image generator to numbered PNGs in frames_dir.

    Parameters
    ----------
    gen        : iterator yielding PIL RGB Images
    frames_dir : str   Output directory (created if needed)
    start_idx  : int   Starting frame index
    label      : str   Label used in progress messages

    Returns
    -------
    (next_frame_index, frames_written_count)
    """
    os.makedirs(frames_dir, exist_ok=True)
    t0    = time.time()
    count = 0
    idx   = start_idx
    for frame in gen:
        frame.save(os.path.join(frames_dir, f"frame_{idx:06d}.png"),
                   compress_level=1)
        idx   += 1
        count += 1
        if count % 100 == 0:
            print(f"    [{label}] {count} frames  ({time.time() - t0:.1f} s)")
    print(f"  [done] {label}: {count} frames  ({time.time() - t0:.1f} s)")
    return idx, count


def encode_video(frames_dir, output_path, fps=FPS, w=W, h=H, crf=16):
    """
    Encode a PNG frame sequence to H264 MP4 via ffmpeg.

    Parameters
    ----------
    frames_dir  : str   Directory containing frame_NNNNNN.png files
    output_path : str   Destination .mp4 path
    fps         : int   Frame rate (default 24)
    w, h        : int   Output resolution (default 1920×1080)
    crf         : int   H264 quality (lower = better; 16 is near-lossless)
    """
    print(f"\n=== Encoding → {output_path} ===")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i",  os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-crf",    str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-vf",  f"scale={w}:{h}",
        output_path,
    ]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        mb = os.path.getsize(output_path) / 1e6
        print(f"  OK → {output_path}  ({mb:.1f} MB)")
    else:
        print(f"  ERROR:\n{result.stderr[-800:]}")
