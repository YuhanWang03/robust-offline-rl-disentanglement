"""
PPF diagram — final refined version for presentation slides.
Output: docs/ppf_diagram.svg  (and .png for preview)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colors ────────────────────────────────────────────────────────────────────
# Boxes: white fill, strong colored border
C_COR   = "#C0392B"   # corrupted obs  — red
C_ENC   = "#1A5276"   # encoder        — navy blue
C_CLE   = "#1E8449"   # clean state    — green
C_LAT   = "#0E6655"   # latent z       — teal
C_POL   = "#6C3483"   # policy         — purple
C_FRZ   = "#A04000"   # freeze label   — dark orange (muted)
C_DARK  = "#1C2833"   # primary text
C_SUB   = "#5D6D7E"   # subtitle text
C_TITLE = "#FFFFFF"   # panel title text (white on colored strip)

# Panel title strip colors
C_TRAIN_STRIP = "#1A5276"   # navy
C_INFER_STRIP = "#512E5F"   # deep purple

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 18)
ax.set_ylim(0, 6.2)
ax.axis("off")

# ── Helpers ───────────────────────────────────────────────────────────────────

def node(cx, cy, w, h, border_color, label, sub=None, lfs=12.5, sfs=9.5):
    """White-fill box with colored border."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        linewidth=2.2, edgecolor=border_color,
        facecolor="white", zorder=4))
    dy = 0.13 if sub else 0
    ax.text(cx, cy + dy, label,
            ha="center", va="center", fontsize=lfs,
            color=C_DARK, fontweight="bold", zorder=5)
    if sub:
        ax.text(cx, cy - 0.22, sub,
                ha="center", va="center", fontsize=sfs,
                color=C_SUB, style="italic", zorder=5)


def arrow_v(cx, y0, y1, color, lw=1.8):
    ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=6)


def arrow_dashed(x0, x1, cy, color, lw=1.8):
    ax.annotate("", xy=(x1, cy), xytext=(x0, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12,
                                linestyle=(0, (5, 3))), zorder=6)


def panel(x0, y0, x1, y1, strip_color, title):
    """Light grey panel background + solid colored title strip at top."""
    # Main panel — very light grey
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        linewidth=1.2, edgecolor="#CCCCCC",
        facecolor="#F8F9FA", zorder=1))
    # Title strip
    strip_h = 0.52
    ax.add_patch(FancyBboxPatch(
        (x0, y1 - strip_h), x1 - x0, strip_h,
        boxstyle="round,pad=0.0,rounding_size=0.0",
        linewidth=0, edgecolor="none",
        facecolor=strip_color, zorder=2,
        clip_on=True))
    # Rounded top corners via overlay of the main panel clipping
    ax.add_patch(FancyBboxPatch(
        (x0, y1 - strip_h), x1 - x0, strip_h,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        linewidth=0, edgecolor="none",
        facecolor=strip_color, zorder=2))
    ax.text((x0 + x1) / 2, y1 - strip_h / 2, title,
            ha="center", va="center", fontsize=13,
            color=C_TITLE, fontweight="bold", zorder=3)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layout constants
# ═══════════════════════════════════════════════════════════════════════════════
TX   = 3.80    # training column center
IX   = 13.85   # inference column center
BW   = 5.0     # main box width
BH   = 0.76    # main box height

PANEL_Y0, PANEL_Y1 = 0.45, 6.15

Y_OBS = 5.18
Y_ENC = 3.92
Y_LAT = 2.66
Y_POL = 1.40

# Clean State: smaller auxiliary box to the right of Encoder
CS_W  = 1.10
CS_H  = 0.82
CS_CX = (TX + BW/2) + 0.28 + CS_W/2   # just right of encoder: 6.30+0.28+0.55=7.13
CS_CY = Y_ENC

Y_FRZ = (Y_ENC + Y_LAT) / 2    # 3.29  — center of gap between encoder and latent

# ── Panels ────────────────────────────────────────────────────────────────────
panel(0.30, PANEL_Y0,  8.00, PANEL_Y1, C_TRAIN_STRIP, "Training Phase")
panel(10.0, PANEL_Y0, 17.70, PANEL_Y1, C_INFER_STRIP, "Inference Phase")

# ── Freeze transition: small italic label on the arrow, no decorative icon ────
ax.annotate("", xy=(10.0, Y_FRZ), xytext=(8.0, Y_FRZ),
            arrowprops=dict(arrowstyle="-|>", color=C_FRZ,
                            lw=1.8, mutation_scale=13), zorder=6)
ax.text(9.0, Y_FRZ - 0.26, "freeze after pretraining",
        ha="center", va="top", fontsize=9.5,
        color=C_FRZ, style="italic", zorder=7)

# ═══════════════════════════════════════════════════════════════════════════════
#  Training Phase
# ═══════════════════════════════════════════════════════════════════════════════
node(TX, Y_OBS, BW, BH, C_COR,
     "Corrupted Observation", "synthetic corruption")

node(TX, Y_ENC, BW, BH, C_ENC,
     "Encoder  (trainable)", "pretrained with privileged target")

node(TX, Y_LAT, BW, BH, C_LAT,
     "Task-Relevant Latent  z", "features for policy learning")

# Clean State — smaller auxiliary box
node(CS_CX, CS_CY, CS_W, CS_H, C_CLE,
     "Clean State",
     "privileged target\n(training only)",
     lfs=9.5, sfs=8.5)

# Dashed arrow: Clean State → Encoder (right edge to right edge)
enc_right = TX + BW / 2
cs_left   = CS_CX - CS_W / 2
arrow_dashed(cs_left, enc_right, Y_ENC, C_CLE, lw=1.6)

arrow_v(TX, Y_OBS - BH/2, Y_ENC + BH/2, C_COR)
arrow_v(TX, Y_ENC - BH/2, Y_LAT + BH/2, C_ENC)

# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Phase
# ═══════════════════════════════════════════════════════════════════════════════
node(IX, Y_OBS, BW, BH, C_COR,
     "Corrupted Observation", "no clean state at inference")

node(IX, Y_ENC, BW, BH, C_ENC,
     "Frozen Encoder", "weights fixed after pretraining")

node(IX, Y_LAT, BW, BH, C_LAT,
     "Task-Relevant Latent  z", "features for policy learning")

node(IX, Y_POL, BW, BH, C_POL,
     "Offline RL Policy", "IQL  /  TD3+BC  /  BC")

arrow_v(IX, Y_OBS - BH/2, Y_ENC + BH/2, C_COR)
arrow_v(IX, Y_ENC - BH/2, Y_LAT + BH/2, C_ENC)
arrow_v(IX, Y_LAT - BH/2, Y_POL + BH/2, C_LAT)

# ── Save ──────────────────────────────────────────────────────────────────────
for fmt in ("svg", "png"):
    plt.savefig(f"docs/ppf_diagram.{fmt}", format=fmt, dpi=180,
                bbox_inches="tight", transparent=False)
    print(f"Saved: docs/ppf_diagram.{fmt}")

plt.close()
