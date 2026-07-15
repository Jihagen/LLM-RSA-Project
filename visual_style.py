"""Shared visual roles for the study figures.

The hues are derived from the original report palette, but color is never
allowed to mean a generic "group A/group B".  Each role below keeps the same
meaning across figures; marker fill and line style carry context/position.
"""

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SENSE = {0: "#4A3AA7", 1: "#D59A18"}  # violet, report-derived amber
ARCHITECTURE = {"encoder": "#277F70", "decoder": "#8A607C"}
METHOD = {
    "adequacy": "#303038",
    "gdv": "#8A6396",
    "oracle": "#111111",
    "final": "#898781",
    "supervised": "#277F70",
}
OUTCOME = {
    "correct": "#4F8A62",
    "ambiguous": "#A6A39B",
    "wrong": "#B65D50",
}
AGREEMENT = {"agree": "#496E5B", "disagree": "#E9E3D7"}
INK = "#242329"
MUTED = "#77746E"
GRID = "#E7E4DE"
BACKGROUND = "#FFFFFF"
NEUTRAL = "#F5F2EC"

PRIOR_CMAP = LinearSegmentedColormap.from_list(
    "sense_prior",
    [SENSE[1], "#F7F4EE", SENSE[0]],
)


def apply_report_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )
