"""IEEE-conference matplotlib font/sizes — call set_ieee_font() before plotting.

ieeeconf.cls renders body text in Times Roman. We match by setting matplotlib
to the same serif family with sizes that match the paper's body type.
"""
from __future__ import annotations


def set_ieee_font():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Sizes tuned so axis labels and legends remain readable when the figure
        # is downscaled to a single IEEE column (~3.5 in wide).
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 12,
    })


# Okabe-Ito colorblind-safe palette, shared across all method-comparison figures
# (teaser, loss curves, MuJoCo teasers) so the same method always gets the same
# color and adjacent hues stay separable under deuteranopia/protanopia and in
# grayscale print (paired with distinct linestyles, never color alone).
METHOD_COLORS = {"Vanilla": "#0072B2", "Frame-stack": "#009E73", "FlyAda": "#D55E00"}
METHOD_LINESTYLES = {"Vanilla": "--", "Frame-stack": (0, (1, 1.3)), "FlyAda": "-"}

# 5-way condition palette for the latent-analysis PCA figure (nominal + 4 axes).
CONDITION_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
