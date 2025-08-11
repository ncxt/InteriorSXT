import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from context import phantom

from phantom.plt_utils import TEXTWIDTH, metallic_mean, set_size, set_params
from phantom.sampling import Sampling, Fradon_quad_180, LogNorm

SCRIPT_DIR = Path(sys.argv[0]).resolve().parent
EXPORT = SCRIPT_DIR / "figures"

# Dimensions for the Janelia phantom
DETECTOR_WIDTH = 801
RADIUS = DETECTOR_WIDTH // 2
PIXEL_SIZE = 20  # nm

sa35 = Sampling()
sa60 = Sampling()
sa35.init_diff(35 / PIXEL_SIZE)
sa60.init_diff(60 / PIXEL_SIZE)


def make_figure():
    set_params()

    f, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=set_size(TEXTWIDTH, height=8 / 5 * TEXTWIDTH),
        gridspec_kw={"height_ratios": [metallic_mean(1), 1]},
    )

    n_angles = np.pi * RADIUS
    eps = 1e-6
    signal_quad = Fradon_quad_180(DETECTOR_WIDTH, n_angles, RADIUS)
    k = np.arange(0, DETECTOR_WIDTH // 2) / DETECTOR_WIDTH
    signal_otf_quad = signal_quad * sa60.OTF(k)[:, np.newaxis]
    signal_otf_quad = signal_otf_quad[: DETECTOR_WIDTH // 2, :1000]

    signal_plot_kwargs = {
        "norm": LogNorm(vmin=1, vmax=n_angles),
        "aspect": 2.5,
        "origin": "lower",
    }

    im_power = ax1.imshow(signal_otf_quad + eps, **signal_plot_kwargs)
    ax1.set_title(r"$|F_{otf}(k_\theta,k_u)|$")
    # ax1.set_xlim(0,1000)
    # ax1.set_ylim(0,detector_width/2)

    # Customize Y-axis labels (e.g., map image pixels to custom range)
    new_y_labels = np.linspace(0, 0.5, num=5)  # Define custom labels
    ax1.set_yticks(
        np.linspace(0, signal_otf_quad.shape[0], num=5)
    )  # Match to image pixels
    ax1.set_yticklabels([f"{y:.1f}" for y in new_y_labels])  # Apply new labels

    x35, y35 = sa35.angular_sampling(DETECTOR_WIDTH, RADIUS)
    x60, y60 = sa60.angular_sampling(DETECTOR_WIDTH, RADIUS)
    ax2.plot(x60, y60 * 100, label="OZW 60 nm")
    ax2.plot(x35, y35 * 100, label="OZW 35 nm")
    ax2.set_xlim(0, 1000)

    ax1.tick_params(axis="x", bottom=True, labelbottom=False)
    ax2.tick_params(axis="x", top=False, bottom=True, labelbottom=True)

    # ax2.set_xticklabels([])

    cutoff60 = sa60.power_inv(0.98)
    cut_x = int(2 * np.pi * cutoff60 * RADIUS)
    cut_y = DETECTOR_WIDTH * cutoff60
    print(cut_x, cut_y, cutoff60)
    ax1.plot([0, cut_x, cut_x], [cut_y, cut_y, 0], "r")

    ax1.set_ylabel(r"$k_u$")

    ax2.set_ylabel("Total Power [\%]")
    ax2.set_xlabel("Number of angles")
    ax2.legend()

    padh = 0.20
    badb = 0.1

    f.subplots_adjust(top=1 - 0.01, bottom=badb, left=padh, right=1 - padh, hspace=0.05)

    pos1 = ax1.get_position()
    cax = f.add_axes([pos1.x1 + 0.02, pos1.y0, pos1.width * 0.05, pos1.height])
    cbar = f.colorbar(im_power, cax=cax)

    plt.show()
    f.savefig(EXPORT / "sampling_power_double.pdf", format="pdf")


if __name__ == "__main__":
    make_figure()
