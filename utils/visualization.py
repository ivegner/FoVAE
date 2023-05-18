import io
from typing import List
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import torch


def fig_to_nparray(fig):
    """Convert a Matplotlib figure to a numpy array with RGBA channels and return it."""
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


# functions to show an image
def imshow_unnorm(img: torch.Tensor, ax=None):
    img = img / 2 + 0.5
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    if ax:
        ax.imshow(npimg, extent=(0, npimg.shape[0], npimg.shape[1], 0))
    else:
        plt.imshow(npimg, extent=(0, npimg.shape[0], npimg.shape[1], 0))


def plot_gaussian_foveation_parameters(
    images: torch.Tensor, foveation_parameters: dict, axs=None, point_size=1
):
    assert images.ndim == 4, "Image must be in BCHW format"

    if axs is not None:
        fig = None
        assert len(axs) == images.shape[0], "Number of axes provided must match number of images"
    else:
        fig, axs = plt.subplots(1, images.shape[0], figsize=(10, 10))

    for i, (image, ax) in enumerate(zip(images, axs)):
        imshow_unnorm(image, ax=ax)

        # fovea_points = foveation_parameters["fovea"]["mus"][i]

        # ax.scatter(fovea_points[:, 0], fovea_points[:, 1], s=point_size, label=f"Fovea")
        for ring_i, ring_specs in enumerate(
            [foveation_parameters["fovea"], *foveation_parameters["peripheral_rings"]]
        ):
            sigmas = (
                1 if ring_i == 0 else ring_specs["sigmas"].mean()
            )  # sigmas the same for all points in ring for every item in batch

            # plot translucent rectangle centered at mu with width and height 2*sigma
            rect_size = sigmas / 2
            for mu in ring_specs["mus"][i]:
                rect = plt.Rectangle(
                    (mu[0] - rect_size, mu[1] - rect_size),
                    2 * rect_size,
                    2 * rect_size,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

            # ax.scatter(
            #     ring_specs["mus"][i, :, 0],
            #     ring_specs["mus"][i, :, 1],
            #     s=sigmas * point_size,
            #     label=f"Ring {ring_i}",
            # )

        # put text of foveal central point in top left corner
        central_foveal_point = foveation_parameters["fovea"]["mus"][i].mean(dim=0)
        ax.text(
            0.05,
            0.05,
            f"({central_foveal_point[0]:.2f}, {central_foveal_point[1]:.2f})",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5),
            # align corner of text to corner of image
            horizontalalignment="left",
            verticalalignment="top",
        )

        # reset axes to show full image with boxes
        ax.set_xlim(min(0, ax.dataLim.x0), max(image.shape[0], ax.dataLim.x1))
        ax.set_ylim(
            max(image.shape[1], ax.dataLim.y1),
            min(0, ax.dataLim.y0),
        )

    return fig, axs


def plot_layer_kl_history_by_dim(kls_by_layer: List[np.ndarray], epoch_indices: List[int]):
    """Plot KL divergence history for each layer in the network, for each dimension of latent space.

    Args:
        kls_by_layer (List[List[np.ndarray]]): KL divergence history, of shape
            (num_layers x num_epochs x num_dims)
        epoch_indices (List[int]): List of epoch indices to use as x-axis
    """
    fig, axs = plt.subplots(len(kls_by_layer), 1, figsize=(6, len(kls_by_layer)*3))
    for layer_i, ax in enumerate(axs):
        a = np.array(kls_by_layer[layer_i]).T
        g = ax.pcolormesh(
            a,
            cmap="Blues",
            shading="auto",
            norm=LogNorm(vmin=max(a.min(), 0.01), vmax=max(a.max(), 0.01)),
        )
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Epoch")
        xticks = list(zip(np.arange(len(epoch_indices)) + 0.5, epoch_indices))
        # subsample xticks
        xticks = xticks[:: max(1, len(xticks) // 10)]
        ax.set_xticks(*zip(*xticks), minor=False)
        ax.set_ylabel("Latent Unit")
        plt.colorbar(g)

    fig.tight_layout()

    return fig
