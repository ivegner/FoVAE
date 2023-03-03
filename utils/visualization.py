import matplotlib.pyplot as plt
import numpy as np
import torch


# functions to show an image
def imshow_unnorm(img: torch.Tensor, ax=None):
    img = img / 2 + 0.5
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    if ax:
        ax.imshow(npimg, extent=(0, npimg.shape[0], npimg.shape[1], 0))
    else:
        plt.imshow(npimg, extent=(0, npimg.shape[0], npimg.shape[1], 0))
        plt.show()


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

        fovea_points = foveation_parameters["fovea"]["mus"][i]

        ax.scatter(fovea_points[:, 0], fovea_points[:, 1], s=point_size, label=f"Fovea")
        for ring_i, ring_specs in enumerate(foveation_parameters["peripheral_rings"]):
            sigmas = ring_specs[
                "sigmas"
            ].mean()  # sigmas the same for all points in ring for every item in batch
            ax.scatter(
                ring_specs["mus"][i, :, 0],
                ring_specs["mus"][i, :, 1],
                s=sigmas * point_size,
                label=f"Ring {ring_i}",
            )

        # put text of foveal central point in top left corner
        central_foveal_point = fovea_points.mean(dim=0)
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

    return fig, axs
