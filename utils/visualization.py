import io
import matplotlib.pyplot as plt
import numpy as np
import torch

def fig_to_nparray(fig):
    """Convert a Matplotlib figure to a numpy array with RGBA channels and return it."""
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
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
