import io
from typing import List, Any, Dict
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import torch

import numpy as np
import torch
from torchvision.utils import make_grid

from utils.misc import recursive_to


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
    fig, axs = plt.subplots(len(kls_by_layer), 1, figsize=(6, len(kls_by_layer) * 3))
    for layer_i, ax in enumerate(axs):
        a = np.array(kls_by_layer[layer_i]).T
        # replace nan with 0
        a = np.nan_to_num(a)
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
        xticks = ([None] + xticks)[:: int(10 ** np.floor(np.log10(len(xticks))))][1:]
        ax.set_xticks(*zip(*xticks), minor=False)
        ax.set_ylabel("Latent Unit")
        plt.colorbar(g)

    fig.tight_layout()

    return fig


def _remove_pos_channels_from_batch(g):
    n_pos_channels = 2  # if model.do_add_pos_encoding else 0
    return g[:, :-n_pos_channels, :, :]


def visualize_model_output(
    model,
    x,
    model_out: Dict[str, Dict[str, Any]],
    n_to_plot: int = 4,
):
    step_sample_zs = model_out["step_vars"]["real_patch_zs"]
    # step_z_recons = forward_out["step_vars"]["z_recons"]
    step_next_z_preds = model_out["step_vars"]["gen_patch_zs"]
    patches = model_out["step_vars"]["patches"]
    step_patch_positions = model_out["step_vars"]["patch_positions"]

    # step_sample_zs: (n_steps, n_layers, batch_size, z_dim)
    assert (
        patches[0][0].size()
        == step_sample_zs[0][0][0].size()
        == torch.Size([model.num_channels * model.patch_dim * model.patch_dim])
    )
    assert step_sample_zs[0][1][0].size() == torch.Size([model.z_dims[0]])

    real_images = make_grid(
        # _remove_pos_channels_from_batch(
        #     x[:n_to_plot].view(-1, model.num_channels, model.patch_dim, model.patch_dim)
        #      / 2
        #      + 0.5
        # ).cpu(),
        x[:n_to_plot].cpu() / 2 + 0.5,
        nrow=8,
        padding=1,
    )

    # real = make_grid(x).cpu()
    # recon = make_grid(x_recon).cpu()
    # img = torch.concat((real, recon), dim=1)

    fov_vis = _visualize_foveations(
        model,
        x,
        step_sample_zs,
        step_next_z_preds,
        patches,
        step_patch_positions,
        n_to_plot=n_to_plot,
    )

    # if model.do_image_reconstruction:
    _, reconstructed_images = model._reconstruct_image(
        [[level[:n_to_plot].to(model.device) for level in step] for step in step_sample_zs],
        image=None,
        return_patches=True,
    )
    reconstructed_images = reconstructed_images.cpu()

    reconstructed_images = [
        make_grid(
            _remove_pos_channels_from_batch(model._patch_to_fovea(r)) / 2 + 0.5,
            nrow=int(np.sqrt(len(r))),
            padding=1,
        )
        for r in reconstructed_images
    ]

    real_patches_grid = make_grid(
        _remove_pos_channels_from_batch(
            patches[0][:32].view(-1, model.num_channels, model.patch_dim, model.patch_dim) / 2 + 0.5
        ).cpu(),
        nrow=8,
        padding=1,
    )

    reconstructed_patches_grid = make_grid(
        _remove_pos_channels_from_batch(
            step_sample_zs[0][0][:32].view(-1, model.num_channels, model.patch_dim, model.patch_dim)
            / 2
            + 0.5
        ).cpu(),
        nrow=8,
        padding=1,
    )

    def stack_traversal_output(g):
        # stack by interp image, then squeeze out the singular batch dimension and
        # index out the 2 position channels
        return [_remove_pos_channels_from_batch(torch.stack(dt).squeeze(1)) for dt in g]

        # img = model._add_pos_encodings_to_img_batch(x[[0]])
        # get top-level z of first step of first image of batch.

    z_level = -1
    first_step_zs = step_sample_zs[0][z_level][0].unsqueeze(0)
    traversal_abs = model.latent_traverse(first_step_zs, z_level=z_level, range_limit=3, step=0.5)
    images_by_row_and_interp = stack_traversal_output(traversal_abs)

    abs_latent_traversal_grid = make_grid(
        torch.concat(images_by_row_and_interp),
        nrow=images_by_row_and_interp[0].size(0),
    )

    traversal_around = model.latent_traverse(
        first_step_zs, z_level=z_level, range_limit=3, step=0.5, around_z=True
    )
    images_by_row_and_interp = stack_traversal_output(traversal_around)

    around_latent_traversal_grid = make_grid(
        torch.concat(images_by_row_and_interp),
        nrow=images_by_row_and_interp[0].size(0),
    )

    return {
        "real_images": real_images,
        "fov_vis": fov_vis,
        "reconstructed_images": reconstructed_images,
        "real_patches_grid": real_patches_grid,
        "reconstructed_patches_grid": reconstructed_patches_grid,
        "abs_latent_traversal_grid": abs_latent_traversal_grid,
        "around_latent_traversal_grid": around_latent_traversal_grid,
    }


def _visualize_foveations(
    model, x, step_sample_zs, step_next_z_preds, patches, step_patch_positions, n_to_plot: int = 4
):
    real_images = x[:n_to_plot].repeat(model.num_steps, 1, 1, 1, 1)
    # plot stepwise foveations on real images
    h, w = real_images.shape[3:]

    # # # # DEBUG: demo foveation to a specific location
    # fig, (ax1, ax2) = plt.subplots(2)
    # loc = torch.tensor([0.0, 0.0]).repeat(1, 1).to("mps")
    # gaussian_filter_params = _recursive_to(
    #     model._move_default_filter_params_to_loc(loc, (h, w), pad_offset=None),
    #     "cpu",
    # )
    # plot_gaussian_foveation_parameters(
    #                     x[[3]].cpu(),
    #                     gaussian_filter_params,
    #                     axs=[ax1],
    #                     point_size=10,
    #                 )
    # fov = model._foveate_to_loc(model._add_pos_encodings_to_img_batch(x[[3]]), loc).cpu()
    # imshow_unnorm(fov[0,[0]], ax=ax2)

    # make figure with a column for each step and 3 rows:
    # 1 for image with foveation, one for patch, one for patch reconstruction

    figs = [plt.figure(figsize=(model.num_steps * 3, 12)) for _ in range(n_to_plot)]
    axs = [f.subplots(4, model.num_steps) for f in figs]

    # plot foveations on images
    for step, img_step_batch in enumerate(real_images):
        # positions = (
        #     patches[step]
        #     .view(-1, model.num_channels, model.patch_dim, model.patch_dim)[:n_to_plot, -2:]
        #     .mean(dim=(2, 3))
        # )
        fov_locations_x = torch.stack([g[0].argmax(dim=-1) for g in step_patch_positions], dim=0)
        fov_locations_y = torch.stack([g[1].argmax(dim=-1) for g in step_patch_positions], dim=0)

        positions = torch.cat(
            (fov_locations_x.unsqueeze(-1), fov_locations_y.unsqueeze(-1)), dim=-1
        ).to(model.device)

        positions = (positions[step] / h) * 2 - 1

        # positions = step_patch_positions[step].to(model.device)
        gaussian_filter_params = recursive_to(
            model._move_default_filter_params_to_loc(positions, (h, w), pad_offset=None),
            "cpu",
        )
        plot_gaussian_foveation_parameters(
            img_step_batch.cpu(),
            gaussian_filter_params,
            axs=[a[0][step] for a in axs],
            point_size=10,
        )
        for ax in [a[0][step] for a in axs]:
            ax.set_title(f"Foveation at step {step}", fontsize=8)

        # plot patches
    for step in range(model.num_steps):
        step_patch_batch = _remove_pos_channels_from_batch(
            patches[step][:n_to_plot].view(-1, model.num_channels, model.patch_dim, model.patch_dim)
        )
        for i in range(n_to_plot):
            imshow_unnorm(step_patch_batch[i].cpu(), ax=axs[i][1][step])
            axs[i][1][step].set_title(f"Patch at step {step}", fontsize=8)

        # plot patch reconstructions
    for step in range(model.num_steps):
        step_patch_batch = _remove_pos_channels_from_batch(
            step_sample_zs[step][0][:n_to_plot].view(
                -1, model.num_channels, model.patch_dim, model.patch_dim
            )
        )
        for i in range(n_to_plot):
            imshow_unnorm(step_patch_batch[i].cpu(), ax=axs[i][2][step])
            axs[i][2][step].set_title(f"Patch reconstruction at step {step}", fontsize=8)

        # plot next patch predictions
    if model.do_next_patch_prediction:
        for step in range(model.num_steps):
            pred_patches = step_next_z_preds[step][0][:n_to_plot].view(
                -1, model.num_channels, model.patch_dim, model.patch_dim
            )
            pred_pos = (pred_patches[:, -2:].mean(dim=(2, 3)) / 2 + 0.5).cpu() * torch.tensor(
                [h, w]
            )
            pred_patches = _remove_pos_channels_from_batch(pred_patches)
            for i in range(n_to_plot):
                ax = axs[i][3][step]
                imshow_unnorm(pred_patches[i].cpu(), ax=ax)
                ax.set_title(
                    f"Next patch pred. at step {step} - "
                    f"({pred_pos[i][0]:.1f}, {pred_pos[i][1]:.1f})",
                    fontsize=8,
                )
                # ax.text(
                #     -0.05,
                #     -0.05,
                #     f"(pred: {pred_pos[i][0]:.2f}, {pred_pos[i][1]:.2f})",
                #     color="white",
                #     fontsize=8,
                #     bbox=dict(facecolor="black", alpha=0.5),
                #     horizontalalignment="left",
                #     verticalalignment="top",
                # )

        # add to tensorboard
    for i, fig in enumerate(figs):
        fig.tight_layout()

    fov_vis = [fig_to_nparray(f) for f in figs]

    plt.close("all")

    return fov_vis

    # del (
    #     figs,
    #     real_images,
    #     axs,
    #     reconstructed_images,
    #     images_by_row_and_interp,
    #     traversal_abs,
    #     traversal_around,
    #     step_patch_batch,
    # )
    # if model.do_next_patch_prediction:
    #     del (pred_patches, pred_pos)


def plot_heatmap(x: np.ndarray, ax=None, item_labels: bool = True, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.imshow(x, cmap="Blues")
    ax.set_xticks(np.arange(x.shape[1]))
    ax.set_yticks(np.arange(x.shape[0]))
    ax.set_xticklabels(np.arange(x.shape[1]))
    ax.set_yticklabels(np.arange(x.shape[0]))
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if item_labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ax.text(j, i, f"{x[i, j]:.4f}", ha="center", va="center", color="w", fontsize=6)
    fig.tight_layout()
    return fig, ax
