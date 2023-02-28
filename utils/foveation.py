from matplotlib import pyplot as plt
import numpy as np
from typing import *

import torch


def _get_num_ring_pixels_from_r(r):
    """Get number of pixels in a square ring of given radius centered at a given point"""
    return ((r * 2) - 1) * 4


def _get_indices_in_ring(x_extent, y_extent, x_center, y_center, radius):
    """Get indices of pixels in a square ring of given radius centered at (x_center, y_center)"""
    x, y = np.meshgrid(np.arange(x_extent), np.arange(y_extent), indexing="ij")
    xy = np.stack([x, y], axis=-1)

    # get indices of pixels in ring
    ring_mask = np.zeros((x_extent, y_extent), dtype=bool)
    ring_mask[x_center - radius : x_center + radius, y_center - radius : y_center + radius] = True
    ring_mask[
        x_center - radius + 1 : x_center + radius - 1, y_center - radius + 1 : y_center + radius - 1
    ] = False

    # plt.imshow(ring_mask)

    # get xy according to ring_mask
    ring_xy = xy[ring_mask]

    return ring_xy


def get_generic_sampled_ring_indices(
    x_extent, y_extent, x_center, y_center, fovea_radius, max_ring_radius, num_peri_rings_to_attempt
):
    """Get indices of pixels in a square foveated image of given radius centered at (x_center, y_center)"""

    # get peri indices
    a = np.exp((np.log(max_ring_radius / fovea_radius)) / num_peri_rings_to_attempt)
    peri_ring_radii = np.floor(
        [fovea_radius * a**i for i in range(1, num_peri_rings_to_attempt + 1)]
    )
    eligible_radius_mask = np.array(
        [fovea_radius < r <= max_ring_radius for r in peri_ring_radii]
    ).astype(bool)
    peri_ring_radii = peri_ring_radii[eligible_radius_mask].astype(int)
    num_peri_rings = len(peri_ring_radii)

    foveated_im_dim = 2 * (fovea_radius + num_peri_rings)

    if num_peri_rings != num_peri_rings_to_attempt:
        print(
            f"WARNING: {num_peri_rings_to_attempt} peri rings requested, but only {num_peri_rings} eligible. "
            f"Resulting size will be {foveated_im_dim}x{foveated_im_dim}"
        )

    # get fovea indices
    fovea_image_indices = np.concatenate(
        [
            _get_indices_in_ring(x_extent, y_extent, x_center, y_center, r)
            for r in range(1, fovea_radius + 1)
        ]
    )
    fovea_mapped_indices = np.concatenate(
        [
            _get_indices_in_ring(
                foveated_im_dim, foveated_im_dim, foveated_im_dim // 2, foveated_im_dim // 2, r
            )
            for r in range(1, fovea_radius + 1)
        ]
    )

    assert fovea_image_indices.shape == fovea_mapped_indices.shape

    # sampling ratios for each ring
    peri_sampling_ratios = [
        _get_num_ring_pixels_from_r(fovea_radius + i + 1) / _get_num_ring_pixels_from_r(r)
        for i, r in enumerate(peri_ring_radii)
    ]

    # sample evenly from each ring according to sampling ratio
    peri_image_indices = []
    peri_mapped_indices = []
    for i, r in enumerate(peri_ring_radii):
        foveated_r = fovea_radius + i + 1
        ri_image = _get_indices_in_ring(x_extent, y_extent, x_center, y_center, r)
        ri_fov = _get_indices_in_ring(
            foveated_im_dim, foveated_im_dim, foveated_im_dim // 2, foveated_im_dim // 2, foveated_r
        )

        # traverse ring indices clockwise starting from top left using arctan2
        ri_image = ri_image[
            np.argsort(np.arctan2(ri_image[:, 0] - x_center, ri_image[:, 1] - y_center))
        ]
        ri_fov = ri_fov[
            np.argsort(
                np.arctan2(ri_fov[:, 0] - foveated_im_dim // 2, ri_fov[:, 1] - foveated_im_dim // 2)
            )
        ]

        # sample according to sampling ratio
        sample_n = np.floor(len(ri_image) * peri_sampling_ratios[i]).astype(int)
        sample = np.round(np.linspace(0, len(ri_image), sample_n, endpoint=False)).astype(int)
        ri_image = ri_image[sample]

        peri_image_indices.append(ri_image)
        peri_mapped_indices.append(ri_fov)

    # all_indices = np.concatenate([fovea_indices, *sampled_indices])

    return dict(
        foveated_image_size=(foveated_im_dim, foveated_im_dim),
        source_indices={
            "fovea": fovea_image_indices,
            "peripheral_rings": peri_image_indices,
        },
        mapped_indices={
            "fovea": fovea_mapped_indices,
            "peripheral_rings": peri_mapped_indices,
        },
    )


def get_default_gaussian_foveation_filter_params(
    image_dim: List[int],
    fovea_radius: int,
    image_out_dim: int,
    ring_sigma_scaling_factor=1,
    device=None,
):  # ring_sigmas: List[float]):
    """Get default gaussian foveation filter params (mus and sigmas) for a given image size and fovea radius

    Args:
        image_dim: [height, width] of image
        fovea_radius: radius of fovea in pixels
        image_out_dim: output image dimension (must be even)
        ring_sigma_scaling_factor: scaling factor for consequent ring sigmas (default: 1)

    Returns:

    """
    h, w = image_dim

    # assert all(0 <= xy_center[:, 0] < w), f"xy_center[0] must be in [0, {w}) (got: {xy_center[:, 0]})"
    # assert all(0 <= xy_center[:, 1] < h), f"xy_center[1] must be in [0, {h}) (got: {xy_center[:, 1]})"
    # assert h % 2 == w % 2 == 0, f"Image must be even-sized (got: {image.shape})"
    assert fovea_radius % 2 == 0, f"Fovea radius must be even (got: {fovea_radius})"
    assert image_out_dim % 2 == 0, f"Image out dim must be even (got: {image_out_dim})"

    num_peri_rings = int(image_out_dim / 2) - fovea_radius
    max_ring_radius = min(h, w) // 2

    generic_center_x, generic_center_y = w // 2, h // 2
    generic_ring_specs = get_generic_sampled_ring_indices(
        x_extent=w,
        y_extent=h,
        x_center=generic_center_x,
        y_center=generic_center_y,
        fovea_radius=fovea_radius,
        max_ring_radius=max_ring_radius,
        num_peri_rings_to_attempt=num_peri_rings,
    )

    fov_h, fov_w = generic_ring_specs["foveated_image_size"]

    ring_sigmas = [ring_sigma_scaling_factor * r for r in range(1, num_peri_rings + 1)]

    # z_fig, z_ax = plt.subplots(
    #     1, len(foveation_params["source_indices"]["peripheral_rings"]), figsize=(5, 5)
    # )

    gaussian_foveation_params = dict(
        # image_dim=image_dim,
        foveated_image_dim=(fov_h, fov_w),
        fovea={
            "mus": torch.tensor(
                generic_ring_specs["source_indices"]["fovea"], dtype=torch.float32, device=device
            ).unsqueeze(0),
            "sigmas": torch.tensor(
                [0] * len(generic_ring_specs["source_indices"]["fovea"]),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "target_indices": torch.tensor(
                generic_ring_specs["mapped_indices"]["fovea"], dtype=torch.int, device=device
            ),
        },
        peripheral_rings=[
            {
                "mus": torch.tensor(ri, dtype=torch.float32, device=device).unsqueeze(0),
                "sigmas": torch.tensor(sigmas, dtype=torch.float32, device=device).unsqueeze(0),
                "target_indices": torch.tensor(t, dtype=torch.int, device=device),
            }
            for ri, sigmas, t in zip(
                generic_ring_specs["source_indices"]["peripheral_rings"],
                ring_sigmas,
                generic_ring_specs["mapped_indices"]["peripheral_rings"],
            )
        ],
    )

    return gaussian_foveation_params


Z_EPS = 1e-2


def apply_gaussian_foveation(image: torch.Tensor, foveation_params: dict):
    """Sample image according to foveation params, sampling each peripheral point based on a Gaussian function
    of its distance to other points in the image
    Sampling is done via matrix-multiplication filtering
    """

    b, c, h, w = image.size()
    fov_h, fov_w = foveation_params["foveated_image_dim"]

    # build filters
    foveation_filters = torch.zeros((b, fov_h, fov_w, h, w), device=image.device, dtype=image.dtype)

    # # fovea filters
    # foveation_filters[
    #     generic_ring_specs["mapped_indices"]["fovea"][:, 0],
    #     generic_ring_specs["mapped_indices"]["fovea"][:, 1],
    #     generic_ring_specs["source_indices"]["fovea"][:, 0],
    #     generic_ring_specs["source_indices"]["fovea"][:, 1],
    # ] = 1

    x = torch.arange(0, w, device=image.device)
    y = torch.arange(0, h, device=image.device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # expand batch dim and ring_n dims
    xx = xx.unsqueeze(0).unsqueeze(-1).to(torch.float)
    yy = yy.unsqueeze(0).unsqueeze(-1).to(torch.float)

    for i, ring in enumerate([foveation_params["fovea"], *foveation_params["peripheral_rings"]]):
        mu = ring["mus"].unsqueeze(1).unsqueeze(1).to(torch.float) # unsqueeze to add h, w dims
        sigma = ring["sigmas"]
        target_indices = ring["target_indices"]

        # build gaussian
        # TODO: check formula
        # TODO: clip to relevant region only
        z = torch.exp(-((xx - mu[:, :, :, :, 0]) ** 2 + (yy - mu[:, :, :, :, 1]) ** 2) / (2 * sigma**2 + Z_EPS))
        z = z / torch.sum(z, axis=(1, 2), keepdim=True) # normalize
        # z_img = z_ax[i].imshow(z.sum(axis=2))
        # z_fig.colorbar(z_img)

        foveation_filters[:,
            target_indices[:, 0],
            target_indices[:, 1],
        ] = z.permute(0, 3, 1, 2) # permute z to (b, ring_n, h, w)

    # filter_fig, filter_ax = plt.subplots(1, 1, figsize=(5, 5))
    # g = filter_ax.imshow(foveation_filters[0].sum(axis=(0, 1)))
    # filter_fig.colorbar(g)
    # plt.show()
    # apply filters
    _img = image.view((b, c, h * w)).transpose(1, 2)
    _filters = foveation_filters.view((b, fov_h * fov_w, h * w))
    foveated_image = torch.matmul(_filters, _img).view((b, fov_h, fov_w, c)).permute(0, 3, 1, 2)

    return foveated_image


# def sample_foveation_gaussian(image: torch.Tensor, xy_center: torch.Tensor, fovea_radius: int, image_out_dim: int, ring_sigmas: List[float]):
#     """Sample image according to foveation params, sampling each peripheral point based on a Gaussian function
#     of its distance to other points in the image
#     Sampling is done via matrix-multiplication filtering
#     """
#     b, c, h, w = image.shape

#     assert all(0 <= xy_center[:, 0] < w), f"xy_center[0] must be in [0, {w}) (got: {xy_center[:, 0]})"
#     assert all(0 <= xy_center[:, 1] < h), f"xy_center[1] must be in [0, {h}) (got: {xy_center[:, 1]})"
#     assert h % 2 == w % 2 == 0, f"Image must be even-sized (got: {image.shape})"
#     assert fovea_radius % 2 == 0, f"Fovea radius must be even (got: {fovea_radius})"
#     assert image_out_dim % 2 == 0, f"Image out dim must be even (got: {image_out_dim})"

#     num_peri_rings = int(image_out_dim / 2) - fovea_radius

#     generic_center = torch.tensor([w // 2, h // 2], device=xy_center.device)
#     foveation_params = get_generic_sampled_ring_indices(
#         x_extent=w,
#         y_extent=h,
#         x_center=generic_center[0],
#         y_center=generic_center[1],
#         fovea_radius=fovea_radius,
#         max_ring_radius=min(h, w) // 2,
#         num_peri_rings_to_attempt=num_peri_rings,
#     )

#     def move_foveation(xy_center, foveation_params):
#         fovea_source_indices = foveation_params["source_indices"]["fovea"] - generic_center + xy_center
#         peri_rings_source_indices = [
#             ri - generic_center + xy_center for ri in foveation_params["source_indices"]["peripheral_rings"]
#         ]

#         return fovea_source_indices, peri_rings_source_indices

#     fovea_source_indices, peri_rings_source_indices = move_foveation(xy_center, foveation_params)
#     fovea_mapped_indices, peri_rings_mapped_indices = foveation_params["mapped_indices"]["fovea"], foveation_params["mapped_indices"]["peripheral_rings"]


#     fov_h, fov_w = foveation_params["foveated_image_size"]

#     assert len(ring_sigmas) == len(
#         foveation_params["source_indices"]["peripheral_rings"]
#     ), f"Must provide mu and sigma for each of {len(foveation_params['source_indices']['peripheral_rings'])} rings"

#     # build filters
#     foveation_filters = np.zeros((fov_h, fov_w, h, w))

#     # fovea filters
#     foveation_filters[
#         foveation_params["mapped_indices"]["fovea"][:, 0],
#         foveation_params["mapped_indices"]["fovea"][:, 1],
#         foveation_params["source_indices"]["fovea"][:, 0],
#         foveation_params["source_indices"]["fovea"][:, 1],
#     ] = 1

#     # z_fig, z_ax = plt.subplots(
#     #     1, len(foveation_params["source_indices"]["peripheral_rings"]), figsize=(5, 5)
#     # )
#     for i, ri in enumerate(foveation_params["source_indices"]["peripheral_rings"]):
#         # build gaussian
#         x = np.arange(0, w)
#         y = np.arange(0, h)
#         xx, yy = np.meshgrid(x, y, indexing="ij")
#         mu = ri
#         sigma = ring_sigmas[i]

#         # build gaussian
#         # TODO: check formula
#         z = np.exp(
#             -((np.expand_dims(xx, -1) - mu[:, 0]) ** 2 + (np.expand_dims(yy, -1) - mu[:, 1]) ** 2)
#             / (2 * sigma**2)
#         )
#         z = z / np.sum(z, axis=(0, 1))
#         # z_img = z_ax[i].imshow(z.sum(axis=2))
#         # z_fig.colorbar(z_img)

#         foveation_filters[
#             foveation_params["mapped_indices"]["peripheral_rings"][i][:, 0],
#             foveation_params["mapped_indices"]["peripheral_rings"][i][:, 1],
#         ] = z.transpose(2, 0, 1)

#     # filter_fig, filter_ax = plt.subplots(1, 1, figsize=(5, 5))
#     # g = filter_ax.imshow(foveation_filters.sum(axis=(0, 1)))
#     # filter_fig.colorbar(g)

#     # apply filters
#     _img = image.reshape((1 * c, h * w)).T
#     _filters = foveation_filters.reshape((fov_h * fov_w, h * w))
#     foveated_image = np.matmul(_filters, _img).reshape((fov_h, fov_w, c)).transpose(2, 0, 1)

#     return foveated_image