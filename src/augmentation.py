"""Scripts for augmenting images for training."""

import numpy as np
import numpy.typing as npt


def zoom_and_shift(
    image: npt.NDArray[np.float64], ground_truth: npt.NDArray[np.bool_], max_zoom_percentage: float = 0.1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Scale and translate image and corresponding ground truth mask.

    Zooms in on the image/mask by a random amount between 0 and
    max_zoom_percentage, then shifts the image/mask by a random amount
    up to the number of zoomed pixels.

    Parameters
    ----------
    image : npt.NDArray[np.float64]
        Image, can be WxH or WxHxC.
    ground_truth : npt.NDArray[np.bool_]
        Mask, can be WxH or WxHxC.
    max_zoom_percentage : float
        Maximum zoom percentage.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]
        Zoomed and shifted image and mask.
    """
    # Choose a zoom percentage and calculate the number of pixels to zoom in
    zoom = np.random.uniform(0, max_zoom_percentage)
    zoom_proportion_pixels = int(image.shape[0] * zoom)

    # If there is zoom, choose a random shift
    if int(zoom_proportion_pixels) > 0:
        # If we only shift at most the number of zoomed pixels, we can't go out of bounds.
        shift_x = np.random.randint(int(-zoom_proportion_pixels), int(zoom_proportion_pixels))
        shift_y = np.random.randint(int(-zoom_proportion_pixels), int(zoom_proportion_pixels))
        assert (
            abs(shift_x) <= zoom_proportion_pixels and abs(shift_y) <= zoom_proportion_pixels
        ), "Shift exceeds zoomed pixels."

        min_row = zoom_proportion_pixels + shift_x
        max_row = -zoom_proportion_pixels + shift_x
        min_col = zoom_proportion_pixels + shift_y
        max_col = -zoom_proportion_pixels + shift_y

        # Zoom and shift the image
        # ... allows us to be agnostic to if the image / mask is WxH or WxHxC, as it ignores the other dimensions.
        image = image[min_row:max_row, min_col:max_col, ...]
        ground_truth = ground_truth[min_row:max_row, min_col:max_col, ...]

    return image, ground_truth


def flip_and_rotate(
    image: npt.NDArray[np.float64],
    ground_truth: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Flip and rotate image and corresponding ground truth mask.

    Flips the image/mask with 50% chance, then rotates to a random
    multiple of 90 degrees.

    Parameters
    ----------
    image : npt.NDArray[np.float64]
        Image, can be WxH or WxHxC.
    ground_truth : npt.NDArray[np.bool_]
        Mask, can be WxH or WxHxC.

    Returns
    -------
    npt.NDArray[np.float64], npt.NDArray[np.bool_]
        Flipped and rotated image and mask.
    """
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1)
        ground_truth = np.flip(ground_truth, axis=1)
    # rotate to random multiple of 90 degrees
    rotation = np.random.randint(0, 4)
    if rotation > 0:
        image = np.rot90(image, k=rotation, axes=(0, 1))
        ground_truth = np.rot90(ground_truth, k=rotation, axes=(0, 1))

    return image, ground_truth
