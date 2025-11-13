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
        Image.
    ground_truth : npt.NDArray[np.bool_]
        Mask.
    max_zoom_percentage : float
        Maximum zoom percentage.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]
        Zoomed and shifted image and mask.
    """
    # Choose a zoom percentage and calculate the number of pixels to zoom in
    zoom = np.random.uniform(0, max_zoom_percentage)
    zoom_pixels = int(image.shape[0] * zoom)

    # If there is zoom, choose a random shift
    if int(zoom_pixels) > 0:
        shift_x = np.random.randint(int(-zoom_pixels), int(zoom_pixels))
        shift_y = np.random.randint(int(-zoom_pixels), int(zoom_pixels))

        # Zoom and shift the image
        image = image[
            zoom_pixels + shift_x : -zoom_pixels + shift_x,
            zoom_pixels + shift_y : -zoom_pixels + shift_y,
        ]
        ground_truth = ground_truth[
            zoom_pixels + shift_x : -zoom_pixels + shift_x,
            zoom_pixels + shift_y : -zoom_pixels + shift_y,
        ]

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
        Image.
    ground_truth : npt.NDArray[np.bool_]
        Mask.

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
    image = np.rot90(image, k=rotation)
    ground_truth = np.rot90(ground_truth, k=rotation)

    return image, ground_truth