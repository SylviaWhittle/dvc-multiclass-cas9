"""Scripts for preprocessing images before passing to the model."""

from PIL import Image
from loguru import logger
import numpy as np
import numpy.typing as npt

# pylint: disable=no-name-in-module
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from keras.utils import to_categorical


def resize_image(
    image: npt.NDArray[np.float64],
    size: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Resize the image to the desired size."""
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(size, resample=Image.NEAREST)
    return np.array(pil_image)


def resize_single_channel_mask(
    mask: npt.NDArray[np.bool_],
    size: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Resize a single channel mask to the desired size."""
    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.resize(size, resample=Image.NEAREST)
    return np.array(pil_mask).astype(bool)


def resize_mask(
    mask: npt.NDArray[np.bool_],
    size: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Resize the mask to the desired size."""
    if len(mask.shape) != 3:
        raise ValueError(f"Mask must have 3 dimensions (H, W, C), got {mask.shape}")

    # Iterate over channels and resize each channel
    mask_channels = []
    for i in range(mask.shape[2]):
        mask_channel = mask[:, :, i]
        resized_channel = resize_single_channel_mask(mask_channel, size)
        mask_channels.append(resized_channel)

    # Stack the channels back together
    resized_mask = np.stack(mask_channels, axis=-1)
    return resized_mask.astype(bool)


def normalise_image(
    image: npt.NDArray[np.float64], norm_upper_bound: float, norm_lower_bound: float
) -> npt.NDArray[np.float64]:
    """Normalise the image to the range [0, 1] based on the provided bounds."""
    # Normalise the image
    image = np.clip(image, norm_lower_bound, norm_upper_bound)
    image = image - norm_lower_bound
    image = image / (norm_upper_bound - norm_lower_bound)
    return image


def apply_hessian_filter(
    image: npt.NDArray[np.float64], hessian_component: str, sigma: int = 1
) -> npt.NDArray[np.float64]:
    """Apply a Hessian filter to the image"""
    hessian_matrix_image = hessian_matrix(image, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    hessian_maximas, hessian_minimas = hessian_matrix_eigvals(hessian_matrix_image)

    if hessian_component == "minima":
        hessian_image = hessian_minimas
    elif hessian_component == "maxima":
        hessian_image = hessian_maximas
    else:
        raise ValueError(f"Invalid hessian_component value: {hessian_component}. Must be 'minima' or 'maxima'.")

    # Normalise the hessian image to [0, 1]
    vmin, vmax = np.percentile(hessian_image, (1.0, 99.0))
    if vmax - vmin == 0:
        # In case of a completely uniform image, return nothing.
        logger.warning("Hessian image has zero variance; returning zeros.")
        return np.zeros_like(hessian_image)
    hessian_image = np.clip(hessian_image, vmin, vmax)
    hessian_image = (hessian_image - vmin) / (vmax - vmin)
    return hessian_image


def preprocess_image(
    image: npt.NDArray[np.float64],
    model_image_size: tuple[int, int],
    norm_upper_bound: float,
    norm_lower_bound: float,
    filter_channels: list[str],
    hessian_component: str,
    hessian_sigma: int,
) -> npt.NDArray[np.float64]:
    """Preprocess the image"""
    # Normalise the image
    image = normalise_image(image, norm_upper_bound, norm_lower_bound)

    image_channels = []

    if "original" in filter_channels:
        image_channels.append(image)
    if "hessian" in filter_channels:
        image_hessian = apply_hessian_filter(image, hessian_component=hessian_component, sigma=hessian_sigma)
        image_channels.append(image_hessian)

    # Resize the images
    for index, image_channel in enumerate(image_channels):
        image_channels[index] = resize_image(image_channel, size=model_image_size)

    # Stack the channels
    image = np.stack(image_channels, axis=-1)

    return image

def preprocess_mask(
    mask: npt.NDArray[np.bool_],
    model_image_size: tuple[int, int],
    channels: int,
) -> npt.NDArray[np.bool_]:
    """Preprocess a mask of dimensions (H, W) or (H, W, C)"""
    # If doing binary segmentation, ensure the mask has a channel dimension
    if len(mask.shape) == 2:
        # Add a channel dimension
        mask = np.expand_dims(mask, axis=-1)
    
    # If doing multi-class segmentation, convert to one-hot encoding
    if channels > 1:
        # If we have a single channel mask with integer labels, convert to one-hot
        if mask.shape[2] == 1:
            mask = to_categorical(mask[:, :, 0], num_classes=channels).astype(bool)
        else:
            # We already have multi-channel binary mask.
            # Check that the number of channels matches
            if mask.shape[2] != channels:
                raise ValueError(f"Mask has {mask.shape[2]} channels, expected {channels} channels.")
    
    mask = resize_mask(mask, model_image_size)
    return mask
