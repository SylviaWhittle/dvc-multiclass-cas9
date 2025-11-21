"""Scripts for predicting masks using a trained model."""

from loguru import logger
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from preprocess import preprocess_image

def predict_mask(
    model: tf.keras.Model,
    image: npt.NDArray,
    norm_lower_bound: float,
    norm_upper_bound: float,
    filter_channels: list[str],
    hessian_component: str,
    hessian_sigma: int,
    do_image_preprocessing: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predict the mask for a given image using the trained model."""
    logger.info("Predict: preprocessing image.")
    if do_image_preprocessing:
    # Preprocess the image
        image = preprocess_image(
            image=image,
            model_image_size=(image.shape[0], image.shape[1]),
            norm_upper_bound=norm_upper_bound,
            norm_lower_bound=norm_lower_bound,
            filter_channels=filter_channels,
            hessian_component=hessian_component,
            hessian_sigma=hessian_sigma,
        )
        logger.info("Predict: adding batch dimension.")
        # Add the batch dimension
        image = np.expand_dims(image, axis=0)

    # Predict the mask
    logger.info(f"Predict: predicting for image of shape {image.shape}.")
    mask_predicted = model.predict(image)
    logger.info(f"Predict: stripping batch dimension from prediction.")
    # Remove the batch dimension
    mask_predicted = np.squeeze(mask_predicted, axis=0)
    logger.info(f"Predict: Done. Final prediction shape: {mask_predicted.shape}.")

    # Construct a binary predicted mask from the predicted mask, by choosing the channel with the highest confidence.
    mask_predicted_flat_discrete = np.argmax(mask_predicted, axis=2)

    # same again, but don't use discrete values, just use the float values for the argmax channel
    mask_predicted_flat = np.zeros_like(mask_predicted)
    for i in range(mask_predicted.shape[2]):
        # find the pixels where this channel is the max
        channel_mask = mask_predicted_flat_discrete == i
        mask_predicted_flat[:, :, i][channel_mask] = mask_predicted[:, :, i][channel_mask]

    # Binarise each channel at 0.5 threshold
    mask_predicted_binary = mask_predicted.copy()
    mask_predicted_binary[mask_predicted_binary >= 0.5] = 1
    mask_predicted_binary[mask_predicted_binary < 0.5] = 0
    mask_predicted_binary = mask_predicted_binary.astype(bool)

    return mask_predicted, mask_predicted_binary, mask_predicted_flat_discrete, mask_predicted_flat

