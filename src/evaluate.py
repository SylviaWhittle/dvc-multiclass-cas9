"""Evaluate the trained model on the test set."""

from pathlib import Path
import re
from typing import Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from loguru import logger
from ruamel.yaml import YAML
from dvclive import Live
import matplotlib.pyplot as plt

from unet import LOSS_REGISTRY, METRIC_REGISTRY
from preprocess import preprocess_image, preprocess_mask

yaml = YAML(typ="safe")


def dice(
    mask_predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes=[0, 1],
    epsilon=1e-6,
):
    dice_list = []
    for c in classes:
        y_true = ground_truth == c
        y_pred = mask_predicted == c
        intersection = 2.0 * np.sum(y_true * y_pred)
        dice_score = intersection / (np.sum(y_true) + np.sum(y_pred) + epsilon)
        dice_list.append(dice_score)
    return np.mean(dice_list)


def evaluate(
    model: tf.keras.Model,
    data_dir: Path,
    model_image_size: Tuple[int, int],
    output_channels: int,
    norm_upper_bound: float,
    norm_lower_bound: float,
    filter_channels: list[str],
    hessian_component: str,
    hessian_sigma: int,
):
    """Evaluate the model on the test set.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model.
    data_dir : Path
        The directory containing the test images and masks as separate numpy files. Eg. image_0.npy and mask_0.npy.
    """
    # Load the test images and masks
    logger.info("Evaluate: Loading the test images and masks.")

    # Find the indexes of all the image files in the format of image_<index>.npy
    image_indexes = [int(re.search(r"\d+", file.name).group()) for file in data_dir.glob("image_*.npy")]
    mask_indexes = [int(re.search(r"\d+", file.name).group()) for file in data_dir.glob("mask_*.npy")]

    # Check that the image and mask indexes are the same
    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    with Live("results/evaluate") as live:
        dice_multi = 0.0

        for index in image_indexes:
            # Load the image and mask
            image = np.load(data_dir / f"image_{index}.npy")
            mask = np.load(data_dir / f"mask_{index}.npy")

            logger.info(f"Evaluate: Image index: {index}")
            logger.info(f"Evaluate: Image shape before reshape: {image.shape}")

            # Preprocess the image and mask
            image = preprocess_image(
                image=image,
                model_image_size=model_image_size,
                norm_upper_bound=norm_upper_bound,
                norm_lower_bound=norm_lower_bound,
                filter_channels=filter_channels,
                hessian_component=hessian_component,
                hessian_sigma=hessian_sigma,
            )

            mask = preprocess_mask(
                mask=mask,
                model_image_size=model_image_size,
                channels=output_channels,
            )

            logger.info(f"Evaluate: Image shape after reshape: {image.shape} | Mask shape: {mask.shape}")

            # Add the batch dimension
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            logger.info(f"Evaluate: Image shape after adding batch dimension: {image.shape} | Mask shape: {mask.shape}")

            # Predict the mask
            mask_predicted = model.predict(image)

            logger.info(f"Evaluate: Predicted mask shape: {mask_predicted.shape}")

            # Remove the batch dimension but keep the channel dimension as dice iterates over channels in case
            # of multi-class segmentation
            image = np.squeeze(image, axis=0)
            mask = np.squeeze(mask, axis=0)
            mask_predicted = np.squeeze(mask_predicted, axis=0)

            # Threshold the predicted mask to get a binary mask
            mask_predicted_binary = mask_predicted >= 0.5

            logger.info(
                f"Evaluate: Post-squeeze image shapes: Image: {image.shape} | Mask: {mask.shape} |"
                f"Predicted Mask: {mask_predicted.shape} Binary predicted mask:"
                f"{mask_predicted_binary.shape} ground truth: {mask.shape}"
            )

            # Calculate the DICE score
            dice_score = dice(mask_predicted, mask)
            dice_multi += dice_score / len(image_indexes)

            # Plot the image, mask and predicted mask and log it
            # Plot the image, mask and predicted mask and log it
            # Plot sequentially with wrapping
            max_num_cols = 3
            num_output_channels = mask.shape[2]
            num_input_channels = image.shape[2]
            # Total plots = number input channels + number output channels (for GT) + number output channels
            # (predicted) *2 (for raw and thresholded)
            total_plots = num_input_channels + num_output_channels + 2*num_output_channels
            logger.info(f"Evaluate: Total plots to make: {total_plots}")
            num_rows = total_plots // max_num_cols + int(total_plots % max_num_cols > 0)
            logger.info(f"Evaluate: Creating subplot with {num_rows} rows and {max_num_cols} columns.")
            fig, axes = plt.subplots(num_rows, min(total_plots, max_num_cols))
            plot_index = 0
            logger.info(
                f"Evaluate: Plotting {num_input_channels} input channels"
                "and {num_output_channels} output channels."
            )
            # Plot input channels
            for i in range(num_input_channels):
                ax = axes.flatten()[plot_index]
                ax.imshow(image[:, :, i], cmap="viridis")
                ax.set_title(f"Input C{i}")
                plot_index += 1
            # Plot ground truth mask channels
            for i in range(mask.shape[2]):
                ax = axes.flatten()[plot_index]
                ax.imshow(mask[:, :, i], cmap="binary")
                ax.set_title(f"GT Mask C{i}")
                plot_index += 1
            # Plot output channels both nonbinary and binary
            for i in range(num_output_channels):
                ax = axes.flatten()[plot_index]
                ax.imshow(mask_predicted[:, :, i], cmap="gray_r")
                ax.set_title(f"Pred Mask C{i}")
                plot_index += 1
                ax = axes.flatten()[plot_index]
                ax.imshow(mask_predicted_binary[:, :, i], cmap="binary")
                ax.set_title(f"Pred bin Mask C{i}")
                plot_index += 1
            plt.tight_layout()
            # plt.savefig(f"{plot_save_dir}/test_image_{index}.png")
            live.log_image(f"test_image_plot_{index}.png", fig)

        live.summary["dice_multi"] = dice_multi

if __name__ == "__main__":
    logger.info("Evaluate: Loading parameters from params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        evaluate_params = all_params["evaluate"]

    logger.info("Evaluate: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    model_path = Path(evaluate_params["model_path"])
    data_path = Path(evaluate_params["test_data_dir"])

    custom_objects = {}
    for name, obj in {**LOSS_REGISTRY, **METRIC_REGISTRY}.items():
        # Only add non-string objects to custom_objects
        if not isinstance(obj, str):
            custom_objects[name] = obj

    # Load the model
    logger.info("Evaluate: Loading the model.")
    loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Evaluate the model
    logger.info("Evaluate: Evaluating the model.")
    evaluate(
        model=loaded_model,
        data_dir=data_path,
        model_image_size=(base_params["model_image_size"], base_params["model_image_size"]),
        output_channels=base_params["output_channels"],
        norm_upper_bound=base_params["norm_upper_bound"],
        norm_lower_bound=base_params["norm_lower_bound"],
        filter_channels=base_params["filter_channels"],
        hessian_component=base_params["hessian_component"],
        hessian_sigma=base_params["hessian_sigma"],
    )
