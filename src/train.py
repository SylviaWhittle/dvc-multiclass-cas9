"""Scripts for training the model."""

import re
from pathlib import Path

from loguru import logger

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from ruamel.yaml import YAML

from augmentation import flip_and_rotate, zoom_and_shift
from unet import unet_model, get_loss_function, get_metric_functions
from preprocess import preprocess_image, preprocess_mask
from plotting import plot_image_mask_prediction
from predict import predict_mask

yaml = YAML(typ="safe")

def image_data_generator(
    data_dir: Path,
    image_indexes: npt.NDArray[np.int64],
    output_channels: int,
    batch_size: int,
    model_image_size: tuple[int, int],
    norm_upper_bound: float,
    norm_lower_bound: float,
    filter_channels: list[str],
    hessian_component: str,
    hessian_sigma: int,
):
    """Generate batches of images and ground truth masks."""

    while True:
        # Select files for the batch
        batch_indexes = np.random.choice(a=image_indexes, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Load images and ground truth
        for index in batch_indexes:
            # Load the image and ground truth
            image = np.load(data_dir / f"image_{index}.npy")
            ground_truth = np.load(data_dir / f"mask_{index}.npy")

            # Augment images and masks
            image, ground_truth = zoom_and_shift(image, ground_truth)
            image, ground_truth = flip_and_rotate(image, ground_truth)

            # Preprocess the image and mask
            image = preprocess_image(
                image=image,
                model_image_size=model_image_size,
                norm_lower_bound=norm_lower_bound,
                norm_upper_bound=norm_upper_bound,
                filter_channels=filter_channels,
                hessian_component=hessian_component,
                hessian_sigma=hessian_sigma,
            )
            ground_truth = preprocess_mask(
                mask=ground_truth,
                model_image_size=model_image_size,
                output_channels=output_channels,
            )

            # Add the image and ground truth to the batch
            batch_input.append(image)
            batch_output.append(ground_truth)

        # Convert the lists to numpy arrays
        batch_x = np.array(batch_input).astype(np.float32)
        batch_y = np.array(batch_output).astype(np.float32)

        # logger.info(f"Batch x shape: {batch_x.shape}")
        # logger.info(f"Batch y shape: {batch_y.shape}")

        yield (batch_x, batch_y)

def _extract_index(file_path):
    match = re.search(r"\d+", file_path.name)
    if match is None:
        raise ValueError(f"Could not extract index from filename: {file_path.name}")
    return int(match.group())

def train_model(
    random_seed: int,
    train_data_dir: Path,
    model_save_dir: Path,
    model_image_size: tuple[int, int],
    output_channels: int,
    activation_function: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    norm_upper_bound: float,
    norm_lower_bound: int,
    filter_channels: list[str],
    hessian_component: str,
    hessian_sigma: int,
    validation_split: float,
    loss_function: str,
    metrics: list[str] | None,
):
    """Train a model to segment images."""

    logger.info("Training: Setup")

    logger.info("Training: Parameters:")
    logger.info(f"|  Random seed: {random_seed}")
    logger.info(f"|  Train data directory: {train_data_dir}")
    logger.info(f"|  Model save directory: {model_save_dir}")
    logger.info(f"|  Model image size: {model_image_size}")
    logger.info(f"|  Output channels: {output_channels}")
    logger.info(f"|  Activation function: {activation_function}")
    logger.info(f"|  Learning rate: {learning_rate}")
    logger.info(f"|  Batch size: {batch_size}")
    logger.info(f"|  Epochs: {epochs}")
    logger.info(f"|  Normalisation upper bound: {norm_upper_bound}")
    logger.info(f"|  Normalisation lower bound: {norm_lower_bound}")
    logger.info(f"|  Test size: {validation_split}")
    logger.info(f"|  Loss function: {loss_function}")

    # Set the random seed
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    logger.info("Training: Loading data")
    # Find the indexes of all the image files in the format of image_<index>.npy
    image_indexes = [_extract_index(file) for file in train_data_dir.glob("image_*.npy")]
    mask_indexes = [_extract_index(file) for file in train_data_dir.glob("mask_*.npy")]

    # Check that the image and mask indexes are the same
    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    # Train test split
    train_indexes, validation_indexes = train_test_split(
        image_indexes, test_size=validation_split, random_state=random_seed
    )
    logger.info(f"Training on {len(train_indexes)} images | validating on {len(validation_indexes)} images.")

    # Create an image data generator
    logger.info("Training: Creating data generators")

    train_generator = image_data_generator(
        data_dir=train_data_dir,
        image_indexes=train_indexes,
        batch_size=batch_size,
        output_channels=output_channels,
        model_image_size=model_image_size,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
        filter_channels=filter_channels,
        hessian_component=hessian_component,
        hessian_sigma=hessian_sigma,
    )

    validation_generator = image_data_generator(
        data_dir=train_data_dir,
        image_indexes=validation_indexes,
        batch_size=batch_size,
        output_channels=output_channels,
        model_image_size=model_image_size,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
        filter_channels=filter_channels,
        hessian_component=hessian_component,
        hessian_sigma=hessian_sigma,
    )

    # Load the model
    logger.info("Training: Loading model")
    model = unet_model(
        image_height=model_image_size[0],
        image_width=model_image_size[1],
        image_channels=len(filter_channels),
        output_channels=output_channels,
        learning_rate=learning_rate,
        activation_function=activation_function,
        loss_function=get_loss_function(loss_function),
        metrics=get_metric_functions(metrics),
    )

    steps_per_epoch = len(train_indexes) // batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # At the end of each epoch, DVCLive will log the metrics
    logger.info("Using DVCLive to log the metrics.")
    with Live("results/train") as live:

        logger.info("Training the model.")
        _history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch,
            verbose=1,
            callbacks=[DVCLiveCallback(live=live)],
        )

        logger.info("Training: Finished training.")

        logger.info(f"Training: Saving model to {model_save_dir}")
        model.save(Path(model_save_dir) / "unet_model.keras")
        live.log_artifact(
            str(Path(model_save_dir) / "unet_model.keras"),
            type="model",
            name="unet_model",
            desc="unet-type DL model.",
            labels=["cv", "segmentation"],
        )
        logger.info("Training: Finished.")

        # Show result on a few training images from the train generator
        logger.info("Training: Evaluating a few training images.")
        for batch_x, batch_y in train_generator:
            # Iterate over a batch of images
            for i in range(min(3, batch_size)):
                image = batch_x[i]
                mask = batch_y[i]

                # Add the batch dimension
                input_image = np.expand_dims(image, axis=0)

                # Predict the mask (without preprocessing as the generator already did that)
                predicted_mask, predicted_mask_binary, mask_predicted_flat_discrete, mask_predicted_flat = predict_mask(
                    model=model,
                    image=input_image,
                    norm_lower_bound=norm_lower_bound,
                    norm_upper_bound=norm_upper_bound,
                    filter_channels=filter_channels,
                    hessian_component=hessian_component,
                    hessian_sigma=hessian_sigma,
                    do_image_preprocessing=False,
                )
                fig = plot_image_mask_prediction(
                    image=image,
                    mask=mask,
                    mask_predicted=predicted_mask,
                    mask_predicted_binary=predicted_mask_binary,
                    mask_predicted_flat_discrete=mask_predicted_flat_discrete,
                    mask_predicted_flat=mask_predicted_flat,
                )
                live.log_image(f"train_image_plot_{i}.png", fig)

            break # stop after one batch


if __name__ == "__main__":
    logger.info("Train: Loading the parameters from the params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r", encoding="utf-8") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        train_params = all_params["train"]

    logger.info("Train: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    train_data_path = Path(train_params["train_data_dir"])
    model_save_path = Path(train_params["model_save_dir"])

    # Train the model
    train_model(
        random_seed=base_params["random_seed"],
        train_data_dir=train_data_path,
        model_save_dir=model_save_path,
        model_image_size=(base_params["model_image_size"], base_params["model_image_size"]),
        output_channels=base_params["output_channels"],
        activation_function=train_params["activation_function"],
        learning_rate=train_params["learning_rate"],
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        norm_upper_bound=base_params["norm_upper_bound"],
        norm_lower_bound=base_params["norm_lower_bound"],
        filter_channels=base_params["filter_channels"],
        hessian_component=base_params["hessian_component"],
        hessian_sigma=base_params["hessian_sigma"],
        validation_split=train_params["validation_split"],
        loss_function=base_params["loss_function"],
        metrics=base_params["metrics"],
    )
