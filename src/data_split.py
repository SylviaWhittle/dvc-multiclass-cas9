"""Scripts for splitting the data into training and test sets."""

from pathlib import Path
import shutil
import re

from sklearn.model_selection import train_test_split
from loguru import logger
from ruamel.yaml import YAML

from file_handling import get_image_and_mask_indexes

yaml = YAML(typ="safe")


def data_split(
    random_seed,
    test_split,
    data_dir,
    train_data_dir,
    test_data_dir,
):
    """Split the data into training and test sets."""

    # Create the directories if they don't exist
    train_data_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Data split: Discovering images and masks.")
    # Find the maximum index of images and ground truth masks
    num_images = len(list(data_dir.glob("image_*.npy")))
    num_masks = len(list(data_dir.glob("mask_*.npy")))
    if num_images != num_masks:
        raise ValueError("Different number of images and masks.")

    # Train test split
    image_indexes, mask_indexes = get_image_and_mask_indexes(data_dir=data_dir)

    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    logger.info(f"Data split: Found {len(image_indexes)} images and masks in {data_dir}.")
    train_indexes, test_indexes = train_test_split(image_indexes, test_size=test_split, random_state=random_seed)
    logger.info(f"Data split: Split into {len(train_indexes)} training images"
                f"and {len(test_indexes)} test images.")

    # Copy the images and masks to the train and test directories
    logger.info("Data split: Copying images and masks to train and test directories.")
    for index in train_indexes:
        shutil.copy(data_dir / f"image_{index}.npy", train_data_dir / f"image_{index}.npy")
        shutil.copy(data_dir / f"mask_{index}.npy", train_data_dir / f"mask_{index}.npy")
        # Copy the pngs too
        shutil.copy(data_dir / f"image_{index}.png", train_data_dir / f"image_{index}.png")
        shutil.copy(data_dir / f"mask_{index}.png", train_data_dir / f"mask_{index}.png")

    for index in test_indexes:
        shutil.copy(data_dir / f"image_{index}.npy", test_data_dir / f"image_{index}.npy")
        shutil.copy(data_dir / f"mask_{index}.npy", test_data_dir / f"mask_{index}.npy")
        # Copy the pngs too
        shutil.copy(data_dir / f"image_{index}.png", test_data_dir / f"image_{index}.png")
        shutil.copy(data_dir / f"mask_{index}.png", test_data_dir / f"mask_{index}.png")

    logger.info(f"Data split: Complete. Saved to {train_data_dir} and {test_data_dir} directories.")


if __name__ == "__main__":
    logger.info("Data split: Loading parameters from params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r", encoding="utf-8") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        data_split_params = all_params["data_split"]

    logger.info("Data split: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    data_path = Path(data_split_params["data_dir"])
    train_data_path = Path(data_split_params["train_data_dir"])
    test_data_path = Path(data_split_params["test_data_dir"])

    # Split the data
    data_split(
        random_seed=base_params["random_seed"],
        test_split=data_split_params["test_split"],
        data_dir=data_path,
        train_data_dir=train_data_path,
        test_data_dir=test_data_path,
    )
