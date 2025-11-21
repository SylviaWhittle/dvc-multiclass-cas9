"""Scripts for handling files."""

import re

from pathlib import Path

def _get_index_from_name(
    file: Path
) -> int:
    """Get the index of a file from the filename"""
    regex_match = re.search(r"\d+", file.name)
    if not regex_match:
        raise ValueError(f"No numeric index found for file {file}")
    return int(regex_match.group())


def get_image_and_mask_indexes(
    data_dir: Path,
    file_ext: str = ".npy"
) -> tuple[list[int], list[int]]:
    """Get all the file indexes in a directory matching image_x and mask_x"""
    image_indexes = [_get_index_from_name(file) for file in data_dir.glob(f"image_*{file_ext}")]
    mask_indexes = [_get_index_from_name(file) for file in data_dir.glob(f"mask_*{file_ext}")]

    return image_indexes, mask_indexes