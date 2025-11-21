"""Scripts for plotting."""

from loguru import logger
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def plot_image_mask_prediction(
    image: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    mask_predicted: npt.NDArray[np.float64],
    mask_predicted_binary: npt.NDArray[np.bool_],
    mask_predicted_flat: npt.NDArray[np.float64],
    mask_predicted_flat_discrete: npt.NDArray[np.float64],
    plot_all_channels_separately: bool = False,
    max_num_cols: int = 4,
) -> plt.Figure:
    """Plot the image, mask and predicted mask."""
    # Plot the image, mask and predicted mask and log it
    # Plot the image, mask and predicted mask and log it
    # Plot sequentially with wrapping
    num_output_channels = mask.shape[2]
    num_input_channels = image.shape[2]
    # Total plots = number input channels + number output channels (for GT) + number output channels
    # (predicted) *2 (for raw and thresholded) + 1 (for combined predicted) + 1 (for combined predicted binary)
    # + 1 (for combined GT mask)
    total_plots = num_input_channels + 3
    if plot_all_channels_separately:
        total_plots += num_output_channels * 3  # for GT and predicted channels separately
    logger.info(f"Evaluate: Total plots to make: {total_plots}")
    num_rows = total_plots // max_num_cols + int(total_plots % max_num_cols > 0)
    num_cols = min(total_plots, max_num_cols)
    logger.info(f"Evaluate: Creating subplot with {num_rows} rows and {max_num_cols} columns.")
    scale = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(scale * num_cols, scale * num_rows))
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
    # Plot combined ground truth mask
    combined_gt_mask = np.zeros_like(mask[:, :, 0]).astype(np.float64)
    for i in range(num_output_channels):
        combined_gt_mask[mask[:, :, i] > 0] = i
    ax = axes.flatten()[plot_index]
    ax.imshow(combined_gt_mask, cmap="Grays")
    ax.set_title("Combined GT Mask")
    plot_index += 1
    # Plot combined predicted masks
    ax = axes.flatten()[plot_index]
    ax.imshow(mask_predicted_flat, cmap="viridis")
    ax.set_title("Combined Pred Mask")
    plot_index += 1
    ax = axes.flatten()[plot_index]
    ax.imshow(mask_predicted_flat_discrete, cmap="viridis")
    ax.set_title("Combined Pred Mask")
    plot_index += 1
    if plot_all_channels_separately:
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
    return fig