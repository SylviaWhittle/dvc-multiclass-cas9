"""Scripts for plotting."""

from loguru import logger
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def plot_image_mask_prediction(
    image: npt.NDArray,
    mask: npt.NDArray,
    mask_predicted: npt.NDArray,
    mask_predicted_binary: npt.NDArray,
) -> plt.Figure:
    """Plot the image, mask and predicted mask."""
    # Plot the image, mask and predicted mask and log it
    # Plot the image, mask and predicted mask and log it
    # Plot sequentially with wrapping
    max_num_cols = 3
    num_output_channels = mask.shape[2]
    num_input_channels = image.shape[2]
    # Total plots = number input channels + number output channels (for GT) + number output channels
    # (predicted) *2 (for raw and thresholded) + 1 (for combined predicted) + 1 (for combined predicted binary)
    total_plots = num_input_channels + num_output_channels + 2*num_output_channels + 2
    logger.info(f"Evaluate: Total plots to make: {total_plots}")
    num_rows = total_plots // max_num_cols + int(total_plots % max_num_cols > 0)
    logger.info(f"Evaluate: Creating subplot with {num_rows} rows and {max_num_cols} columns.")
    # Combine the predicted masks into one image for visualisation, don't include the background channel ofc
    combined_predicted = np.sum(mask_predicted[:, :, 1:], axis=2)
    # for the binary predicted mask, choose the value of the channel with the highest confidence per pixel
    combined_predicted_binary = np.argmax(mask_predicted_binary[:, :, 1:], axis=2)
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
    # Plot combined predicted masks
    ax = axes.flatten()[plot_index]
    ax.imshow(combined_predicted, cmap="viridis")
    ax.set_title("Combined Pred Mask")
    plot_index += 1
    ax = axes.flatten()[plot_index]
    ax.imshow(combined_predicted_binary, cmap="viridis")
    ax.set_title("Combined Pred bin Mask")
    plot_index += 1
    plt.tight_layout()
    return fig