# pylint: disable=import-error
# pylint: disable=unused-import
"""A U-NET model for segmentation of atomic force microscopy image grains."""

from typing import Callable

from loguru import logger
import numpy as np
import numpy.typing as npt
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import Dice
from keras.metrics import IoU
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
from keras.utils import register_keras_serializable
import tensorflow as tf


@register_keras_serializable(package="custom_losses")
def multiclass_dice_loss_optionally_ignore_background(
    y_true,
    y_pred,
    ignore_background: bool,
    smooth: float = 1e-5,
    class_weights: list[float] | None = None,
) -> tf.Tensor:
    """Multiclass DICE loss function for masks of shape (batch, H, W, C) where C > 2."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Validate shapes
    if y_true.shape.rank != 4 or y_pred.shape.rank != 4:
        raise ValueError(f"Expected 4D tensors, got ranks {y_true.shape.rank} and {y_pred.shape.rank}")
    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError(f"Last dim (channels) mismatch: y_true={y_true.shape[-1]} y_pred={y_pred.shape[-1]}")

    # One-hot check
    max_per_pixel = tf.reduce_max(y_true, axis=-1)
    # Check by seeing if max per pixel is either 0 or 1 everywhere.
    one_hot_mask = tf.logical_or(tf.equal(max_per_pixel, 0.0), tf.equal(max_per_pixel, 1.0))
    tf.debugging.assert_equal(
        tf.reduce_all(one_hot_mask),
        tf.constant(True),
        message="y_true mask is not one-hot encoded along the channel dimension."
    )

    # Compute per-class dice
    axes = [1, 2] # spatial dimensions excluding batch and channel
    # compute intersection per class by multiplying each pixel of y_true and y_pred and summing over spatial dimensions
    # this calculates the number of pixels correctly predicted for each class
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes) # shape (batch, channel)
    sum_true = tf.reduce_sum(y_true, axis=axes) # shape (batch, channel)
    sum_pred = tf.reduce_sum(y_pred, axis=axes) # shape (batch, channel)
    # Calculate DICE per class
    dice_per_class = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth) # shape (batch, channel)
    
    # Don't average over batch, this will be handled by the class reduction.
    
    # Handle background exclusion optionally
    if ignore_background:
        # note: 1: selects all but background channel, : selects all batch slices.
        dice_per_class = dice_per_class[:, 1:] # shape (batch, channel-1)
        # also ignore background in class weights if provided
        if class_weights is not None:
            class_weights = class_weights[1:]

    # apply class weights (if provided)
    if class_weights is not None:
        # Create a tensor from the class weights since we can assume they all add to 1 and so don't need normalization
        weights = tf.convert_to_tensor(class_weights, dtype=tf.float32) # shape (channels)
        # normalise weights just in case, with small epsilon to avoid zero divisions.
        weights = weights / (tf.reduce_sum(weights) + 1e-12)
        # weighted sum over classes
        dice_loss_per_sample = tf.reduce_sum(dice_per_class * weights, axis=-1) # shape (batch)
    else:
        # Average over classes
        dice_loss_per_sample = tf.reduce_mean(dice_per_class, axis=-1) # shape (batch)

    return 1.0 - dice_loss_per_sample # convert from score to loss. shape (batch)


@register_keras_serializable(package="custom_losses")
def multiclass_dice_loss_ignore_background(y_true, y_pred, smooth: float = 1e-5) -> tf.Tensor:
    """Multiclass DICE loss ignoring background class."""
    return multiclass_dice_loss_optionally_ignore_background(y_true, y_pred, ignore_background=True, smooth=smooth)

@register_keras_serializable(package="custom_losses")
def multiclass_dice_loss_include_background(y_true, y_pred, smooth: float = 1e-5) -> tf.Tensor:
    """Multiclass DICE loss including background class."""
    return multiclass_dice_loss_optionally_ignore_background(y_true, y_pred, ignore_background=False, smooth=smooth)

@register_keras_serializable(package="custom_losses")
def dice_loss(y_true, y_pred, smooth=1e-5):
    """DICE loss function.

    Parameters
    ----------
    y_true : tf.Tensor
        True values.
    y_pred : tf.Tensor
        Predicted values.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    dice : tf.Tensor
        The DICE loss.
    """
    # Ensure floats not bool
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Check the dimensions are expected: (batch_size, H, W, C)
    if len(y_true.shape) != 4 or len(y_pred.shape) != 4:
        raise ValueError(
            f"Expected y_true and y_pred to have 4 dimensions," f"got {len(y_true.shape)} and {len(y_pred.shape)}"
        )

    # Flatten spatial+channel dims from a tensor of shape (batch_size, H, W, C) to (batch_size, H*W*C).
    # This allows us to compute DICE per sample in the batch and then average over the batch.
    y_true_flat = tf.reshape(y_true, shape=[tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, shape=[tf.shape(y_pred)[0], -1])

    # Per-sample DICE calculation
    # Axis=1 sums over the H*W*C dimension, leaving a tensor of shape (batch_size,)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    sum_true = tf.reduce_sum(y_true_flat, axis=1)
    sum_pred = tf.reduce_sum(y_pred_flat, axis=1)

    dice_per_sample = 1 - (2 * intersection + smooth) / (sum_true + sum_pred + smooth)
    return tf.reduce_mean(dice_per_sample)

@register_keras_serializable(package="custom_losses")
class ClassWeightedMulticlassDice(tf.keras.losses.Loss):
    """Class-weighted multiclass DICE loss function wrapper."""

    def __init__(
        self,
        class_weights: list[float] | None,
        ignore_background: bool,
        smooth: float = 1e-5,
        reduction: str = "sum_over_batch_size",
        name: str | None = None
    ):
        """Initialize the class-weighted multiclass DICE loss.
        
        Parameters
        ----------
        class_weights : list[float] | None
            Class weights to apply. Must sum to 1.0 and be as long as the number of classes (channels), or None for equal weighting.
        ignore_background : bool
            Whether to ignore the background class (assumed to be channel 0) in the loss calculation.
        smooth : float
            Smoothing factor to prevent division by zero.
        reduction : str
            Type of reduction to apply to loss. Options: "sum_over_batch_size", "sum", "none", None.
        name : str | None
            Name for the loss function.
        """
        name = name or f"class_weighted_multiclass_dice_{'ignore_bg' if ignore_background else 'include_bg'}"
        super().__init__(name=name, reduction=reduction)
        self.ignore_background = bool(ignore_background)
        self.class_weights = class_weights
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        return multiclass_dice_loss_optionally_ignore_background(
            y_true,
            y_pred,
            ignore_background=self.ignore_background,
            smooth=self.smooth,
            class_weights=self.class_weights
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "ignore_background": self.ignore_background,
            "class_weights": list(self.class_weights) if self.class_weights is not None else None,
            "smooth": self.smooth,
        })
        return config


# IoU Loss
@register_keras_serializable(package="custom_losses")
def iou_loss(y_true, y_pred, smooth=1e-5):
    """Intersection over Union loss function.

    Parameters
    ----------
    y_true : tf.Tensor
        True values.
    y_pred : tf.Tensor
        Predicted values.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    iou : tf.Tensor
        The IoU loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Check the dimensions are expected: (batch_size, H, W, C)
    if len(y_true.shape) != 4 or len(y_pred.shape) != 4:
        raise ValueError(
            f"Expected y_true and y_pred to have 4 dimensions," f"got {len(y_true.shape)} and {len(y_pred.shape)}"
        )

    # Flatten spatial+channel dims from a tensor of shape (batch_size, H, W, C) to (batch_size, H*W*C).
    # This allows us to compute IoU per sample in the batch and then average over the batch.
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    # Per-sample IoU calculation
    # Axis=1 sums over the H*W*C dimension, leaving a tensor of shape (batch_size,)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    union = tf.reduce_sum(y_true_flat + y_pred_flat, axis=1) - intersection
    iou_per_sample = 1 - (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou_per_sample)


# BCE loss
@register_keras_serializable(package="custom_losses")
def bce_loss(y_true, y_pred, epsilon=1e-7):
    """Manual binary crossentropy loss function.

    Parameters
    ----------
    y_true : tf.Tensor
        True values.
    y_pred : tf.Tensor
        Predicted values.
    epsilon : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    bce : tf.Tensor
        The binary crossentropy loss.
    """
    # Ensure the tensors are of the same shape
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape[-1] == 1 else y_true
    y_pred = tf.squeeze(y_pred, axis=-1) if y_pred.shape[-1] == 1 else y_pred
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)  # To ensure no log(0) occurs
    bce = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return bce


LOSS_REGISTRY = {
    "dice_loss": dice_loss,
    "iou_loss": iou_loss,
    "binary_crossentropy": "binary_crossentropy",
    "keras_dice": Dice(),
    "multiclass_dice_ignore_background": multiclass_dice_loss_ignore_background,
    "multiclass_dice_include_background": multiclass_dice_loss_include_background,
    "categorical_crossentropy": "categorical_crossentropy",
}
METRIC_REGISTRY = {
    "dice_loss": dice_loss,
    "iou_loss": iou_loss,
    "binary_crossentropy": "binary_crossentropy",
    "accuracy": "accuracy",
    # num_classes=2 for binary. target_class_ids=[0, 1] to compute IoU for both classes.
    "keras_iou": IoU(num_classes=2, target_class_ids=[0, 1]),
}


def get_loss_function(loss_function: str, class_weights: list[float] | None = None) -> str | Callable:
    """Get the loss function based on the provided string.

    Parameters
    ----------
    loss_function : str
        The name of the loss function.
    class_weights : list[float] | None
        Class weights for multi-class, if applicable.

    Returns
    -------
    str | Callable
        The corresponding loss function.
    """
    if loss_function not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss function {loss_function} not recognized." f"Available options: {list(LOSS_REGISTRY.keys())}"
        )
    if loss_function == "multiclass_dice_ignore_background":
        return ClassWeightedMulticlassDice(
            class_weights=class_weights,
            ignore_background=True
        )
    if loss_function == "multiclass_dice_include_background":
        return ClassWeightedMulticlassDice(
            class_weights=class_weights,
            ignore_background=False
        )
    # Default: return the base loss function configuration from the registry.
    return LOSS_REGISTRY[loss_function]


def get_metric_functions(metrics: list[str] | None) -> list[Callable | str]:
    """Get the list of metric functions based on the provided list of strings.

    Parameters
    ----------
    metrics : list[str] | None
        The list of metric names.

    Returns
    -------
    list[Callable | str]
        The corresponding list of metric functions.
    """
    if metrics is None:
        return ["accuracy"]
    model_metrics = []
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} not recognized." f"Available options: {list(METRIC_REGISTRY.keys())}")
        model_metrics.append(METRIC_REGISTRY[metric])
    return model_metrics


def unet_model(
    image_height: int,
    image_width: int,
    image_channels: int,
    output_channels: int,
    learning_rate: float,
    activation_function: str,
    loss_function: str | Callable,
    metrics: list[str | Callable],
) -> Model:
    """U-NET model definition function.

    Parameters
    ----------
    image_height : int
        Image height.
    image_width : int
        Image width.
    image_channels : int
        Number of image channels.
    output_channels : int
        Number of output channels.
    learning_rate : float
        Learning rate for the Adam optimizer.
    activation_function : str
        Activation function to use in the model.
    loss_function : str
        Loss function to use in the model.
    metrics : list[str | Callable]
        List of metrics to use in the model.

    Returns
    -------
    model : keras.models.Model
        Single channel U-NET model for segmentation.
    """

    inputs = Input((image_height, image_width, image_channels))

    # Downsampling
    # Downsample with increasing numbers of filters to try to capture more complex features (first argument)
    # Dropout is used to try to prevent overfitting. Increase if overfitting happens.
    # Dropout increases deeper into the model to further help prevent overfitting.

    conv1 = Conv2D(
        16, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(
        16, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv1)
    pooled1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(
        32, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(
        32, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv2)
    pooled2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(
        64, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(
        64, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv3)
    pooled3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(
        128, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(
        128, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv4)
    pooled4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        256, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(
        256, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv5)

    # Upsampling
    # Conv2DTranspose is used as a sort of inverse convolution, to upsample the image
    # A concatenation is used to force context from the original image, providing information about what context a
    # feature stems from.

    up6 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(
        128, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(
        128, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv6)

    up7 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(
        64, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(
        64, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv7)

    up8 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(
        32, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(
        32, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv8)

    up9 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(
        16, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(
        16, kernel_size=(3, 3), activation=activation_function, kernel_initializer="he_normal", padding="same"
    )(conv9)

    # Make predictions of classes based on the culminated data
    final_layer_activation = "sigmoid" if output_channels == 1 else "softmax"
    logger.info(f"Final layer activation function set to {final_layer_activation} since there"
                f"are {output_channels} output channels.")
    outputs = Conv2D(output_channels, kernel_size=(1, 1), activation=final_layer_activation)(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # custom learning rate
    optimizer = Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    model.summary()

    return model
