# pylint: disable=import-error
# pylint: disable=unused-import
"""A U-NET model for segmentation of atomic force microscopy image grains."""

from typing import Callable

from loguru import logger

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
import tensorflow as tf

def multiclass_dice_loss_optionally_ignore_background(y_true, y_pred, ignore_background: bool, smooth: float = 1e-5) -> tf.Tensor:
    """Multiclass DICE loss function for masks of shape (batch, H, W, C) where C > 2."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Check that is one-hot encoded
    # Ensure dimensions expected
    if len(y_true.shape) != 4 or len(y_pred.shape) != 4:
        raise ValueError(
            f"Expected y_true and y_pred to have 4 dimensions (batch, H, W, C)," f"got {len(y_true.shape)} and {len(y_pred.shape)}"
        )
    if y_true.shape[-1] < 2:
        raise ValueError("y_true must be one-hot encoded with at least 2 channels for multiclass DICE loss.")
    
    # Compute per-class dice
    axes = [1, 2] # spatial dimensions excluding batch and channel
    # compute intersection per class by multiplying each pixel of y_true and y_pred and summing over spatial dimensions
    # this calculates the number of pixels correctly predicted for each class
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    sum_true = tf.reduce_sum(y_true, axis=axes)
    sum_pred = tf.reduce_sum(y_pred, axis=axes)

    dice_per_class = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth) # shape (batch, channel)
    assert len(dice_per_class.shape) == 2, f"Expected dice_per_class to have 2 dimensions (B, C), got {len(dice_per_class.shape)}"
    dice_per_class = tf.reduce_mean(dice_per_class, axis=0) # average over batch, shape (channel)

    if ignore_background:
        dice_per_class = dice_per_class[1:] # ignore background class (assumed to be channel 0)
    dice_loss = 1.0 - tf.reduce_mean(dice_per_class) # average over classes
    return dice_loss

def multiclass_dice_loss_ignore_background(y_true, y_pred, smooth: float = 1e-5) -> tf.Tensor:
    """Multiclass DICE loss ignoring background class."""
    return multiclass_dice_loss_optionally_ignore_background(y_true, y_pred, ignore_background=True, smooth=smooth)

def multiclass_dice_loss_include_background(y_true, y_pred, smooth: float = 1e-5) -> tf.Tensor:
    """Multiclass DICE loss including background class."""
    return multiclass_dice_loss_optionally_ignore_background(y_true, y_pred, ignore_background=False, smooth=smooth)


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


# IoU Loss
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
}
METRIC_REGISTRY = {
    "dice_loss": dice_loss,
    "iou_loss": iou_loss,
    "binary_crossentropy": "binary_crossentropy",
    "accuracy": "accuracy",
    # num_classes=2 for binary. target_class_ids=[0, 1] to compute IoU for both classes.
    "keras_iou": IoU(num_classes=2, target_class_ids=[0, 1]),
}


def get_loss_function(loss_function: str) -> str | Callable:
    """Get the loss function based on the provided string.

    Parameters
    ----------
    loss_function : str
        The name of the loss function.

    Returns
    -------
    str | Callable
        The corresponding loss function.
    """
    if loss_function not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss function {loss_function} not recognized." f"Available options: {list(LOSS_REGISTRY.keys())}"
        )
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
    print(type(model))
    # custom learning rate
    optimizer = Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    model.summary()

    return model
