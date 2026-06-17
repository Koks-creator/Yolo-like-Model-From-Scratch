from typing import Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numpy sigmoid, stabilna numerycznie."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def conv_block(x, filters: int, kernel_size: int = 3, strides: int = 1):
    """Conv -> BN -> LeakyReLU."""
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    return x

def residual_block(x, filters: int):
    """
    Darknet-style residual block: 1x1 redukcja -> 3x3 ekspansja -> skip.
    Wejście i wyjście mają `filters` kanałów (skip wymaga zgodności).
    """
    shortcut = x
    x = conv_block(x, filters // 2, kernel_size=1)
    x = conv_block(x, filters, kernel_size=3)
    return layers.Add()([shortcut, x])

def build_detector(img_size: int,
                   num_classes: int,
                   num_boxes: int,
                   width_multiplier: float = 0.5,
                   num_residual: Tuple[int, int, int, int, int] = (1, 2, 4, 4, 2)):
    """
    Darknet-style residual block: 1x1 reduction -> 3x3 expansion -> skip.
    The input and output have `filters` channels (skip requires compatibility).
    """
    def w(n):
        scaled = int(n * width_multiplier)
        return max(8, (scaled + 4) // 8 * 8)

    out_channels = num_boxes * 5 + num_classes

    inputs = layers.Input(shape=(img_size, img_size, 3), name='image')

    # ============================================================
    #                  BACKBONE (Darknet-lite)
    # ============================================================
    # Stem
    x = conv_block(inputs, w(32))

    # Stage 1: 416 -> 208
    x = conv_block(x, w(64), strides=2)
    for _ in range(num_residual[0]):
        x = residual_block(x, w(64))

    # Stage 2: 208 -> 104
    x = conv_block(x, w(128), strides=2)
    for _ in range(num_residual[1]):
        x = residual_block(x, w(128))

    # Stage 3: 104 -> 52 - route to scale 52x52 (small objects)
    x = conv_block(x, w(256), strides=2)
    for _ in range(num_residual[2]):
        x = residual_block(x, w(256))
    feat_52 = x

    # Stage 4: 52 -> 26 - route to scale 26x26 (medium objects)
    x = conv_block(x, w(512), strides=2)
    for _ in range(num_residual[3]):
        x = residual_block(x, w(512))
    feat_26 = x

    # Stage 5: 26 -> 13 - route to scale 13x13 (big objects)
    x = conv_block(x, w(1024), strides=2)
    for _ in range(num_residual[4]):
        x = residual_block(x, w(1024))
    feat_13 = x

    # ============================================================
    #              DETECTION HEADS z FPN (top-down)
    # ============================================================
    # --- Scale 13x13 (large objs) ---
    x = conv_block(feat_13, w(512), kernel_size=1)
    x = conv_block(x, w(1024))
    x = conv_block(x, w(512), kernel_size=1)
    branch_13 = x
    head = conv_block(x, w(1024))
    output_13 = layers.Conv2D(out_channels, 1, name='output_13')(head)

    # --- Upsample 13 -> 26, connect with feat_26 ---
    x = conv_block(branch_13, w(256), kernel_size=1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, feat_26])

    # --- Scale 26x26 (medium objects) ---
    x = conv_block(x, w(256), kernel_size=1)
    x = conv_block(x, w(512))
    x = conv_block(x, w(256), kernel_size=1)
    branch_26 = x
    head = conv_block(x, w(512))
    output_26 = layers.Conv2D(out_channels, 1, name='output_26')(head)

    # --- Upsample 26 -> 52, connect z feat_52 ---
    x = conv_block(branch_26, w(128), kernel_size=1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, feat_52])

    # --- Scale 52x52 (small objects) ---
    x = conv_block(x, w(128), kernel_size=1)
    x = conv_block(x, w(256))
    x = conv_block(x, w(128), kernel_size=1)
    head = conv_block(x, w(256))
    output_52 = layers.Conv2D(out_channels, 1, name='output_52')(head)

    # [13, 26, 52] — large, medium, small
    return Model(inputs, [output_13, output_26, output_52],
                 name=f'yolo_multiscale_{width_multiplier}')

def build_detector_anchor(
    img_size: int,
    num_classes: int,
    num_anchors: int,
    width_multiplier: float = 0.5,
    num_residual: tuple[int, ...] = (1, 2, 4, 4, 2),
) -> Model:
    """
    Multi-scale detector with anchor boxes.

    Returns a Keras model with 3 outputs (a list of tensors).
    Each output: (batch, S, S, num_anchors, 5 + num_classes).

    width_multiplier: channel scale (as before)
    num_residual: residual blocks per backbone stage. (1,2,4,4,2) = Darknet-lite.
    num_anchors: anchors per cell per scale. Must match len(ANCHORS_PER_SCALE[i]).
    """
    def w(n: int) -> int:
        """Scales the number of filters, rounding to multiples of 8 (TPU/GPU-friendly)"""
        scaled: int = int(n * width_multiplier)
        return max(8, (scaled + 4) // 8 * 8)

    per_anchor_channels: int = 5 + num_classes
    total_channels_per_cell: int = num_anchors * per_anchor_channels  # 3 × 6 = 18

    inputs: tf.Tensor = layers.Input(shape=(img_size, img_size, 3), name='image')

    # ============================================================
    #              BACKBONE (Darknet-lite)
    # ============================================================
    # Stem
    x: tf.Tensor = conv_block(inputs, w(32))

    # Stage 1: 416 → 208
    x = conv_block(x, w(64), strides=2)
    for _ in range(num_residual[0]):
        x = residual_block(x, w(64))

    # Stage 2: 208 → 104
    x = conv_block(x, w(128), strides=2)
    for _ in range(num_residual[1]):
        x = residual_block(x, w(128))

    # Stage 3: 104 → 52   ← route for 52×52 scale (small objs)
    x = conv_block(x, w(256), strides=2)
    for _ in range(num_residual[2]):
        x = residual_block(x, w(256))
    feat_52: tf.Tensor = x

    # Stage 4: 52 → 26    ← route for 26×26 scale (medium objs)
    x = conv_block(x, w(512), strides=2)
    for _ in range(num_residual[3]):
        x = residual_block(x, w(512))
    feat_26: tf.Tensor = x

    # Stage 5: 26 → 13    ← route for 13×13 scale (large objs)
    x = conv_block(x, w(1024), strides=2)
    for _ in range(num_residual[4]):
        x = residual_block(x, w(1024))
    feat_13: tf.Tensor = x

    # ============================================================
    #              DETECTION HEADS witj FPN (top-down)
    # ============================================================
    # --- Scale 13×13 (large) ---
    x = conv_block(feat_13, w(512), kernel_size=1)
    x = conv_block(x, w(1024))
    x = conv_block(x, w(512), kernel_size=1)
    branch_13: tf.Tensor = x 
    head: tf.Tensor = conv_block(x, w(1024))
    raw_13: tf.Tensor = layers.Conv2D(
        total_channels_per_cell, 1, name='output_13_flat',
    )(head)
    # Reshape (B, 13, 13, A * (5+C)) → (B, 13, 13, A, 5+C)
    output_13: tf.Tensor = layers.Reshape(
        (13, 13, num_anchors, per_anchor_channels),
        name='output_13',
    )(raw_13)

    # --- Upsample 13 → 26, connect with feat_26 ---
    x = conv_block(branch_13, w(256), kernel_size=1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, feat_26])

    # --- Scale 26×26 (medium) ---
    x = conv_block(x, w(256), kernel_size=1)
    x = conv_block(x, w(512))
    x = conv_block(x, w(256), kernel_size=1)
    branch_26: tf.Tensor = x
    head = conv_block(x, w(512))
    raw_26: tf.Tensor = layers.Conv2D(
        total_channels_per_cell, 1, name='output_26_flat',
    )(head)
    output_26: tf.Tensor = layers.Reshape(
        (26, 26, num_anchors, per_anchor_channels),
        name='output_26',
    )(raw_26)

    # --- Upsample 26 → 52, connect feat_52 ---
    x = conv_block(branch_26, w(128), kernel_size=1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, feat_52])

    # --- Scale 52×52 (small) ---
    x = conv_block(x, w(128), kernel_size=1)
    x = conv_block(x, w(256))
    x = conv_block(x, w(128), kernel_size=1)
    head = conv_block(x, w(256))
    raw_52: tf.Tensor = layers.Conv2D(
        total_channels_per_cell, 1, name='output_52_flat',
    )(head)
    output_52: tf.Tensor = layers.Reshape(
        (52, 52, num_anchors, per_anchor_channels),
        name='output_52',
    )(raw_52)

    return Model(
        inputs, [output_13, output_26, output_52],
        name=f'yolo_anchors_{width_multiplier}',
    )

