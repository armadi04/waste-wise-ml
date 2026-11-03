"""MobileNetV2 transfer-learning model definition."""

import tensorflow as tf

from waste_ml.config import CLASSES, INPUT_SHAPE

BACKBONE_NAME = "mobilenetv2_backbone"
MODEL_NAME = "waste_classifier"


def build_model(
    num_classes: int = len(CLASSES),
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    dropout_rate: float = 0.2,
    trainable_backbone: bool = False,
) -> tf.keras.Model:
    """Build a MobileNetV2 classifier with a softmax classification head."""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        name=BACKBONE_NAME,
    )
    backbone.trainable = trainable_backbone

    inputs = tf.keras.Input(shape=input_shape, name="image")
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    dropped = tf.keras.layers.Dropout(dropout_rate, name="dropout")(pooled)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(dropped)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    return model.get_layer(BACKBONE_NAME)


def unfreeze_backbone(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze the last N backbone layers while keeping BatchNorm frozen."""
    backbone = get_backbone(model)
    backbone.trainable = True

    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

    for layer in backbone.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
