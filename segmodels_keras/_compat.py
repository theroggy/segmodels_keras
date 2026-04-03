import keras
from packaging import version

KERAS_GTE_3 = version.parse(keras.version()) >= version.parse("3.0.0")
