import keras
from packaging import version

KERAS_GTE_3 = version.parse(keras.__version__) >= version.parse("3.0.0")
