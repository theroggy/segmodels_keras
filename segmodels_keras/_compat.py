from keras import __version__ as keras_version
from packaging import version

KERAS_GTE_3 = version.parse(keras_version) >= version.parse("3.0.0")
