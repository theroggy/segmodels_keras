import keras
from packaging.version import parse

keras_version = keras.__version__
KERAS_GTE_3 = parse(keras_version) >= parse("3.0.0")
