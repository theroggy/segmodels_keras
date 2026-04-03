import keras
from packaging.version import parse

KERAS_GTE_3 = parse(keras.__version__) >= parse("3.0.0")
