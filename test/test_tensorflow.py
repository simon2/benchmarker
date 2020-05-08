import unittest
import logging
from .helpers import run_module
import os

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# vatai: I think loading TF here makes the tests a little faster :-/
import tensorflow as tf


class TensorflowTests(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=4",
            "--batch_size=2",
            "--epochs=1",
        ]

    def test_vgg16(self):
        run_module(self.name, "--problem=vgg16", *self.imgnet_args)

    def test_resnet50(self):
        run_module(self.name, "--problem=resnet50", *self.imgnet_args)
