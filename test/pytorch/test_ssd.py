import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchVgg16Test(unittest.TestCase):
    def setUp(self):
        self.args = [
            "benchmarker",
            "--framework=pytorch",
            "--problem=ssd",
            "--problem_size=4",
            "--batch_size=2",
            "--mode=inference",
        ]

    def test_ssd(self):
        run_module(*self.args)
