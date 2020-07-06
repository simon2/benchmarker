import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchVgg16Test(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--framework=pytorch",
            "--problem=ssd",
            "--problem_size=4",
            "--batch_size=2",
            "--nb_epoch=1",
            "--mode=inference",
        ]

    def test_ssd(self):
        run(self.args)
