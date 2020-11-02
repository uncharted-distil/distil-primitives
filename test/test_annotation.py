import unittest
import distil.primitives.utils as utils

class AnnotationPrimitiveTestCase(unittest.TestCase):

    @utils.timed
    def square_it(self, x: int) -> int:
        return x*x

    def test_basic(self) -> None:
        squared = self.square_it(10)
        self.assertEqual(squared, 100)
