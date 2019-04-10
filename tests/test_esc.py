import unittest
import kitchen20
import numpy as np
from numpy import testing
from kitchen20 import utils as U
from kitchen20 import esc


class TestESC(unittest.TestCase):
    def test_esc10_loading(self):
        """Test that esc50 gets properly loaded"""
        ESC10 = esc.ESC10
        self._test_dataset_load(ESC10)

    def test_esc50_loading(self):
        """Test that esc50 gets properly loaded"""
        ESC50 = esc.ESC50
        self._test_dataset_load(ESC50)

    def test_esc70_loading(self):
        """Test that esc10 gets properly loaded"""
        ESC70 = esc.ESC70
        self._test_dataset_load(ESC70)

    def test_kitchen20_loading(self):
        """Test that kitchen20 gets properly loaded"""
        Kitchen20 = esc.Kitchen20
        self._test_dataset_load(Kitchen20)

    def _test_dataset_load(self, Dataset, root=None):
        """Generic test over loading subclass of esc"""
        inputLength = 48000

        params = {}
        params['folds'] = [1]
        params['use_bc_learning'] = False
        params['overwrite'] = True
        params['audio_rate'] = 16000
        if root is not None:
            params['root'] = root

        train = Dataset(**params)
        testing.assert_equal(train[1], train[1])

        # Test 2, assert transformations are happening
        params['folds'] = [1, 2, 3, 4, 5]
        params['audio_rate'] = 44100
        params['transforms'] = [
            U.random_scale(1.25),  # Strong augment
            U.padding(inputLength // 2),  # Padding
            U.random_crop(inputLength),  # Random crop
            U.normalize(float(2 ** 16 / 2)),  # 16 bit signed
            U.random_flip()]

        train = Dataset(**params)

        self.assertRaises(AssertionError,
                          testing.assert_array_equal,
                          train[1], train[1])
        self.assertEqual(len(train.sounds),
                         40 * train.nClasses)


if __name__ == '__main__':
    unittest.main()
